# api/resources/TradingSessionResource.py

from decimal import Decimal
import hashlib
import json
import logging
from django.http import HttpResponse

import contextlib

from django.shortcuts import get_object_or_404
from django.db import transaction
from django.http import JsonResponse
from django import http
from django.urls import re_path

from tastypie.resources import ModelResource
from tastypie.utils import trailing_slash
from tastypie.authorization import Authorization
from tastypie.exceptions import ImmediateHttpResponse

from api.authentication import CustomApiKeyAuthentication

from shop.models import (
    Study,
    StudyIndicator,
    StudyTradingPlan,
)
from shop.session_models import (
    TradingSession,
    SessionPotentialOrder,
    SessionStockData,
    SessionStockDataIndicatorValue,
)
from shop.services import calculateSession
from shop.utils.pricing import round_to_tick, decimals_from_tick

logger = logging.getLogger(__name__)



# ------------------------- Local helpers (scoped to this resource file) -------------------------

def _safe_params(si):
    val = getattr(si, "params", None) or getattr(si, "parameters", None) or {}
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return val or {}

def _indicator_code_name(si):
    ind = getattr(si, "indicator", None)
    code = getattr(ind, "code", None) or getattr(ind, "slug", None) or str(getattr(si, "indicator_id", ""))
    name = getattr(ind, "name", None) or "Indicator"
    return code, name

def _to_epoch_ms(value):
    if value is None:
        return value
    if isinstance(value, str) and value.isdigit():
        n = int(value)
        return n * 1000 if n < 1_000_000_000_000 else n
    if isinstance(value, int):
        return value * 1000 if value < 1_000_000_000_000 else value
    return value

def _q_price(p: Decimal, tick_size: Decimal) -> Decimal:
    """Round to nearest valid tick and return Decimal."""
    if not tick_size:
        return p
    # scale to integer ticks, round to nearest, then scale back
    ticks = (p / tick_size).quantize(Decimal('1'))
    return ticks * tick_size

def _get_satr_for_candle(session, candle):
    from django.http import HttpResponse
    study = getattr(session, "study", None)
    if not study or not getattr(study, "mainSatrIndicator_id", None):
        raise ImmediateHttpResponse(HttpResponse("Study.mainSatrIndicator must be set", status=422))

    from shop.session_models import SessionStockDataIndicatorValue
    iv = (SessionStockDataIndicatorValue.objects
          .filter(sessionStockDataItem=candle, studyIndicator_id=study.mainSatrIndicator_id)
          .only("value")
          .first())
    if not iv:
        raise ImmediateHttpResponse(HttpResponse("sATR value not found for this candle", status=422))

    import json
    try:
        parsed = json.loads(iv.value)
        if isinstance(parsed, dict) and "value" in parsed:
            return float(parsed["value"])
        return float(iv.value)
    except Exception:
        raise ImmediateHttpResponse(HttpResponse("sATR value parse error", status=422))

def _parse_indicator_value(iv):
        """
        Unified parser for SessionStockDataIndicatorValue.value:
        - plain numeric string "123.45"
        - or JSON like {"value": 123.45}
        Returns float or raises.
        """
        try:
            parsed = json.loads(iv.value)
            if isinstance(parsed, dict) and "value" in parsed:
                return float(parsed["value"])
            return float(parsed)
        except Exception:
            # Re-raise to keep behavior explicit; you can fall back to None if preferred
            raise

def _build_nn_input_raw(session, candle, *, direction, lp_offset, sl_atr, tp_mult,
                        limit_price, stop_price, take_profit_price):
    """
    Build nn_input_raw JSON snapshot for a single potential order preview
    (no DB writes, only reads).
    """
    study = getattr(session, "study", None)

    # ---- indicators dict ----
    indicators = {}
    iv_qs = (
        SessionStockDataIndicatorValue.objects
        .filter(sessionStockDataItem=candle)
        .select_related("studyIndicator", "studyIndicator__indicator")
    )

    for iv in iv_qs:
        # Try to resolve a stable code/name for the indicator
        code = None
        si = getattr(iv, "studyIndicator", None)
        if si is not None:
            # adjust these attribute names to your actual models
            code = getattr(si, "code", None)
            if code is None:
                indicator_obj = getattr(si, "indicator", None)
                code = getattr(indicator_obj, "code", None) if indicator_obj is not None else None

        if code is None:
            code = f"studyIndicator_{iv.studyIndicator_id}"

        value = _parse_indicator_value(iv)
        indicators[code] = value

    # ---- normalizers (names depend on your Study model) ----
    # TODO: adjust these attribute names to match your Study fields
    price_indicator_code = getattr(study, "priceNormalizerIndicatorCode", None) if study else None
    volume_indicator_code = getattr(study, "volumeNormalizerIndicatorCode", None) if study else None

    normalizers = {
        "price_indicator": price_indicator_code,
        "price_value": indicators.get(price_indicator_code) if price_indicator_code else None,
        "volume_indicator": volume_indicator_code,
        "volume_value": indicators.get(volume_indicator_code) if volume_indicator_code else None,
    }

    # ---- meta ----
    nn_meta = {
        "schema_version": 1,
        "session_id": session.id,
        "candle_id": candle.id,
        "timestamp_ms": candle.timestamp,
        "symbol": getattr(session, "symbol", None),
        "study_id": getattr(study, "id", None) if study else None,
        "trained_model_id": None,  # preview only; real SPO may set this later
    }

    # ---- order params ----
    order_params = {
        "direction": direction,
        "lpOffset": float(lp_offset),
        "slATR": float(sl_atr),
        "tp": int(tp_mult),
    }

    # ---- candle snapshot ----
    candle_snapshot = {
        "open": float(candle.open),
        "high": float(candle.high),
        "low": float(candle.low),
        "close": float(candle.close),
        "volume": int(candle.volume) if candle.volume is not None else None,
        "timestamp_ms": candle.timestamp,
    }

    # ---- derived prices ----
    derived_prices = {
        "limitPrice": float(limit_price),
        "stopPrice": float(stop_price),
        "takeProfitPrice": float(take_profit_price),
    }

    return {
        "meta": nn_meta,
        "order_params": order_params,
        "candle": candle_snapshot,
        "derived_prices": derived_prices,
        "indicators": indicators,
        "normalizers": normalizers,
    }

# ----------------------------------- TradingSessionResource -----------------------------------

class TradingSessionResource(ModelResource):
    class Meta:
        queryset = TradingSession.objects.all()
        resource_name = 'sessions'
        allowed_methods = ['get', 'post', 'delete']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
        always_return_data = True
        filtering = {
            'type': ['exact'],
            'state': ['exact'],
            'status': ['exact'],
            'session_id': ['exact'],
            'name': ['icontains'],
            'createdAt': ['range', 'gte', 'lte'],
            'sessionStart': ['range', 'gte', 'lte', 'exact'],
            'sessionEnd': ['range', 'gte', 'lte', 'exact'],
        }

    def prepend_urls(self):
        return [
            # GET /api/v1/sessions/<pk>/stockdata/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/stockdata%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('stockdata'),
                name="api_trading_session_stockdata",
            ),

            # GET /api/v1/sessions/<pk>/indicators/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\d+)/indicators%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('get_session_indicators'),
                name='api_trading_session_indicators',
            ),

            # GET /api/v1/sessions/<pk>/indicators/values/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\d+)/indicators/values%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('get_session_indicators_values'),
                name='api_trading_session_indicators_values',
            ),

            # GET /api/v1/sessions/<pk>/calculate/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/calculate%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('session_calculate'),
                name="api_trading_session_calculate",
            ),

            # POST /api/v1/sessions/<pk>/start/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/start%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('start_session'),
                name="api_trading_session_start",
            ),

            # POST /api/v1/sessions/<pk>/stop/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/stop%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('stop_session'),
                name="api_trading_session_stop",
            ),

            # POST /api/v1/sessions/<pk>/update_balance/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/update_balance%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('update_balance'),
                name="api_trading_session_update_balance",
            ),

            # POST /api/v1/sessions/<pk>/save_indicators_bulk/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/save_indicators_bulk%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('save_indicators_bulk'),
                name="api_trading_session_save_indicators_bulk",
            ),

            # POST /api/v1/sessions/<pk>/save_orders_bulk/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/save_orders_bulk%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('save_orders_bulk'),
                name="api_trading_session_save_orders_bulk",
            ),

            # POST /api/v1/sessions/<pk>/candles/<candle_id>/potential-orders:generate
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\d+)/candles/(?P<candle_id>\d+)/potential-orders:generate%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('generate_potential_orders'),
                name='api_trading_session_generate_potential_orders',
            ),

            # POST /api/v1/sessions/<pk>/candles:generate?timestamp=...
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\d+)/candles:generate%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('candles_generate'),
                name='api_trading_session_generate_potential_orders_by_ts',
            ),
        ]

    # ----------------------------- Basic data endpoints -----------------------------

    def stockdata(self, request, **kwargs):
        self.method_check(request, allowed=['get'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])

        qs = (
            SessionStockData.objects
            .filter(trading_session=session)
            .order_by('timestamp')
            .values('timestamp', 'open', 'high', 'low', 'close', 'volume')
        )
        rows = list(qs)
        if rows:
            return self.create_response(request, rows)

        # Fallback to JSON bag if you still keep it on the session
        payload = session.sessionStockData
        if payload and payload != "{}":
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict) and "objects" in parsed:
                    parsed = parsed["objects"]
                elif not isinstance(parsed, list):
                    parsed = []
                return self.create_response(request, parsed)
            except Exception:
                pass

        return self.create_response(request, [])

    def dehydrate(self, bundle):
        study = getattr(bundle.obj, 'study', None)
        if not study:
            bundle.data["study"] = None
            return bundle

        sis = (
            StudyIndicator.objects
            .filter(study_id=study.id)
            .select_related('indicator')
            .order_by('id')
        )

        indicators = []
        for si in sis:
            code, name = _indicator_code_name(si)
            indicators.append({
                "id": si.id,
                "code": code,
                "name": name,
                "params": _safe_params(si),
            })

        buf = json.dumps(indicators, sort_keys=True, ensure_ascii=False).encode("utf-8")
        indicators_hash = hashlib.sha256(buf).hexdigest()

        bundle.data["study"] = {
            "id": study.id,
            "ticker": getattr(study, "ticker", None),
            "timeFrame": getattr(study, "timeFrame", None),
            "indicators": indicators,
            "indicatorsHash": indicators_hash,
        }
        return bundle

    # ----------------------------- Indicators & values -----------------------------

    def get_session_indicators(self, request, **kwargs):
        """
        GET /api/v1/sessions/<pk>/indicators/
        Mirrors StudyResource.get_study_indicators (bare array, same fields).
        """
        self.method_check(request, allowed=['get'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession.objects.select_related('study'), pk=kwargs['pk'])
        study = getattr(session, 'study', None)
        if not study:
            return self.create_response(request, {"error_message":"Session has no linked Study"},
                            response_class=HttpResponse, status=404)

        indicators = (
            StudyIndicator.objects
            .filter(study=study)
            .select_related('indicator')
            .order_by('id')
        )

        data = []
        for indicator in indicators:
            ind = indicator.indicator  # may be None if not linked
            data.append({
                'id': indicator.id,
                'mask': indicator.mask,
                'indicator_id': ind.id if ind else None,
                'indicator_name': ind.name if ind else None,
                'indicator_function': ind.functionName if ind else None,
                'indicator_parameters': ind.parameters if ind else None,
                'parametersValue': indicator.parametersValue,
            })

        self.log_throttled_access(request)
        return JsonResponse(data, safe=False)

    def get_session_indicators_values(self, request, **kwargs):
        """
        GET /api/v1/sessions/<pk>/indicators/values/
        (Legacy alias shape kept identical to Study.get_study_indicators_values)
        Optional filters: studyIndicator_id, mask, mask__icontains, from_ts, to_ts, limit
        """
        self.method_check(request, allowed=['get'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])

        qs = (
            SessionStockDataIndicatorValue.objects
            .select_related('sessionStockDataItem', 'studyIndicator', 'studyIndicator__indicator')
            .filter(sessionStockDataItem__trading_session=session)
        )

        # filters
        study_indicator_id = request.GET.get('studyIndicator_id')
        mask = request.GET.get('mask')
        mask_icontains = request.GET.get('mask__icontains')
        from_ts = request.GET.get('from_ts')
        to_ts = request.GET.get('to_ts')
        try:
            limit = int(request.GET.get('limit', 2000))
        except ValueError:
            limit = 2000

        if study_indicator_id:
            qs = qs.filter(studyIndicator_id=study_indicator_id)
        if mask:
            qs = qs.filter(studyIndicator__mask=mask)
        if mask_icontains:
            qs = qs.filter(studyIndicator__mask__icontains=mask_icontains)
        if from_ts:
            qs = qs.filter(sessionStockDataItem__timestamp__gte=from_ts)
        if to_ts:
            qs = qs.filter(sessionStockDataItem__timestamp__lt=to_ts)

        qs = qs.order_by('sessionStockDataItem__timestamp', 'studyIndicator_id')[:limit]

        out = []
        for iv in qs:
            val_str = iv.value
            # Nicely format JSON numeric {"value": ...} to 3 decimals when possible
            try:
                parsed = json.loads(val_str) if isinstance(val_str, str) else val_str
                if isinstance(parsed, dict) and "value" in parsed and isinstance(parsed["value"], (int, float)):
                    parsed["value"] = round(float(parsed["value"]), 3)
                    val_str = json.dumps(parsed)
                elif isinstance(parsed, (int, float)):
                    val_str = str(round(float(parsed), 3))
            except Exception:
                try:
                    num = float(val_str)
                    val_str = str(round(num, 3))
                except Exception:
                    pass

            out.append({
                'id': iv.id,
                'stockDataItem_id': iv.sessionStockDataItem_id,
                'stockDataItem_timestamp': _to_epoch_ms(iv.sessionStockDataItem.timestamp),
                'studyIndicator_id': iv.studyIndicator_id,
                'value': val_str,
                'indicator_name': (
                    iv.studyIndicator.indicator.name
                    if iv.studyIndicator and iv.studyIndicator.indicator_id else None
                ),
                'indicator_mask': iv.studyIndicator.mask if iv.studyIndicator else None,
            })

        self.log_throttled_access(request)
        return self.create_response(request, out)

    # ----------------------------- Session lifecycle & bookkeeping -----------------------------

    def session_calculate(self, request, **kwargs):
        self.method_check(request, allowed=['get'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        result = calculateSession(session)

        self.log_throttled_access(request)
        return self.create_response(request, {'result': result})

    def start_session(self, request, **kwargs):
        logger.info("start_session called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        session.state = 'RUNNING'
        session.status = 'OK'
        session.save(update_fields=['state', 'status', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'state': session.state, 'status': session.status})

    def stop_session(self, request, **kwargs):
        logger.info("stop_session called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        session.state = 'PAUSED'
        session.save(update_fields=['state', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'state': session.state})

    def update_balance(self, request, **kwargs):
        logger.info("update_balance called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        try:
            data = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(http.HttpResponseBadRequest("Invalid JSON"))

        for f in [
            'InitialBalance', 'EquityNow', 'CashNow', 'BuyingPowerNow',
            'UnrealizedPnlNow', 'RealizedPnlTotal', 'MarginUsedNow', 'Currency'
        ]:
            if f in data:
                setattr(session, f, data[f])

        session.save()
        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'balance': {
            'InitialBalance': str(session.InitialBalance),
            'EquityNow': str(session.EquityNow),
            'CashNow': str(session.CashNow),
            'BuyingPowerNow': str(session.BuyingPowerNow),
            'UnrealizedPnlNow': str(session.UnrealizedPnlNow),
            'RealizedPnlTotal': str(session.RealizedPnlTotal),
            'MarginUsedNow': str(session.MarginUsedNow),
            'Currency': session.Currency,
        }})

    def save_indicators_bulk(self, request, **kwargs):
        logger.info("save_indicators_bulk called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        try:
            data = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(http.HttpResponseBadRequest("Invalid JSON"))

        indicators = data.get('indicators')
        if indicators is None or not isinstance(indicators, dict):
            raise ImmediateHttpResponse(http.HttpResponseBadRequest("Missing or invalid 'indicators'"))

        current = session.sessionIndicatorsValues or {}
        current.update(indicators)
        session.sessionIndicatorsValues = current
        session.save(update_fields=['sessionIndicatorsValues', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True})

    def save_orders_bulk(self, request, **kwargs):
        logger.info("save_orders_bulk called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        try:
            data = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(http.HttpResponseBadRequest("Invalid JSON"))

        orders = data.get('orders', [])
        if not isinstance(orders, list):
            raise ImmediateHttpResponse(http.HttpResponseBadRequest("Missing or invalid 'orders' array"))

        current = session.sessionOrders or {}
        current_items = current.get('items', [])
        current_items.extend(orders)
        current['items'] = current_items

        session.sessionOrders = current
        session.save(update_fields=['sessionOrders', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'added': len(orders)})

    # ----------------------------- Potential orders generation -----------------------------

    def _resolve_trading_plan_for_session(self, session):
        """
        Return normalized trading plan dict:
        {
            "lp_offsets":   [...],
            "stop_losses":  [...],
            "take_profits": [...]
        }
        Source: Session → studyTradingPlan → TradingPlan → tradingPlanParams (stringified JSON).
        """
        from django.http import HttpResponse
        import json

        # Require a Study for sanity (model consistency)
        study = getattr(session, "study", None)
        if not study:
            raise ImmediateHttpResponse(HttpResponse("Session has no linked Study", status=422))

        # ✅ Correct attribute name on TradingSession
        stp = getattr(session, "studyTradingPlan", None)
        if not stp or not getattr(stp, "tradingPlan", None):
            raise ImmediateHttpResponse(HttpResponse("Session has no TradingPlan attached", status=422))

        # Field name tolerance (tradingPlanParams vs TradingPlanParams)
        tp = stp.tradingPlan
        params_str = getattr(tp, "tradingPlanParams", None) or getattr(tp, "TradingPlanParams", None)
        if not params_str:
            raise ImmediateHttpResponse(HttpResponse("TradingPlanParams not set", status=422))

        try:
            raw = json.loads(params_str)
        except Exception:
            raise ImmediateHttpResponse(HttpResponse("TradingPlanParams is not valid JSON", status=422))

        # Normalize keys
        lp = raw.get("LPoffset", [])   or raw.get("lp_offsets", [])
        sl = raw.get("stopLoss", [])   or raw.get("stop_losses", [])
        tpv = raw.get("takeProfit", []) or raw.get("take_profits", [])

        if not lp or not sl or not tpv:
            raise ImmediateHttpResponse(HttpResponse(
                "TradingPlanParams must include non-empty LPoffset, stopLoss, takeProfit", status=422
            ))

        return {
            "lp_offsets":   lp,
            "stop_losses":  sl,
            "take_profits": tpv,
        }

    def _price_decimals_from_tick(self, tick_size):
        return None if not tick_size else decimals_from_tick(tick_size)
    
    def _resolve_market_meta(self, session):
        """
        Only what we need now:
        - tick_size from session (if absent -> no quantization)
        - price_decimals derived from tick_size
        """
        tick_size = getattr(session, "tick_size", None)
        tick_size = Decimal(str(tick_size)) if tick_size is not None else None
        price_decimals = self._price_decimals_from_tick(tick_size)
        return tick_size, price_decimals

    def _format_price(self, x: Decimal, decimals: int) -> float:
        q = Decimal('1').scaleb(-decimals) if decimals else None
        return float(x.quantize(q)) if q else float(x)

    def _generate_for_candle(self, session, candle):
        """
        Build BUY/SELL × lp_offsets × stop_losses × take_profits for a single candle.
        Returns a list of PotentialOrders as JSON-ready dicts. NO DB WRITES.
        """
        plan = self._resolve_trading_plan_for_session(session)

        tick_size, price_decimals = self._resolve_market_meta(session)
        close = Decimal(str(candle.close))
        satr  = Decimal(str(_get_satr_for_candle(session, candle)))

        def _fmt(d: Decimal) -> float:
            if price_decimals is None:
                return float(d)
            q = Decimal('1').scaleb(-price_decimals)
            return float(d.quantize(q))

        results = []
        for direction in ("BUY", "SELL"):
            for lpo in (Decimal(str(x)) for x in plan["lp_offsets"]):
                # raw entry for this lpo
                raw_entry = close - (lpo * satr) if direction == "BUY" else close + (lpo * satr)

                for slatr in (Decimal(str(x)) for x in plan["stop_losses"]):
                    # raw stop for this (entry, slatr)
                    raw_stop = raw_entry - (slatr * satr) if direction == "BUY" else raw_entry + (slatr * satr)

                    for tpmult in (Decimal(str(x)) for x in plan["take_profits"]):
                        # raw take for this (entry, slatr, tpmult)
                        raw_take = (
                            raw_entry + (slatr * satr) * tpmult if direction == "BUY"
                            else raw_entry - (slatr * satr) * tpmult
                        )

                        # quantize (snap to tick) without mutating the raw values
                        e = _q_price(raw_entry, tick_size) if tick_size else raw_entry
                        s = _q_price(raw_stop,  tick_size) if tick_size else raw_stop
                        t = _q_price(raw_take,  tick_size) if tick_size else raw_take

                        # build nn_input_raw snapshot for this combo
                        nn_input_raw = _build_nn_input_raw(
                            session,
                            candle,
                            direction=direction,
                            lp_offset=lpo,
                            sl_atr=slatr,
                            tp_mult=tpmult,
                            limit_price=e,
                            stop_price=s,
                            take_profit_price=t,
                        )

                        results.append({
                            "id": None,                        # preview only
                            "session_id": session.id,
                            "candle_id": candle.id,
                            "direction": direction,
                            "lpOffset": float(lpo),
                            "slATR": float(slatr),
                            "tp": int(tpmult),
                            "limitPrice": _fmt(e),
                            "stopPrice":  _fmt(s),
                            "takeProfitPrice": _fmt(t),
                            "nn_input_raw": nn_input_raw,
                        })

        results.sort(key=lambda r: (r["direction"], r["lpOffset"], r["slATR"], r["tp"]))
        return results

    def _get_satr_for_candle(session, candle):
        from django.http import HttpResponse
        study = getattr(session, "study", None)
        if not study or not getattr(study, "mainSatrIndicator_id", None):
            raise ImmediateHttpResponse(HttpResponse("Study.mainSatrIndicator must be set", status=422))

        from shop.session_models import SessionStockDataIndicatorValue
        iv = (SessionStockDataIndicatorValue.objects
            .filter(sessionStockDataItem=candle, studyIndicator_id=study.mainSatrIndicator_id)
            .only("value")
            .first())
        if not iv:
            raise ImmediateHttpResponse(HttpResponse("sATR value not found for this candle", status=422))

        import json
        try:
            parsed = json.loads(iv.value)
            if isinstance(parsed, dict) and "value" in parsed:
                return float(parsed["value"])
            return float(iv.value)
        except Exception:
            raise ImmediateHttpResponse(HttpResponse("sATR value parse error", status=422))

    # ----------------------------- Routes: generators -----------------------------

    def generate_potential_orders(self, request, **kwargs):
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        candle  = get_object_or_404(SessionStockData, pk=kwargs['candle_id'], trading_session=session)

        try:
            data = self._generate_for_candle(session, candle)  # no extra args
            return self.create_response(request, data)
        except ImmediateHttpResponse:
            raise
        except Exception as e:
            return self.create_response(request, {"error_message": str(e)}, response_class=HttpResponse, status=500)



    def candles_generate(self, request, **kwargs):
        """
        POST /api/v1/sessions/{id}/candles:generate/?timestamp=<epoch_ms>
        Returns PotentialOrders JSON ONLY (no DB writes), with prices quantized if tick_size exists.
        """
        # Basic guards
        try:
            session = TradingSession.objects.get(pk=kwargs["pk"])
        except TradingSession.DoesNotExist:
            return self.create_response(
                request,
                {"error_message": "Session not found"},
                response_class=HttpResponse,
                status=404,
            )

        ts_ms = request.GET.get("timestamp")
        if not ts_ms:
            return self.create_response(
                request,
                {"error_message": "timestamp is required (epoch ms)"},
                response_class=HttpResponse,
                status=400,
            )
        try:
            ts_ms = int(ts_ms)
        except ValueError:
            return self.create_response(
                request,
                {"error_message": "timestamp must be integer epoch ms"},
                response_class=HttpResponse,
                status=400,
            )
        # Find exact session candle
        try:
            candle = SessionStockData.objects.get(trading_session=session, timestamp=ts_ms)
        except SessionStockData.DoesNotExist:
            return self.create_response(
                request,
                {"error_message": "Candle not found for timestamp"},
                response_class=HttpResponse,
                status=404,
            )

        try:
            data = self._generate_for_candle(session, candle)
            # 200 OK, preview payload
            return self.create_response(request, data)
        except ImmediateHttpResponse:
            # passthrough for our 422-type short-circuits
            raise
        except Exception as e:
            return self.create_response(
                request,
                {"error_message": str(e)},
                response_class=HttpResponse,
                status=500,
            )


