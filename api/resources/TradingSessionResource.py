# api/resources/TradingSessionResource.py

from decimal import Decimal
import hashlib
import json
import logging

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


def _normalize_plan_keys(plan: dict) -> dict:
    """
    Canonical keys: lp_offsets, stop_losses, take_profits
    Accept legacy: LPoffset, stopLoss, takeProfit
    """
    if not isinstance(plan, dict):
        return {}
    mapped = {
        "lp_offsets": plan.get("lp_offsets", plan.get("LPoffset")),
        "stop_losses": plan.get("stop_losses", plan.get("stopLoss")),
        "take_profits": plan.get("take_profits", plan.get("takeProfit")),
    }
    out = {}
    for k, v in mapped.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out[k] = [float(x) for x in v]
        else:
            try:
                out[k] = [float(v)]
            except Exception:
                pass
    return out


def _merge_plan(base: dict, override: dict) -> dict:
    base = base or {}
    override = override or {}
    return {
        "lp_offsets": override.get("lp_offsets", base.get("lp_offsets")),
        "stop_losses": override.get("stop_losses", base.get("stop_losses")),
        "take_profits": override.get("take_profits", base.get("take_profits")),
    }


def _validate_plan(plan: dict):
    for k in ("lp_offsets", "stop_losses", "take_profits"):
        if k not in plan:
            raise ValueError(f"trading_plan_invalid_keys: missing '{k}'")
        if not isinstance(plan[k], list) or len(plan[k]) == 0:
            raise ValueError(f"trading_plan_empty_values: '{k}' is empty")


def _extract_plan_from_tradingplan_obj(tp_obj):
    """
    Study -> TradingPlan (separate model). Its JSON lives as a STRING field.
    Try common attribute names and parse to dict with canonical keys.
    """
    if not tp_obj:
        return {}
    candidates = ("plan", "json", "plan_json", "planJson",
                  "data", "content", "body", "value", "parameters", "config", "tradingPlanParams")
    raw = None
    for attr in candidates:
        if hasattr(tp_obj, attr):
            raw = getattr(tp_obj, attr)
            if raw:
                break
    if not raw:
        return {}
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        return _normalize_plan_keys(raw)
    except Exception:
        return {}


def _resolve_trading_plan_for_session(session, request_body: dict):
    """
    Priority:
      1) request overrides (partial ok)
      2) session.plan_override (if you keep such a field; optional)
      3) study.tradingPlan (FK -> TradingPlan) JSON string
    """
    req_norm = _normalize_plan_keys(request_body or {})
    request_has_any = any(k in req_norm for k in ("lp_offsets", "stop_losses", "take_profits"))

    # Optional per-session override JSON (if model has it)
    session_norm = {}
    if hasattr(session, "plan_override") and session.plan_override:
        try:
            session_norm = _normalize_plan_keys(session.plan_override)
        except Exception:
            session_norm = {}

    # From Study → TradingPlan via StudyTradingPlan or direct FK
    study_norm = {}
    study = getattr(session, "study", None)
    if study:
        tp_obj = None
        # via StudyTradingPlan link (first one)
        link = StudyTradingPlan.objects.filter(study=study).select_related("tradingPlan").first()
        if link and link.tradingPlan:
            tp_obj = link.tradingPlan
        # or via direct FK (if present in your Study model)
        if not tp_obj and hasattr(study, "tradingPlan") and study.tradingPlan:
            tp_obj = study.tradingPlan
        study_norm = _extract_plan_from_tradingplan_obj(tp_obj)

    if request_has_any:
        base = _merge_plan(study_norm, session_norm)
        plan_used = _merge_plan(base, req_norm)
        src = "request"
    elif session_norm:
        plan_used = _merge_plan(study_norm, session_norm)
        src = "session"
    elif study_norm:
        plan_used = study_norm
        src = "study"
    else:
        raise ValueError("trading_plan_not_found")

    _validate_plan(plan_used)
    return plan_used, src, request_has_any


def _get_satr_for_candle(session, candle, default_mask="sATR"):
    """
    Fetch sATR value for this candle.
    Tries exact mask first; falls back to icontains('satr').
    Returns float; raises if not found/parsable.
    """
    iv = (
        SessionStockDataIndicatorValue.objects
        .select_related("studyIndicator")
        .filter(sessionStockDataItem=candle, studyIndicator__mask=default_mask)
        .first()
    )
    if not iv:
        iv = (
            SessionStockDataIndicatorValue.objects
            .select_related("studyIndicator")
            .filter(sessionStockDataItem=candle, studyIndicator__mask__icontains="satr")
            .first()
        )
    if not iv:
        raise ValueError("sATR_not_found")

    raw = iv.value
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "value" in parsed and parsed["value"] is not None:
            return float(parsed["value"])
    except Exception:
        pass
    return float(raw)


def _price_decimals_from_tick(tick_size):
    if not tick_size:
        return 3
    return decimals_from_tick(Decimal(str(tick_size)))


def _q_price(p, tick_size):
    if not tick_size:
        return float(p)
    return float(round_to_tick(p, Decimal(str(tick_size))))


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
            return self.create_response(request, [], response_class=http.HttpResponse)

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

    def _parse_json(self, request):
        try:
            return json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(http.HttpResponseBadRequest("Invalid JSON"))

    def _resolve_market_meta(self, session, body):
        tick_size = body.get('tick_size') or body.get('tickSize') or getattr(session, 'tick_size', None)
        lot_step  = body.get('lot_step')  or body.get('lotStep')  or getattr(session, 'lot_step', None)
        tick_size = Decimal(str(tick_size)) if tick_size is not None else None
        lot_step  = Decimal(str(lot_step))  if lot_step  is not None else None
        price_decimals = decimals_from_tick(tick_size)
        symbol = getattr(session, 'ticker', None) or getattr(session.study, 'ticker', '')
        timeframe = getattr(session, 'timeFrame', None) or getattr(session.study, 'timeFrame', '')
        return symbol, timeframe, tick_size, lot_step, price_decimals

    def _get_indicator_value_for_candle(self, session, candle, mask='sATR14'):
        si = (
            StudyIndicator.objects
            .filter(study=session.study, mask=mask)
            .order_by('id')
            .first()
        )
        if not si:
            raise ImmediateHttpResponse(http.HttpResponseBadRequest(f"StudyIndicator with mask '{mask}' not found"))

        iv = (
            SessionStockDataIndicatorValue.objects
            .filter(sessionStockDataItem=candle, studyIndicator=si)
            .first()
        )
        if not iv:
            raise ImmediateHttpResponse(http.HttpResponseBadRequest(f"Indicator '{mask}' value not found for this candle"))

        try:
            obj = json.loads(iv.value)
            v = obj.get('value', None) if isinstance(obj, dict) else None
        except Exception:
            v = None
        if v is None:
            raise ImmediateHttpResponse(http.HttpResponseBadRequest(f"Indicator '{mask}' value is empty for this candle"))

        return Decimal(str(v))

    def _format_price(self, x: Decimal, decimals: int) -> float:
        q = Decimal('1').scaleb(-decimals) if decimals else None
        return float(x.quantize(q)) if q else float(x)

    def _generate_for_candle(self, session: TradingSession, candle: SessionStockData, body: dict):
        """
        Builds BUY/SELL × lp_offsets × stop_losses × take_profits combinations,
        quantizes prices, and (if not dry_run) persists SessionPotentialOrder.
        """
        # Use study-linked trading plan unless overridden in body
        plan_used, plan_source, _ = _resolve_trading_plan_for_session(session, body)

        symbol, timeframe, tick_size, lot_step, price_decimals = self._resolve_market_meta(session, body)
        dry_run = bool(body.get('dry_run', False))
        quantize = bool(body.get('quantize', True))

        close = Decimal(str(candle.close))
        satr = body.get('satr')
        satr = Decimal(str(satr)) if satr is not None else self._get_indicator_value_for_candle(session, candle, mask='sATR14')

        results = []
        with transaction.atomic():
            for direction in ('BUY', 'SELL'):
                for lpo in (Decimal(str(x)) for x in plan_used["lp_offsets"]):
                    entry = close - (lpo * satr) if direction == 'BUY' else close + (lpo * satr)

                    for slatr in (Decimal(str(x)) for x in plan_used["stop_losses"]):
                        stop = entry - (slatr * satr) if direction == 'BUY' else entry + (slatr * satr)

                        for tpmult in (Decimal(str(x)) for x in plan_used["take_profits"]):
                            take = entry + (slatr * satr) * tpmult if direction == 'BUY' else entry - (slatr * satr) * tpmult

                            if quantize and tick_size:
                                entry_q = Decimal(str(_q_price(entry, tick_size)))
                                stop_q  = Decimal(str(_q_price(stop,  tick_size)))
                                take_q  = Decimal(str(_q_price(take,  tick_size)))
                            else:
                                entry_q, stop_q, take_q = entry, stop, take

                            idem_src = f"{session.id}|{candle.id}|{direction}|{lpo}|{slatr}|{tpmult}|{tick_size}|{lot_step}|{quantize}|{close}|{satr}"
                            idem = hashlib.sha256(idem_src.encode("utf-8")).hexdigest()

                            spo_id = None
                            if not dry_run:
                                spo, _ = SessionPotentialOrder.objects.get_or_create(
                                    idempotency_hash=idem,
                                    defaults=dict(
                                        trading_session=session,
                                        sessionStockDataItem=candle,
                                        direction=direction,
                                        lpOffset=float(lpo),
                                        slATR=float(slatr),
                                        tp=int(tpmult),
                                        limitPrice=float(entry_q),
                                        stopPrice=float(stop_q),
                                        takeProfitPrice=float(take_q),
                                        decision='NONE',
                                        status='NEW',
                                        meta={
                                            "symbol": symbol,
                                            "timeframe": timeframe,
                                            "tick_size": float(tick_size) if tick_size else None,
                                            "lot_step": float(lot_step) if lot_step else None,
                                            "close_used": float(close),
                                            "satr_used": float(satr),
                                            "plan_source": plan_source,
                                            "plan_used": plan_used,
                                        }
                                    )
                                )
                                spo_id = spo.id

                            results.append({
                                "id": spo_id,
                                "session_id": session.id,
                                "candle_id": candle.id,
                                "direction": direction,
                                "lpOffset": float(lpo),
                                "slATR": float(slatr),
                                "tp": int(tpmult),
                                "limitPrice": self._format_price(entry_q, price_decimals),
                                "stopPrice":  self._format_price(stop_q,  price_decimals),
                                "takeProfitPrice": self._format_price(take_q, price_decimals),
                                "format": {"price_decimals": price_decimals},
                                "meta": {
                                    "close_used": self._format_price(close, price_decimals),
                                    "satr_used": float(satr),
                                    "tick_size": float(tick_size) if tick_size else None,
                                    "lot_step": float(lot_step) if lot_step else None,
                                    "plan_source": plan_source,
                                    "plan_used": plan_used,
                                    "idempotency_hash": idem,
                                },
                                "status": "NEW" if not dry_run else "DRY_RUN",
                            })

        results.sort(key=lambda r: (r["direction"], r["lpOffset"], r["slATR"], r["tp"]))
        return results

    # ----------------------------- Routes: generators -----------------------------

    def generate_potential_orders(self, request, **kwargs):
        """
        POST /api/v1/sessions/<pk>/candles/<candle_id>/potential-orders:generate
        Body may include overrides; otherwise uses study-linked trading plan.
        """
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        candle = get_object_or_404(SessionStockData, pk=kwargs['candle_id'], trading_session=session)

        body = self._parse_json(request)
        data = self._generate_for_candle(session, candle, body)

        self.log_throttled_access(request)
        return self.create_response(request, data)

    def candles_generate(self, request, **kwargs):
        """
        POST /api/v1/sessions/<pk>/candles:generate/?timestamp=<epoch_ms>
        Body:
          {
            "dry_run": true|false,
            // optional overrides; if omitted -> Study->TradingPlan JSON used
            "lp_offsets": [...],
            "stop_losses": [...],
            "take_profits": [...],
            // broker formatting
            "quantize": true|false,
            "tick_size": 0.001,
            "lot_step": 1
          }
        """
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])

        try:
            body = json.loads(request.body.decode('utf-8') or "{}")
        except Exception:
            body = {}

        ts = request.GET.get('timestamp') or body.get('timestamp')
        if ts is None:
            return self.create_response(request, {"error": "timestamp_required"}, http.HttpBadRequest)

        # Resolve candle
        candle = get_object_or_404(SessionStockData, trading_session=session, timestamp=ts)

        # Generate using study-linked plan (with optional overrides in body)
        data = self._generate_for_candle(session, candle, body)

        payload = {
            "results": data
        }
        self.log_throttled_access(request)
        return self.create_response(request, payload)
