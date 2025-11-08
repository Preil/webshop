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


def _get_satr_for_candle(session, candle):
    """
    Fetch sATR value for this candle using the Study's mainSatrIndicator.
    Returns float; raises ValueError if not found/parsable.
    """
    from shop.session_models import SessionStockDataIndicatorValue
    from shop.models import StudyIndicator

    study = getattr(session, "study", None)
    if not study or not getattr(study, "mainSatrIndicator_id", None):
        raise ValueError("mainSatrIndicator_not_set")

    iv = (
        SessionStockDataIndicatorValue.objects
        .filter(sessionStockDataItem=candle, studyIndicator_id=study.mainSatrIndicator_id)
        .only("value")
        .first()
    )
    if not iv:
        raise ValueError("sATR_not_found")

    raw = iv.value
    # Stored as string; allow plain number or {"value": number}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "value" in parsed:
            return float(parsed["value"])
    except Exception:
        pass
    return float(raw)


def _price_decimals_from_tick(tick_size):
    if not tick_size:
        return 3
    return decimals_from_tick(Decimal(str(tick_size)))


def _q_price(p: Decimal, tick_size: Decimal) -> Decimal:
    """Round to the nearest valid tick."""
    if not tick_size:
        return p
    return round_to_tick(p, tick_size)  # keep it a Decimal



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
    def _resolve_trading_plan_for_session(self, session):
        """
        Return normalized trading plan dict:
        {
            "lp_offsets":   [ ... ],
            "stop_losses":  [ ... ],
            "take_profits": [ ... ],
        }
        Source: Session → Study → StudyTradingPlan → TradingPlan → TradingPlanParams (stringified JSON).
        """
        study = getattr(session, "study", None)
        if not study:
            raise ImmediateHttpResponse(HttpResponse("Session has no linked Study", status=422))

        # StudyTradingPlan.TradingPlan.TradingPlanParams is a string like:
        # {"LPoffset":[0.1,0.2], "stopLoss":[0.3,0.5], "takeProfit":[3,4]}
        stp = getattr(study, "studytradingplan", None) or getattr(study, "StudyTradingPlan", None)
        if not stp or not getattr(stp, "tradingPlan", None):
            raise ImmediateHttpResponse(HttpResponse("Study has no TradingPlan attached", status=422))
        params_str = getattr(stp.tradingPlan, "tradingPlanParams", None) or getattr(stp.tradingPlan, "TradingPlanParams", None)
        if not params_str:
            raise ImmediateHttpResponse(HttpResponse("TradingPlanParams not set", status=422))
        try:
            raw = json.loads(params_str)
        except Exception:
            raise ImmediateHttpResponse(HttpResponse("TradingPlanParams is not valid JSON", status=422))
            
        # Normalize keys
        lp  = raw.get("LPoffset", []) or raw.get("lp_offsets", [])
        sl  = raw.get("stopLoss", []) or raw.get("stop_losses", [])
        tp  = raw.get("takeProfit", []) or raw.get("take_profits", [])

        if not lp or not sl or not tp:
            raise ImmediateHttpResponse(HttpResponse("TradingPlanParams must include non-empty LPoffset, stopLoss, takeProfit", status=422))

        return {
            "lp_offsets":   lp,
            "stop_losses":  sl,
            "take_profits": tp,
        }

    def _price_decimals_from_tick(self, tick_size):
        """e.g., 0.001 -> 3; None -> None"""
        if not tick_size:
            return None
        s = str(tick_size).rstrip("0")
        if "." in s:
            return len(s.split(".")[1])
        return 0
    
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

    def _generate_for_candle(self, session, candle, dry_run: bool = True, quantize: bool = True):
        """
        Build BUY/SELL × lp_offsets × stop_losses × take_profits for a single candle.
        Persist when dry_run=False. Return list of rows (with id when saved).
        """
        plan = self._resolve_trading_plan_for_session(session)
        tick_size, price_decimals = self._resolve_market_meta(session)

        close = Decimal(str(candle.close))
        satr  = Decimal(str(self._get_satr_for_candle(session, candle)))

        results = []
        ctx = transaction.atomic() if not dry_run else contextlib.nullcontext()
        with ctx:
            for direction in ("BUY", "SELL"):
                for lpo in (Decimal(str(x)) for x in plan["lp_offsets"]):
                    entry = close - (lpo * satr) if direction == "BUY" else close + (lpo * satr)

                    for slatr in (Decimal(str(x)) for x in plan["stop_losses"]):
                        stop = entry - (slatr * satr) if direction == "BUY" else entry + (slatr * satr)

                        for tpmult in (Decimal(str(x)) for x in plan["take_profits"]):
                            take = entry + (slatr * satr) * tpmult if direction == "BUY" else entry - (slatr * satr) * tpmult

                            entry_q = _q_price(entry, tick_size) if (quantize and tick_size) else entry
                            stop_q  = _q_price(stop,  tick_size) if (quantize and tick_size) else stop
                            take_q  = _q_price(take,  tick_size) if (quantize and tick_size) else take


                            spo_id = None
                            if not dry_run:
                                spo = SessionPotentialOrder.objects.create(
                                    session=session,
                                    sessionStockDataItem=candle,   # ✅ now points directly to SessionStockData
                                    direction=direction,
                                    limitPrice=entry_q,
                                    stopPrice=stop_q,
                                    takeProfitPrice=take_q,
                                    lpOffset=lpo,
                                    slATR=slatr,
                                    tp=int(tpmult),
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
                                "limitPrice": float(entry_q) if price_decimals is None else float(entry_q.quantize(Decimal('1').scaleb(-price_decimals))),
                                "stopPrice":  float(stop_q)  if price_decimals is None else float(stop_q.quantize(Decimal('1').scaleb(-price_decimals))),
                                "takeProfitPrice": float(take_q) if price_decimals is None else float(take_q.quantize(Decimal('1').scaleb(-price_decimals))),
                            })

        results.sort(key=lambda r: (r["direction"], r["lpOffset"], r["slATR"], r["tp"]))
        return results


    # ----------------------------- Routes: generators -----------------------------

    def generate_potential_orders(self, request, **kwargs):
        """
        POST /api/v1/sessions/<pk>/candles/<candle_id>/potential-orders:generate
        Uses study-linked trading plan. Optional query: ?dry_run=1&quantize=1
        """
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        session = get_object_or_404(TradingSession, pk=kwargs['pk'])
        candle = get_object_or_404(SessionStockData, pk=kwargs['candle_id'], trading_session=session)

        dry_run  = request.GET.get("dry_run", "1") != "0"
        quantize = request.GET.get("quantize", "1") != "0"

        try:
            data = self._generate_for_candle(session, candle, dry_run=dry_run, quantize=quantize)
            return self.create_response(request, data)
        except ImmediateHttpResponse:
            raise
        except Exception as e:
            from django.http import HttpResponse
            return self.create_response(request, {"error_message": str(e)}, HttpResponse(status=500))


    def candles_generate(self, request, **kwargs):
        """
        POST /api/v1/sessions/{id}/candles:generate/?timestamp=<epoch_ms>[&dry_run=1][&quantize=1]
        """
        try:
            session = TradingSession.objects.get(pk=kwargs["pk"])
        except TradingSession.DoesNotExist:
            return self.create_response(request, {"error": "Session not found"}, http.HttpNotFound)

        ts_ms = request.GET.get("timestamp")
        if not ts_ms:
            return self.create_response(request, {"error": "timestamp is required (epoch ms)"}, http.HttpBadRequest)

        try:
            ts_ms = int(ts_ms)
        except ValueError:
            return self.create_response(request, {"error": "timestamp must be integer epoch ms"}, http.HttpBadRequest)

        # Find exact candle by timestamp (adjust if you need nearest <= ts)
        try:
            candle = SessionStockData.objects.get(trading_session=session, timestamp=ts_ms)
        except SessionStockData.DoesNotExist:
            return self.create_response(request, {"error": "Candle not found for timestamp"}, http.HttpNotFound)

        dry_run  = request.GET.get("dry_run", "1") != "0"
        quantize = request.GET.get("quantize", "1") != "0"

        try:
            results = self._generate_for_candle(session, candle, dry_run=dry_run, quantize=quantize)
            return self.create_response(request, results, http.HttpAccepted if dry_run else http.HttpCreated)
        except ImmediateHttpResponse as e:
            # re-raise ImmediateHttpResponse to Tastypie
            raise
        except Exception as e:
            return self.create_response(request, {"error_message": str(e)}, HttpResponse(status=500))

