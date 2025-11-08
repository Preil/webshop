import json

from django.urls import re_path
from django.db import IntegrityError
from tastypie.resources import ModelResource
from tastypie.authorization import Authorization
from tastypie.utils import trailing_slash
from api.authentication import CustomApiKeyAuthentication
from django.shortcuts import get_object_or_404
import logging
logger = logging.getLogger(__name__)
from tastypie.exceptions import ImmediateHttpResponse
from django.http import HttpResponseBadRequest
from decimal import Decimal
from django.utils import timezone

from shop.session_models import (
    SessionFill, TradingSession, SessionOrder,
    )


class SessionFillResource(ModelResource):
    class Meta:
        queryset = SessionFill.objects.all()
        resource_name = 'sessionFills'
        allowed_methods = ['get', 'post', 'delete']  # add 'put' if you plan to edit fills
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
        always_return_data = True
        filtering = {
            'session': ['exact'],
            'sessionOrder': ['exact'],
            'brokerTradeId': ['exact'],
            'ts': ['range', 'gte', 'lte'],
        }

    # --------- URLs ----------
    def prepend_urls(self):
        return [
            # POST /api/v1/sessionFills/save_bulk/
            re_path(
                r'^sessionFills/save_bulk%s$' % trailing_slash(),
                self.wrap_view('save_bulk'),
                name='api_session_fills_save_bulk',
            ),
        ]

    # --------- helpers ----------
    @staticmethod
    def D(v):
        return None if v in (None, '') else Decimal(str(v))

    def _recalc_order_rollups(self, so: SessionOrder):
        """Recalculate filledQty, avgFillPrice, fees, status, filledAt from all fills."""
        fills = list(so.fills.order_by('ts', 'id').values('qty', 'price', 'fee'))
        total_qty = sum((f['qty'] for f in fills), Decimal('0'))
        total_value = sum((Decimal(f['qty']) * Decimal(f['price']) for f in fills), Decimal('0'))
        total_fees = sum((f['fee'] for f in fills), Decimal('0'))

        so.filledQty = total_qty
        so.avgFillPrice = (total_value / total_qty) if total_qty > 0 else None
        so.fees = total_fees

        # Status transitions
        if so.quantity is not None and total_qty >= so.quantity:
            so.status = 'FILLED'
            if not so.filledAt:
                so.filledAt = timezone.now()
        elif total_qty > 0:
            so.status = 'PARTIAL'
        # else keep existing status (PLACED/CANCELLED/EXPIRED/REJECTED)

        so.save(update_fields=['filledQty', 'avgFillPrice', 'fees', 'status', 'filledAt', 'updatedAt'])

    # --------- bulk create ----------
    def save_bulk(self, request, *args, **kwargs):
        """
        Payload examples:

        1) Common session + order:
        {
          "sessionId": 42,
          "sessionOrderId": 101,
          "fills": [
            {"ts": "2025-10-31T08:12:00Z", "qty": 0.5, "price": 100.01, "fee": 0.0005, "liquidityType": "MAKER", "brokerTradeId": "T1"},
            {"ts": "2025-10-31T08:13:00Z", "qty": 0.5, "price": 99.99,  "fee": 0.0005, "liquidityType": "TAKER", "brokerTradeId": "T2"}
          ]
        }

        2) Mixed orders per fill:
        {
          "sessionId": 42,
          "fills": [
            {"sessionOrderId": 101, "ts": "...", "qty": 0.1, "price": 10.0},
            {"sessionOrderId": 102, "ts": "...", "qty": 0.2, "price": 10.1}
          ]
        }
        """
        logger.info("SessionFill.save_bulk called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        try:
            payload = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Invalid JSON"))

        session_id = payload.get('sessionId')
        if not session_id:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Missing 'sessionId'"))
        session = get_object_or_404(TradingSession, pk=session_id)

        default_order_id = payload.get('sessionOrderId')
        fills = payload.get('fills', [])
        if not isinstance(fills, list):
            raise ImmediateHttpResponse(HttpResponseBadRequest("'fills' must be a list"))

        created, duplicates = 0, 0
        touched_orders = set()

        for f in fills:
            order_id = f.get('sessionOrderId', default_order_id)
            if not order_id:
                raise ImmediateHttpResponse(HttpResponseBadRequest("Each fill must provide 'sessionOrderId' or top-level 'sessionOrderId'"))

            so = get_object_or_404(SessionOrder, pk=order_id, session=session)

            qty = self.D(f.get('qty'))
            price = self.D(f.get('price'))
            if qty in (None, '') or price in (None, ''):
                raise ImmediateHttpResponse(HttpResponseBadRequest("Each fill requires 'qty' and 'price'"))

            fee = self.D(f.get('fee')) or Decimal('0')
            ts = f.get('ts') or timezone.now()
            liquidity = f.get('liquidityType')  # optional: "MAKER"/"TAKER"/None
            broker_trade_id = f.get('brokerTradeId')

            try:
                sf = SessionFill.objects.create(
                    session=session,
                    sessionOrder=so,
                    ts=ts,
                    qty=qty,
                    price=price,
                    fee=fee,
                    liquidityType=liquidity,
                    brokerTradeId=broker_trade_id,
                    metadata=f.get('metadata'),
                )
                created += 1
                touched_orders.add(so.id)
            except IntegrityError:
                # Likely the unique constraint (sessionOrder, brokerTradeId) hit for duplicates
                duplicates += 1
                continue

        # Recalculate roll-ups once per touched order
        for oid in touched_orders:
            so = SessionOrder.objects.get(pk=oid)
            self._recalc_order_rollups(so)

        self.log_throttled_access(request)
        return self.create_response(
            request,
            {'success': True, 'created': created, 'duplicates': duplicates, 'touchedOrders': list(touched_orders)}
        )