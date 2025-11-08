import json
from django.http import JsonResponse, Http404
from django.urls import re_path
from django.core import serializers
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
    SessionPotentialOrder, TradingSession, SessionOrder,
    )

class SessionOrderResource(ModelResource):
    class Meta:
        queryset = SessionOrder.objects.all()
        resource_name = 'sessionOrders'
        allowed_methods = ['get', 'post', 'put', 'delete']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
        always_return_data = True
        filtering = {
            'session': ['exact'],
            'sessionPotentialOrder': ['exact'],
            'ticker': ['exact', 'icontains'],
            'status': ['exact'],
            'clientOrderId': ['exact'],
            'brokerOrderId': ['exact'],
            'createdAt': ['range', 'gte', 'lte'],
        }

    # ---------- URL extras ----------
    def prepend_urls(self):
        return [
            # POST /api/v1/sessionOrders/save_bulk/
            re_path(
                r'^sessionOrders/save_bulk%s$' % trailing_slash(),
                self.wrap_view('save_bulk'),
                name='api_session_orders_save_bulk',
            ),
            # POST /api/v1/sessionOrders/<pk>/cancel/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/cancel%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('cancel'),
                name='api_session_order_cancel',
            ),
            # POST /api/v1/sessionOrders/<pk>/ack/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/ack%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('acknowledge'),
                name='api_session_order_ack',
            ),
            # POST /api/v1/sessionOrders/<pk>/fill/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/fill%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('apply_fill'),
                name='api_session_order_fill',
            ),
            # POST /api/v1/sessionOrders/<pk>/expire/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/expire%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('expire'),
                name='api_session_order_expire',
            ),
        ]

    # ---------- Helpers ----------
    @staticmethod
    def D(v):
        return None if v in (None, '') else Decimal(str(v))

    def _validate_payload(self, od):
        # Basic enum checks (keep in sync with model)
        if od.get('side') not in ('BUY', 'SELL'):
            raise ImmediateHttpResponse(HttpResponseBadRequest("side must be BUY or SELL"))
        if od.get('orderType') not in ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'):
            raise ImmediateHttpResponse(HttpResponseBadRequest("orderType invalid"))
        if od.get('timeInForce') and od['timeInForce'] not in ('GTC', 'IOC', 'FOK', 'DAY'):
            raise ImmediateHttpResponse(HttpResponseBadRequest("timeInForce invalid"))
        # Price requirements
        ot = od.get('orderType')
        if ot in ('LIMIT', 'STOP_LIMIT') and od.get('limitPrice') is None:
            raise ImmediateHttpResponse(HttpResponseBadRequest("limitPrice required for LIMIT/STOP_LIMIT"))
        if ot in ('STOP', 'STOP_LIMIT') and od.get('stopPrice') is None:
            raise ImmediateHttpResponse(HttpResponseBadRequest("stopPrice required for STOP/STOP_LIMIT"))
        # Quantity
        if od.get('quantity') in (None, ''):
            raise ImmediateHttpResponse(HttpResponseBadRequest("quantity required"))

    # ---------- Bulk create ----------
    def save_bulk(self, request, *args, **kwargs):
        logger.info("SessionOrder.save_bulk called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        try:
            payload = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Invalid JSON"))

        session_id = payload.get('sessionId')
        orders = payload.get('orders', [])
        if not session_id:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Missing 'sessionId'"))
        if not isinstance(orders, list):
            raise ImmediateHttpResponse(HttpResponseBadRequest("'orders' must be a list"))

        session = get_object_or_404(TradingSession, pk=session_id)

        created, skipped, ids = 0, 0, []
        for od in orders:
            # Validate fields
            self._validate_payload(od)

            # Idempotency within session
            coid = od.get('clientOrderId')
            if not coid:
                raise ImmediateHttpResponse(HttpResponseBadRequest("clientOrderId is required for idempotency"))
            if SessionOrder.objects.filter(session=session, clientOrderId=coid).exists():
                skipped += 1
                continue

            spo = None
            if od.get('sessionPotentialOrderId') is not None:
                spo = get_object_or_404(SessionPotentialOrder, pk=od['sessionPotentialOrderId'])

            so = SessionOrder(
                session=session,
                sessionPotentialOrder=spo,
                parentOrder_id=od.get('parentOrderId'),
                ticker=od.get('ticker', ''),
                timeframe=od.get('timeframe'),
                venue=od.get('venue'),
                clientOrderId=coid,
                brokerOrderId=od.get('brokerOrderId'),
                ocoGroupId=od.get('ocoGroupId'),
                exitRole=od.get('exitRole', 'NONE'),

                side=od.get('side'),
                orderType=od.get('orderType'),
                timeInForce=od.get('timeInForce', 'GTC'),

                quantity=self.D(od.get('quantity')),
                limitPrice=self.D(od.get('limitPrice')),
                stopPrice=self.D(od.get('stopPrice')),
                reduceOnly=bool(od.get('reduceOnly', False)),
                postOnly=bool(od.get('postOnly', False)),
                expireAt=od.get('expireAt'),

                status='PLACED',
                createdAt=timezone.now(),  # keep created time here; placedAt set on ack
                rejectReason=None,

                filledQty=Decimal('0'),
                avgFillPrice=None,
                fees=Decimal('0'),
                slippage=self.D(od.get('slippage')),
                metadata=od.get('metadata'),
            )
            so.save()
            created += 1
            ids.append(so.id)

        self.log_throttled_access(request)
        return self.create_response(
            request,
            {'success': True, 'created': created, 'skipped': skipped, 'ids': ids, 'sessionId': session.id},
        )

    # ---------- Status transitions ----------
    def acknowledge(self, request, **kwargs):
        """Broker ack: set brokerOrderId and placedAt."""
        logger.info("SessionOrder.acknowledge called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        so = get_object_or_404(SessionOrder, pk=kwargs['pk'])
        try:
            payload = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Invalid JSON"))

        if payload.get('brokerOrderId'):
            so.brokerOrderId = payload['brokerOrderId']
        if not so.placedAt:
            so.placedAt = timezone.now()
        if so.status == 'PLACED':
            so.status = 'PLACED'  # remains placed; keeping for clarity

        so.save(update_fields=['brokerOrderId', 'placedAt', 'status', 'updatedAt'])
        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'id': so.id, 'brokerOrderId': so.brokerOrderId})

    def apply_fill(self, request, **kwargs):
        """Apply a (partial) fill; updates filledQty, avgFillPrice, fees, status."""
        logger.info("SessionOrder.apply_fill called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        so = get_object_or_404(SessionOrder, pk=kwargs['pk'])
        try:
            payload = json.loads(request.body or "{}")
        except ValueError:
            raise ImmediateHttpResponse(HttpResponseBadRequest("Invalid JSON"))

        qty = payload.get('qty')
        price = payload.get('price')
        fee = payload.get('fee', 0)

        if qty in (None, '') or price in (None, ''):
            raise ImmediateHttpResponse(HttpResponseBadRequest("qty and price are required"))

        qty = Decimal(str(qty))
        price = Decimal(str(price))
        fee = Decimal(str(fee))

        # VWAP update
        prev_qty = so.filledQty or Decimal('0')
        prev_value = (so.avgFillPrice or Decimal('0')) * prev_qty
        new_qty = prev_qty + qty
        new_value = prev_value + (price * qty)

        so.filledQty = new_qty
        so.avgFillPrice = (new_value / new_qty) if new_qty > 0 else None
        so.fees = (so.fees or Decimal('0')) + fee

        # Status transitions
        if so.quantity and new_qty >= so.quantity:
            so.status = 'FILLED'
            if not so.filledAt:
                so.filledAt = timezone.now()
        else:
            so.status = 'PARTIAL'

        so.save(update_fields=['filledQty', 'avgFillPrice', 'fees', 'status', 'filledAt', 'updatedAt'])
        self.log_throttled_access(request)
        return self.create_response(request, {
            'success': True,
            'id': so.id,
            'filledQty': str(so.filledQty),
            'avgFillPrice': str(so.avgFillPrice) if so.avgFillPrice is not None else None,
            'fees': str(so.fees),
            'status': so.status,
        })

    def cancel(self, request, **kwargs):
        logger.info("SessionOrder.cancel called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        so = get_object_or_404(SessionOrder, pk=kwargs['pk'])
        so.status = 'CANCELLED'
        so.cancelledAt = timezone.now()
        so.save(update_fields=['status', 'cancelledAt', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'id': so.id, 'status': so.status})

    def expire(self, request, **kwargs):
        logger.info("SessionOrder.expire called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        so = get_object_or_404(SessionOrder, pk=kwargs['pk'])
        so.status = 'EXPIRED'
        so.expiredAt = timezone.now()
        so.save(update_fields=['status', 'expiredAt', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'id': so.id, 'status': so.status})