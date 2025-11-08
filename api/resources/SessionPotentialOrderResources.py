
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

from shop.session_models import (
    SessionPotentialOrder, TradingSession, SessionStockData,
    )
from shop.models import TrainedNnModel

class SessionPotentialOrderResource(ModelResource):
    class Meta:
        queryset = SessionPotentialOrder.objects.all()
        resource_name = 'sessionPotentialOrders'
        allowed_methods = ['get', 'post', 'delete']  # add 'put' if you want updates
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
        always_return_data = True
        filtering = {
            'session': ['exact'],
            'sessionStockDataItem': ['exact'],
            'trainedModel': ['exact'],
            'direction': ['exact'],
            'decision': ['exact'],
            'createdAt': ['range', 'gte', 'lte'],
        }

    # ---------- Custom endpoints ----------
    def prepend_urls(self):
        return [
            # POST /api/v1/sessionPotentialOrders/save_bulk/
            re_path(
                r'^sessionPotentialOrders/save_bulk%s$' % trailing_slash(),
                self.wrap_view('save_bulk'),
                name='api_session_potential_orders_save_bulk',
            ),
            # POST /api/v1/sessionPotentialOrders/<pk>/approve/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/approve%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('approve'),
                name='api_session_potential_order_approve',
            ),
            # POST /api/v1/sessionPotentialOrders/<pk>/reject/
            re_path(
                r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/reject%s$'
                % (self._meta.resource_name, trailing_slash()),
                self.wrap_view('reject'),
                name='api_session_potential_order_reject',
            ),
        ]

    # ---------- Bulk create ----------
    def save_bulk(self, request, *args, **kwargs):
        logger.info("SessionPotentialOrder.save_bulk called")

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

        created = 0
        items = []
        for od in orders:
            # Expected keys in each order dict:
            # sessionStockDataItemId, trainedModelId, direction, limitPrice,
            # takeProfitPrice, stopPrice, lpOffset, slATR, tp, prediction, decision
            try:
                direction = od.get('direction')
                if direction not in ('BUY', 'SELL'):
                    raise ValueError("direction must be BUY or SELL")

                spdi = None
                if od.get('sessionStockDataItemId') is not None:
                    spdi = get_object_or_404(SessionStockData, pk=od['sessionStockDataItemId'])

                tmodel = None
                if od.get('trainedModelId') is not None:
                    tmodel = get_object_or_404(TrainedNnModel, pk=od['trainedModelId'])

                def D(v):
                    return None if v in (None, '') else Decimal(str(v))

                spo = SessionPotentialOrder(
                    session=session,
                    sessionStockDataItem=spdi,
                    trainedModel=tmodel,
                    direction=direction,
                    limitPrice=D(od.get('limitPrice')),
                    takeProfitPrice=D(od.get('takeProfitPrice')),
                    stopPrice=D(od.get('stopPrice')),
                    lpOffset=D(od.get('lpOffset')),
                    slATR=D(od.get('slATR')),
                    tp=od.get('tp'),
                    prediction=od.get('prediction'),
                    decision=od.get('decision', 'NONE'),
                )
                # minimal validation mirrors model intent
                if spo.tp is not None and int(spo.tp) <= 0:
                    raise ValueError("tp must be a positive integer")
                spo.save()
                created += 1
                items.append(spo.id)
            except Exception as e:
                logger.exception("Failed to create SessionPotentialOrder: %s", e)
                # continue creating others; you can change to fail-fast if you prefer

        self.log_throttled_access(request)
        return self.create_response(
            request,
            {'success': True, 'created': created, 'ids': items, 'sessionId': session.id},
        )

    # ---------- Simple state helpers ----------
    def approve(self, request, **kwargs):
        logger.info("SessionPotentialOrder.approve called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        spo = get_object_or_404(SessionPotentialOrder, pk=kwargs['pk'])
        spo.decision = 'APPROVED'
        spo.save(update_fields=['decision', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'id': spo.id, 'decision': spo.decision})

    def reject(self, request, **kwargs):
        logger.info("SessionPotentialOrder.reject called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        spo = get_object_or_404(SessionPotentialOrder, pk=kwargs['pk'])
        spo.decision = 'REJECTED'
        spo.save(update_fields=['decision', 'updatedAt'])

        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'id': spo.id, 'decision': spo.decision})