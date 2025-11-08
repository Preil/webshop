import json

from django.urls import re_path

from tastypie.resources import ModelResource
from tastypie.authorization import Authorization
from django import http
from api.authentication import CustomApiKeyAuthentication
from django.shortcuts import get_object_or_404
import logging
logger = logging.getLogger(__name__)


from shop.session_models import (
    SessionStockDataIndicatorValue, SessionStockData
    )
from shop.models import StudyIndicator


class SessionStockDataIndicatorValueResource(ModelResource):
    class Meta:
        queryset = SessionStockDataIndicatorValue.objects.all()
        resource_name = 'SessionStockDataIndicatorValue'
        allowed_methods = ['get', 'post', 'delete']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()

    def prepend_urls(self):
        return [
            re_path(
                r'^SessionStockDataIndicatorValue/save_bulk/$',
                self.wrap_view('save_bulk'),
                name='api_session_stock_data_indicator_value_save_bulk',
            ),
        ]

    def save_bulk(self, request, *args, **kwargs):
        """
        Bulk save endpoint for indicator values during trading session.
        Expected JSON payload:
        {
            "results": [
                {"sessionStockDataItemId": 101, "studyIndicatorId": 5, "value": "1.234"},
                ...
            ]
        }
        """
        logger.info("SessionStockDataIndicatorValueResource.save_bulk called")
        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        try:
            data = json.loads(request.body or "{}")
        except json.JSONDecodeError:
            return self.error_response(
                request,
                {"success": False, "message": "Invalid JSON format"},
                response_class=http.HttpBadRequest,
            )

        results = data.get('results', [])
        if not isinstance(results, list):
            return self.error_response(
                request,
                {"success": False, "message": "'results' must be a list"},
                response_class=http.HttpBadRequest,
            )

        created, updated = 0, 0

        for item in results:
            try:
                session_stock_data_item = get_object_or_404(SessionStockData, id=item.get('sessionStockDataItemId'))
                study_indicator = get_object_or_404(StudyIndicator, id=item.get('studyIndicatorId'))
                value = str(item.get('value', ''))

                obj, was_created = SessionStockDataIndicatorValue.objects.update_or_create(
                    sessionStockDataItem=session_stock_data_item,
                    studyIndicator=study_indicator,
                    defaults={'value': value},
                )
                created += int(was_created)
                updated += int(not was_created)
            except Exception as e:
                logger.error(f"Failed to process indicator value: {e}")

        self.log_throttled_access(request)
        return self.create_response(request, {
            "success": True,
            "message": "Session indicator values saved successfully",
            "created": created,
            "updated": updated,
        })