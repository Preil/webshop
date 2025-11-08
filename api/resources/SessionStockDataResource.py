
from django.urls import re_path
from tastypie.resources import ModelResource
from tastypie.authorization import Authorization
from api.authentication import CustomApiKeyAuthentication
import logging
logger = logging.getLogger(__name__)


from shop.session_models import (
    SessionStockData,
    )


class SessionStockDataResource(ModelResource):
    class Meta:
        queryset = SessionStockData.objects.all()
        resource_name = 'SessionStockData'
        allowed_methods = ['get', 'delete', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()

    def prepend_urls(self):
        return [
            re_path(r'^SessionStockData/save_bulk/$', self.wrap_view('save_bulk'), name='api_session_save_bulk'),
        ]
