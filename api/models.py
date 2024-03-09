from django.shortcuts import get_object_or_404
from django.core import serializers
from tastypie.resources import ModelResource
from tastypie.utils import trailing_slash
from shop.models import Category, Course, StockData, Study
from api.authentication import CustomApiKeyAuthentication
from tastypie.authorization import Authorization
from django.http import HttpResponse
import json
from django.urls import re_path
import logging
logger = logging.getLogger(__name__)

# endpoints examples

# /api/categories/
# /api/courses/

# /api/categories/2/
# /api/courses/3/

class StockDataResource(ModelResource):
    class Meta:
        queryset = StockData.objects.all()
        resource_name = 'StockData'
        allowed_methods = ['get', 'delete', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()

    def prepend_urls(self):
        return [
            re_path(r'^StockData/save_bulk/$', self.wrap_view('save_bulk'), name='api_save_bulk'),
        ]

    def save_bulk(self, request, *args, **kwargs):
        logger.info("save_bulk method is called")

        self.method_check(request, allowed=['post'])
        self.is_authenticated(request)
        self.throttle_check(request)

        # Assuming POST request with JSON body
        data = json.loads(request.body)
        results = data.get('results', [])
        study_id = data.get('studyId')

        # Get the study object
        study = get_object_or_404(Study, id=study_id)

        for stock_data in results:
            StockData.objects.create(
                ticker=data.get('ticker'),
                volume=stock_data.get('v'),
                vw=stock_data.get('vw'),
                open=stock_data.get('o'),
                close=stock_data.get('c'),
                high=stock_data.get('h'),
                low=stock_data.get('l'),
                timestamp=stock_data.get('t'),
                transactions=stock_data.get('n'),
                timeframe=data.get('timeFrame'),
                study=study
            )
        self.log_throttled_access(request)
        return self.create_response(request, {'success': True, 'message': 'Stock data saved successfully'})


class CategoryResource(ModelResource):
    class Meta:
        queryset = Category.objects.all()
        resource_name = 'categories'
        allowed_methods = ['get']


class StudyResource(ModelResource):
    class Meta:
        queryset = Study.objects.all()
        resource_name = 'studies'
        allowed_methods = ['get', 'delete', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()
    def prepend_urls(self):
        return [
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/stockdata%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('get_stockdata'), name="api_get_stockdata"),
        ]

    def get_stockdata(self, request, **kwargs):
        try:
            study = Study.objects.get(pk=kwargs['pk'])
        except Study.DoesNotExist:
            return self.create_response(request, {'error': 'not found'}, Http404)

        stockdata = StockData.objects.filter(study=study)
        stockdata_json = serializers.serialize('json', stockdata)

        return self.create_response(request, stockdata_json)



class CourseResource(ModelResource):
    class Meta:
        queryset = Course.objects.all()
        resource_name = 'courses'
        allowed_methods = ['get', 'delete', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()

    def hydrate(self, bundle):
        bundle.obj.category_id = bundle.data['category_id']
        return bundle
    
    def dehydrate(self, bundle):
        bundle.data['category_id'] = bundle.obj.category_id
        bundle.data['category'] = bundle.obj.category
        return bundle
    

    
