from tastypie.resources import ModelResource
from shop.models import Indicator

class IndicatorResource(ModelResource):
    class Meta:
        queryset = Indicator.objects.all()
        resource_name = 'indicators'
        allowed_methods = ['get']