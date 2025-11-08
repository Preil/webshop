from tastypie.resources import ModelResource
from shop.models import TradingPlan
from tastypie.authorization import Authorization
from api.authentication import CustomApiKeyAuthentication

class TradingPlanResource(ModelResource):
    class Meta:
        queryset = TradingPlan.objects.all()
        resource_name = 'tradingPlans'
        allowed_methods = ['get']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()