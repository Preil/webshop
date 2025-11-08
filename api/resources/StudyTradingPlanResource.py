from tastypie.resources import ModelResource
from shop.models import StudyTradingPlan
from tastypie.authorization import Authorization
from api.authentication import CustomApiKeyAuthentication

class StudyTradingPlanResource(ModelResource):
    class Meta:
        queryset = StudyTradingPlan.objects.all()
        resource_name = 'studyTradingPlans'
        allowed_methods = ['get']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()