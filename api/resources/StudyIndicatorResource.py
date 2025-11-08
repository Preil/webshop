from tastypie.resources import ModelResource
from shop.models import StudyIndicator
from tastypie.authorization import Authorization
from api.authentication import CustomApiKeyAuthentication

class StudyIndicatorResource(ModelResource):
    class Meta:
        queryset = StudyIndicator.objects.all()
        resource_name = 'StudyIndicators'
        allowed_methods = ['get', 'delete', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()