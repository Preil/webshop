from tastypie.resources import ModelResource
from shop.models import StudyOrder
from tastypie.authorization import Authorization
from api.authentication import CustomApiKeyAuthentication

class StudyOrderResource(ModelResource):
    class Meta:
        queryset = StudyOrder.objects.all()
        resource_name = 'studyOrders'
        allowed_methods = ['get']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()