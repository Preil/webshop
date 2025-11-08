
from tastypie.resources import ModelResource
from tastypie.authorization import Authorization
from api.authentication import CustomApiKeyAuthentication
from shop.models import (
    TrainedNnModel,
)
class TrainedNnModelResource(ModelResource):
    class Meta:
        queryset = TrainedNnModel.objects.all()
        resource_name = 'trainedNnModels'
        allowed_methods = ['get', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()