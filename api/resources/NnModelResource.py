import json
from django.http import JsonResponse, Http404
from django.urls import re_path
from django.core import serializers
from tastypie.resources import ModelResource
from tastypie.authorization import Authorization
from tastypie.utils import trailing_slash
from api.authentication import CustomApiKeyAuthentication
from django.shortcuts import get_object_or_404
from shop.bakery import train_nn_model, check_training_status

# Models used by the exact methods below
from shop.models import (
    NnModel,
)

class NnModelResource(ModelResource):
    class Meta:
        queryset = NnModel.objects.all()
        resource_name = 'nnModels'
        allowed_methods = ['get', 'delete', 'post']
        authentication = CustomApiKeyAuthentication()
        authorization = Authorization()

    def prepend_urls(self):
        return [
            # /api/nnModels/1/train/ - to train model with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/train%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('train_model'), name="api_train_model"),
            # /api/nnModels/1/status/ - to check training status of model with id 1
            re_path(r'^(?P<resource_name>%s)/(?P<pk>\w[\w/-]*)/status%s$' % (self._meta.resource_name, trailing_slash()), self.wrap_view('check_status'), name="api_check_status"),
        ]

    def train_model(self, request, **kwargs):
        model = get_object_or_404(NnModel, pk=kwargs['pk'])
        train_nn_model(model)
        return self.create_response(request, {'status': 'training started'})

    def check_status(self, request, **kwargs):
        model = get_object_or_404(NnModel, pk=kwargs['pk'])
        status = check_training_status(model)
        return self.create_response(request, {'status': status})