from tastypie.resources import ModelResource
from tastypie.authorization import Authorization
from tastypie import fields


from shop.session_models import (
    SessionSettings
    )


class SessionSettingsResource(ModelResource):
    session_id = fields.IntegerField(attribute="session_id", readonly=True)

    class Meta:
        queryset = SessionSettings.objects.all().select_related("session")
        resource_name = "sessionSettings"
        allowed_methods = ["get", "post", "put", "delete"]
        authorization = Authorization()
        filtering = {
            "id": ["exact"],
            "session_id": ["exact"],
            "name": ["exact", "icontains"],
            "created_at": ["range", "gte", "lte"],
            "updated_at": ["range", "gte", "lte"],
        }
        always_return_data = True