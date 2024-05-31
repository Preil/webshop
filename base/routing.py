from django.urls import re_path
from api import consumers

websocket_urlpatterns = [
    re_path(r'ws/train/(?P<model_id>\d+)/$', consumers.TrainModelConsumer.as_asgi()),
]
