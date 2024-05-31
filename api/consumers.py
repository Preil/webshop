import json
from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync

class TrainModelConsumer(WebsocketConsumer):
    def connect(self):
        self.model_id = self.scope['url_route']['kwargs']['model_id']
        self.room_group_name = f'train_model_{self.model_id}'

        # Join room group
        async_to_sync(self.channel_layer.group_add)(
            self.room_group_name,
            self.channel_name
        )

        self.accept()

    def disconnect(self, close_code):
        # Leave room group
        async_to_sync(self.channel_layer.group_discard)(
            self.room_group_name,
            self.channel_name
        )

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        # Send message to room group
        async_to_sync(self.channel_layer.group_send)(
            self.room_group_name,
            {
                'type': 'training_status',
                'message': message
            }
        )

    def training_status(self, event):
        message = event['message']

        # Send message to WebSocket
        self.send(text_data=json.dumps({
            'message': message
        }))
