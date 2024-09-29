# from qdrant_functions import *
from models import models

from fastapi import FastAPI
import pika
import json
import uuid

app = FastAPI()

RABBITMQ_HOST = 'localhost'
RABBITMQ_USER = 'user'
RABBITMQ_PASS = 'password'


class RpcClient:
    def __init__(self):
        # Подключаемся к RabbitMQ
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials)
        )
        self.channel = self.connection.channel()

        # Объявляем очередь для получения ответов
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        # Устанавливаем функцию обратного вызова для получения ответов
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )
        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, uuid_video):
        self.response = None
        self.corr_id = str(uuid.uuid4())

        # Отправляем запрос с указанием очереди для ответа (reply_to) и уникального ID (correlation_id)
        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id
            ),
            body=json.dumps({"uuid_video": uuid_video})
        )

        # Ожидаем ответа
        while self.response is None:
            self.connection.process_data_events()
        return json.loads(self.response)


# https://s3.ritm.media/yappy-db-duplicates/2408bc4f-9411-470e-b7c3-8c19b964f7be.mp4
@app.post("/check-video-duplicate")
async def check_video(video_link: models.VideoLink) -> models.Response:
    uuid_video = video_link.link.split('/')[-1][:-4]
    rabbitmq_client = RpcClient()
    res = rabbitmq_client.call(uuid_video)
    is_duplicate, duplicate_for = res["is_duplicate"], res["duplicate_for"]
    return models.Response(is_duplicate=is_duplicate, duplicate_for=duplicate_for)
