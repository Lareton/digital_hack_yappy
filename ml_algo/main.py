from algo_utils import download_video
from ml_algo import get_res_by_uuid
from check_duplicate import model, df, vector_db
import pika
import json

RABBITMQ_HOST = 'rabbitmq'
RABBITMQ_USER = 'user'
RABBITMQ_PASS = 'password'


def check_video_is_duplicate_by_uuid(uuid_video: str) -> tuple:
    """
    Проверяет, является ли видео дубликатом на основе его UUID.

    Параметры:
    ----------
    uuid_video : str
        Уникальный идентификатор видео.

    Возвращает:
    ----------
    tuple : (bool, str)
        - pred_is_dup (bool): True, если видео является дубликатом, иначе False.
        - pred_uuid (str): UUID видео, которое было определено как оригинал.

    Описание работы:
    ---------------
    1. По переданному UUID формируется ссылка на видео в облачном хранилище.
    2. Видео скачивается с помощью функции `download_video` модуля utils.
    3. Проводится поиск дубликатов с помощью модели `ViSiL`, предварительно загруженных данных и векторного индекса.
    4. Возвращается результат в виде кортежа: является ли видео дубликатом и UUID оригинала, если найдено совпадение.
    """
    # Формируем ссылку на видео по переданному UUID
    video_link = f"https://s3.ritm.media/yappy-db-duplicates/{uuid_video}.mp4"

    # Скачиваем видеофайл
    download_video(video_link, f"{uuid_video}.mp4")

    # Выполняем проверку на дублирование
    pred_is_dup, pred_uuid = get_res_by_uuid(df, vector_db, model, uuid_video, 0.3)

    return pred_is_dup, pred_uuid


# Подключаемся к RabbitMQ
credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials)
)
channel = connection.channel()

# Объявляем очередь для RPC
channel.queue_declare(queue='rpc_queue')


# Определяем функцию обратного вызова для обработки запросов
def on_request(ch, method, props, body):
    uuid_video = json.loads(body)["uuid_video"]
    print(f" [.] {uuid_video}")

    is_duplicate, duplicated_for = check_video_is_duplicate_by_uuid(uuid_video)

    # Отправляем ответ обратно клиенту
    ch.basic_publish(
        exchange='',
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id=props.correlation_id),
        body=json.dumps({"is_duplicate": is_duplicate, "duplicate_for": duplicated_for})
    )

    # Подтверждаем обработку сообщения
    ch.basic_ack(delivery_tag=method.delivery_tag)


# Настраиваем очередь на получение сообщений
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='rpc_queue', on_message_callback=on_request)

print(" [x] Awaiting RPC requests")
channel.start_consuming()
