import os
import sys
import gdown
import pandas as pd

# Добавляем путь к пользовательскому модулю visil_pytorch
sys.path.append('visil_pytorch/')

# Импортируем необходимые компоненты для работы модели и утилиты
from model.visil import ViSiL
from ml_algo import get_res_by_uuid
from ml_algo import device
from vector_db_interface import VectorDatabase
from utils import download_video

# Проверка наличия файла индекса "full_index2.pkl", если отсутствует - скачиваем его
if not "full_index2.pkl" in os.listdir():
    gdown.download(url="https://drive.google.com/uc?id=1iRJNmlb7SlWwc8iiurQ6cudjaP8kBaxz")

# Инициализация модели ViSiL с предварительно обученными весами
model = ViSiL(pretrained=True).to(device)
model.eval()

# Загрузка и предварительная обработка данных из CSV файла
df = pd.read_csv("cp_vseros_train_1000.csv", index_col=0)
df["created"] = pd.to_datetime(df["created"])
df = df.sort_values(by="created").reset_index(drop=True)

# Инициализация базы данных векторов
vector_db = VectorDatabase("full_index2.pkl")


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
    2. Видео скачивается с помощью утилиты `download_video`.
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
