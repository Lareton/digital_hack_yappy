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
from algo_utils import download_video

# Проверка наличия файла индекса "full_index2.pkl", если отсутствует - скачиваем его
if not "full_index2.pkl" in os.listdir():
    # gdown.download(url="https://drive.google.com/uc?id=1iRJNmlb7SlWwc8iiurQ6cudjaP8kBaxz")
    gdown.download(url="https://drive.google.com/uc?id=1q4NOo3ZwcBWt1-OMZ5oh2HCfaL9WJbo4")

# Инициализация модели ViSiL с предварительно обученными весами
model = ViSiL(pretrained=True).to(device)
model.eval()

# Загрузка и предварительная обработка данных из CSV файла
df = pd.read_csv("cp_vseros_train_1000.csv", index_col=0)
df["created"] = pd.to_datetime(df["created"])
df = df.sort_values(by="created").reset_index(drop=True)

# Инициализация базы данных векторов
vector_db = VectorDatabase("full_index2.pkl")
