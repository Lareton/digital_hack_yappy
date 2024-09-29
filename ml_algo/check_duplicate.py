import os
import sys
import gdown
import pandas as pd
import torch

# Добавляем путь к пользовательскому модулю visil_pytorch
sys.path.append('visil_pytorch/')

# Импортируем необходимые компоненты для работы модели и утилиты
from model.visil import ViSiL
from ml_algo import get_res_by_uuid
from ml_algo import device
# from vector_db_interface import VectorDatabase
from algo_utils import download_video

# Проверка наличия файла индекса "full_index2.pkl", если отсутствует - скачиваем его
if not "full_index2.pkl" in os.listdir():
    gdown.download(url="https://drive.google.com/uc?id=1iRJNmlb7SlWwc8iiurQ6cudjaP8kBaxz")

# Инициализация модели ViSiL с предварительно обученными весами
model_path = 'model_finetuned.pt'
if not model_path in os.listdir():
    gdown.download(url="https://drive.google.com/uc?id=1xVT3EPK7wacnwYeE4PAObM5K_XDu-B1D")
    model = ViSiL(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
else:
    model = ViSiL(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Загрузка и предварительная обработка данных из CSV файла
df = pd.read_csv("cp_vseros_train_1000.csv", index_col=0)
df["created"] = pd.to_datetime(df["created"])
df = df.sort_values(by="created").reset_index(drop=True)

# Инициализация базы данных векторов
vector_db = VectorDatabase("full_index2.pkl")
