import os
import sys
import gdown
import pandas as pd

sys.path.append('visil_pytorch/')
from model.visil import ViSiL

from ml_algo import get_res_by_uuid
from ml_algo import device
from vector_db_interface import VectorDatabase
from utils import download_video

if not "full_index2.pkl" in os.listdir():
    gdown.download(url="https://drive.google.com/uc?id=1iRJNmlb7SlWwc8iiurQ6cudjaP8kBaxz")

# инициализируем модель
model = ViSiL(pretrained=True).to(device)
model.eval()
df = pd.read_csv("cp_vseros_train_1000.csv", index_col=0)
df["created"] = pd.to_datetime(df["created"])
df = df.sort_values(by="created").reset_index(drop=True)

# подгружаем индексированные видео
vector_db = VectorDatabase("full_index2.pkl")


def check_video_is_duplicate_by_uuid(uuid_video):
    video_link = f"https://s3.ritm.media/yappy-db-duplicates/{uuid_video}.mp4"
    download_video(video_link, f"{uuid_video}.mp4")
    pred_is_dup, pred_uuid = get_res_by_uuid(df, vector_db, model, uuid_video, 0.3)
    return pred_is_dup, pred_uuid


