import pandas as pd
from tqdm import tqdm
import numpy as np
import sys

# sys.path.append('visil_pytorch/')
from visil_pytorch.model.visil import ViSiL

from digital_hack_yappy.ml_algo.ml_algo import get_res_by_uuid
from digital_hack_yappy.ml_algo.ml_algo import device
from digital_hack_yappy.ml_algo.vector_db_interface import VectorDatabase

# инициализируем модель
model = ViSiL(pretrained=True).to(device)
model.eval()
df = pd.read_csv("cp_vseros_train_1000.csv", index_col=0)
df["created"] = pd.to_datetime(df["created"])
df = df.sort_values(by="created").reset_index(drop=True)

# подгружаем индексированные видео
vector_db = VectorDatabase("full_index2.pkl")


results = []
for ind, i in tqdm(df.iterrows()):
    # print(i["uuid"], i["is_duplicate"], i["duplicate_for"])
    # if not  i["is_duplicate"]:
    # continue

    pred_is_dup, pred_uuid = get_res_by_uuid(df, vector_db, model, i["uuid"], 0.3)

    print(pred_is_dup, i["is_duplicate"], (pred_is_dup and pred_uuid == i["duplicate_for"]))

    if pred_is_dup == i["is_duplicate"]:
        if pred_is_dup:
            results.append(int((pred_is_dup and pred_uuid == i["duplicate_for"])))
        else:
            results.append(1)
    else:
        results.append(0)

    print(np.mean(results))


