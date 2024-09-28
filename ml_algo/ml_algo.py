import sys
import torch

from sklearn.metrics.pairwise import cosine_similarity
from torch import linalg as LA

import numpy as np
import os

device = torch.device('cuda')

# sys.path.append('visil_pytorch/')
from visil_pytorch.utils import load_video

def get_emb_for_video(model, video_frames):
    video_frames = torch.from_numpy(video_frames)
    with torch.no_grad():
        embedding = model.extract_features(video_frames.to(device)).cpu()
    embedding = embedding.mean(dim=0)
    embedding = embedding / LA.vector_norm(embedding)
    return embedding


def get_res_by_uuid(df, vector_db, model, uuid, threshold):
    # определяем
    index_after_skip = df[df["uuid"] == uuid].index
    if len(index_after_skip) == 0:
        index_after_skip = np.inf
    else:
        index_after_skip = index_after_skip[0]

    path = f"{uuid}.mp4"
    video_frames = load_video(path)
    embedding = get_emb_for_video(model, video_frames)
    embedding = embedding.cpu().numpy()

    nearest_indexes = vector_db.query(embedding, k=30)
    nearest_indexes = [i for i in nearest_indexes if i[0] < index_after_skip]

    nearest_indexes = [(i, cosine_similarity([embedding.reshape(-1)], [vec])[0][0]) for i, vec in nearest_indexes]
    nearest_indexes.sort(key=lambda x: x[1], reverse=True)
    print("nearest: ", nearest_indexes)

    if len(nearest_indexes) == 0:
        return False, ""

    best_index, best_dist = nearest_indexes[0]
    if best_dist > threshold:
        return True, df.iloc[best_index]["uuid"]
    return False, ""


