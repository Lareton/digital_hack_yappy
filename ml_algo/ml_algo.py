import sys
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import linalg as LA
import numpy as np
import os
import pandas as pd

# Устройство для выполнения вычислений (используется GPU, если доступен)
device = torch.device('cuda')

# Добавляем путь к пользовательским модулям
sys.path.append('visil_pytorch/')
from utils import load_video


def get_emb_for_video(model, video_frames: np.ndarray) -> torch.Tensor:
    """
    Извлекает вектор признаков (embedding) для видеоролика с использованием указанной модели.

    Параметры:
    ----------
    model : torch.nn.Module
        Предобученная модель, используемая для извлечения признаков.
    video_frames : np.ndarray
        Набор кадров видео в формате numpy массива.

    Возвращает:
    ----------
    torch.Tensor :
        Нормализованный вектор признаков для данного видеоролика.

    Описание:
    --------
    1. Конвертирует входной массив кадров в `torch.Tensor`.
    2. Пропускает через модель без вычисления градиентов (`torch.no_grad()`).
    3. Извлекает и усредняет векторы признаков по всем кадрам.
    4. Нормализует результирующий вектор признаков с помощью нормы L2
    (для упрощения дальнейшего рассчитыавния косинусного сходства)

    Пример использования:
    ---------------------
    >>> emb = get_emb_for_video(model, video_frames)
    >>> print(emb.shape)  # torch.Size([D]), где D — размерность признаков
    """
    # Преобразование numpy массива в tensor и перемещение на устройство
    video_frames = torch.from_numpy(video_frames)
    with torch.no_grad():
        # Извлечение признаков с помощью модели
        embedding = model.extract_features(video_frames.to(device)).cpu()

    # Усреднение векторов признаков по кадрам
    embedding = embedding.mean(dim=0)

    # Нормализация L2 для признаков
    embedding = embedding / LA.vector_norm(embedding)
    return embedding


def get_res_by_uuid(df: pd.DataFrame, vector_db, model, uuid: str, threshold: float) -> tuple:
    """
    Выполняет поиск дубликатов видео на основе UUID и модели.

    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с метаинформацией о видео, включая столбец "uuid".
    vector_db : VectorDatabase
        Объект, представляющий базу данных векторов признаков для видео.
    model : torch.nn.Module
        Предобученная модель для извлечения признаков из видео.
    uuid : str
        Уникальный идентификатор видео, для которого выполняется проверка.
    threshold : float
        Пороговое значение косинусного сходства для определения дубликатов.

    Возвращает:
    ----------
    tuple : (bool, str)
        - (bool): True, если найден дубликат, иначе False.
        - (str): UUID наиболее похожего видео, если дубликат найден.

    Пример использования:
    ---------------------
    >>> is_duplicate, dup_uuid = get_res_by_uuid(df, vector_db, model, "video_uuid", 0.3)
    >>> print(f"Дубликат найден: {is_duplicate}, UUID дубликата: {dup_uuid}")
    """

    # Определение индекса видео на основе его UUID
    index_after_skip = df[df["uuid"] == uuid].index
    if len(index_after_skip) == 0:
        index_after_skip = np.inf
    else:
        index_after_skip = index_after_skip[0]

    # Загрузка видеофайла и извлечение признаков
    path = f"{uuid}.mp4"
    video_frames = load_video(path)
    embedding = get_emb_for_video(model, video_frames)
    embedding = embedding.cpu().numpy()

    # Поиск ближайших соседей с помощью векторной базы данных
    nearest_indexes = vector_db.query(embedding, k=30)

    # Фильтрация соседей, чтобы учитывать только те, чьи индексы меньше текущего
    nearest_indexes = [i for i in nearest_indexes if i[0] < index_after_skip]

    # Расчет косинусного сходства для отфильтрованных соседей
    nearest_indexes = [(i, cosine_similarity([embedding.reshape(-1)], [vec])[0][0]) for i, vec in nearest_indexes]
    nearest_indexes.sort(key=lambda x: x[1], reverse=True)

    # Если нет ближайших соседей, возвращаем False и пустую строку
    if len(nearest_indexes) == 0:
        return False, ""

    # Извлечение ближайшего соседа и проверка порогового значения
    best_index, best_dist = nearest_indexes[0]
    if best_dist > threshold:
        return True, df.iloc[best_index]["uuid"]
    return False, ""
