import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
import gdown

# Добавляем путь к пользовательскому модулю visil_pytorch
sys.path.append('visil_pytorch/')

# Импортируем необходимые компоненты для работы модели и утилиты
from model.visil import ViSiL
from ml_algo import get_res_by_uuid
from ml_algo import device
from vector_db_interface import VectorDatabase

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

# Переменная для хранения результатов проверки
results = []

# Проход по всем строкам датафрейма с помощью tqdm для отображения прогресса
for ind, i in tqdm(df.iterrows(), total=len(df)):
    """
    Основной цикл для проверки всех видеороликов на наличие дубликатов.

    Параметры:
    ----------
    ind : int
        Индекс текущей строки.
    i : pd.Series
        Текущая строка датафрейма, содержащая информацию о видео.

    Описание работы:
    ---------------
    1. Для каждого видео выполняется проверка, является ли оно дубликатом с помощью функции `get_res_by_uuid`.
    2. Сравнивается предсказание модели (`pred_is_dup`) и оригинальная метка из датафрейма (`i["is_duplicate"]`).
    3. Если предсказание совпадает с оригинальной меткой:
       - Если видео определено как дубликат, проверяется, совпадает ли предсказанный UUID с оригинальным дубликатом (`i["duplicate_for"]`).
       - В противном случае, считается корректным, если модель предсказала отсутствие дубликата.
    4. Результаты (1 — верное предсказание, 0 — неверное) сохраняются в список `results`.
    5. Выводится текущее значение средней точности (Accuracy).
    """

    # Выполняем проверку на дублирование
    pred_is_dup, pred_uuid = get_res_by_uuid(df, vector_db, model, i["uuid"], 0.3)

    # Вывод промежуточного результата для текущего видео
    print(f"Проверка видео UUID: {i['uuid']}, предсказание: {pred_is_dup}, метка: {i['is_duplicate']}, корректность: {(pred_is_dup and pred_uuid == i['duplicate_for'])}")

    # Сравнение предсказания с меткой дубликата
    if pred_is_dup == i["is_duplicate"]:
        if pred_is_dup:
            # Если предсказание определяет как дубликат, сравниваем с ожидаемым дубликатом
            results.append(int(pred_is_dup and pred_uuid == i["duplicate_for"]))
        else:
            # Если предсказано отсутствие дубликата — верное предсказание
            results.append(1)
    else:
        # Если предсказание отличается от метки — неверное предсказание
        results.append(0)

    # Вывод текущей средней точности (Accuracy)
    print(f"Текущая средняя точность: {np.mean(results):.4f}")
