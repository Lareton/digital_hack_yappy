from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams

HOST = "localhost"
VECTOR_SIZE = 128
VECTOR_DISTANCE = "Cosine"
VECTOR_CONFIG = VectorParams(size=VECTOR_SIZE, distance=VECTOR_DISTANCE)

# Connect to a local Qdrant instance (running on localhost:6333)
client = QdrantClient(host=HOST, port=6333)

if not client.collection_exists("licensed_videos"):
    client.create_collection(
        collection_name="licensed_videos",
        vectors_config=VECTOR_CONFIG)

import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def download_video(link, output_dir):
    """Загружает видео по ссылке и сохраняет его в указанную папку."""

    filename = link.split('/')[-1]
    try:
        response = requests.get(link, stream=True)
        response.raise_for_status()

        with open(f'{output_dir}/{filename}', 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        print(f'Видео {filename} успешно сохранено.')

    except requests.exceptions.RequestException as e:
        print(f'Ошибка загрузки видео {filename}: {e}')


def download_videos(links, output_dir, max_workers=5):
    """Загружает видео по ссылкам из DataFrame с использованием многопотока.

    Args:
        df: DataFrame с колонкой 'link', содержащей ссылки на видео.
        output_dir: Путь к папке для сохранения видео.
        max_workers: Максимальное количество потоков (по умолчанию 5).
    """

    # Создаем папку, если ее нет
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, link in enumerate(links):
            executor.submit(download_video, link, output_dir)


download_videos(links, 'downloaded_videos')

if __name__ == '__main__':
    print(client.collection_exists("licensed_videos"))
