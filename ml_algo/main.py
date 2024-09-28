from algo_utils import download_video
from ml_algo import get_res_by_uuid
from check_duplicate import model, df, vector_db


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



if __name__ == '__main__':
    print(check_video_is_duplicate_by_uuid("6d3233b6-f8de-49ba-8697-bb30dbf825f7"))
