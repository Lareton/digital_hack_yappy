import requests

def download_video(link: str, filename: str) -> None:
    """
    Загружает видеофайл по указанной ссылке и сохраняет его с заданным именем файла.

    Параметры:
    ----------
    link : str
        Ссылка на видеофайл для загрузки.
    filename : str
        Имя файла, под которым будет сохранено видео на локальном диске.


    Пример использования:
    ---------------------
    >>> download_video("https://example.com/video.mp4", "sample_video.mp4")

    Сообщения в консоли:
    --------------------
    - При успешной загрузке: "Видео sample_video.mp4 успешно сохранено."
    - В случае ошибки: "Ошибка загрузки видео sample_video.mp4: <описание ошибки>"
    """

    try:
        # Отправка HTTP-запроса для загрузки файла
        response = requests.get(link, stream=True)
        response.raise_for_status()

        # Запись загруженного файла на диск по частям (размер каждой части 1 МБ)
        with open(f'{filename}', 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:  # Проверяем, что полученная часть не пустая
                    f.write(chunk)

        # Вывод сообщения об успешной загрузке
        print(f'Видео {filename} успешно сохранено.')

    except requests.exceptions.RequestException as e:
        # Обработка ошибок, связанных с HTTP-запросами
        print(f'Ошибка загрузки видео {filename}: {e}')
