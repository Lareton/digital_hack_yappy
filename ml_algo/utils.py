import requests
import wget

def download_video(link, filename):
    """Загружает видео по ссылке и сохраняет его в указанную папку."""

    try:
        wget.download(link,filename)
        print(f'Видео {filename} успешно сохранено.')

    except requests.exceptions.RequestException as e:
        print(f'Ошибка загрузки видео {filename}: {e}')
