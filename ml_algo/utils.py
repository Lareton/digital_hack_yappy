import requests


def download_video(link, filename):
    """Загружает видео по ссылке и сохраняет его в указанную папку."""

    try:
        response = requests.get(link, stream=True)
        response.raise_for_status()

        with open(f'{filename}', 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        print(f'Видео {filename} успешно сохранено.')

    except requests.exceptions.RequestException as e:
        print(f'Ошибка загрузки видео {filename}: {e}')
