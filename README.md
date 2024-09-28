<p align="center"><img align=center src="images/README_IMAGES/selling_pandas_LOGO.png" width="508" alt="Selling Pandas"/></p>
<h1 align="center">Сервис по определению дубликатов видео</h1>

## Основные функции

- `Проверка видеофайлов на наличие нарушений авторских прав.`
- `Загрузка видео в базу лицензионных видео.`

## Основная структура
Сервис "Дубль" предназначен для обнаружения дубликатов видео с целью соблюдения авторских прав. Основной подход решения включает использование передовых методов машинного обучения, таких как модели ViSiL (Video Similarity Learning) и S2VS (Self Supervised Video Similarity), что позволяет эффективно определять схожесть видеоконтента. 

В процессе обработки строится матрица сходства (SSM) на основе признаков каждого кадра, а для анализа применяется метод Cross-Region Attention. Для повышения точности решения была внедрена дообучаемая модель с контекстно-зависимым сравнением, которая достигает F1 score 0.98. Оптимизация архитектуры позволяет достичь высокой скорости обработки (2.5 секунды на видео) и гибкости в масштабировании благодаря предварительному индексированию и разделению логики ML и API.

![](images/README_IMAGES/ml_pipeline.PNG)

## Пайплайн API
Архитектура сервиса "Дубль" разделена на две основные части: ML-обработку и API-логику, что обеспечивает гибкость и высокую отказоустойчивость. Основное преимущество заключается в возможности предварительного индексирования данных, что позволяет существенно сократить время обработки и повысить скорость работы сервиса, не увеличивая время отклика при росте объема базы.

![](images/README_IMAGES/architecture.jpg)

***Проект построен на принципах микросервисной архитектуры с целью обеспечить надежность и дальнейшую масштабируемость.***

## Используемые технологии

### RabbitMQ

- RabbitMQ является мощной и надежной системой обмена сообщениями. Она обеспечивает асинхронную коммуникацию между
  сервисами, что важно для построения отказоустойчивых и масштабируемых микросервисных архитектур.

### PostgreSQL

- PostgreSQL является одной из самых мощных и надежных реляционных баз данных с открытым исходным кодом.

### Qrand

- Qrand современная векторная база данных, помогающая хранить и поддерживать огромное количество размеченных видео

### REST API

- REST API обеспечивает стандартизированный подход к построению веб-сервисов, что делает их легко доступными и
  интегрируемыми с различными клиентами и сервисами.

### Docker-compose

- Docker-compose помогает инкапсулированно развертывать приложения на сервере при деплое. Так же дает огромные
  возможности в рамках добавления сложных CI/CD инструментов

## Документация
Алгоритмы ML расположены в [ml_algo](ml_algo)

Функционал и пайплайны разделены по различным модулям:
Код обучения и валидации основных моделей. Все модели и функции документированы 
1. Использование SSL модели [notebook](ml_train/s2vs_model_using.ipynb) 
2. Использование ViSiL модели [notebook](ml_train/ViSiL_finetuning.ipynb) 
3. Локальная валидации (сбалансирована относительно генеральной совокупности и не пересекается с ней) [.csv](ml_train/cp_vseros_train_1000.csv) 
4. Функция скачивания видео [module](ml_algo/utils.py) 
5. Функция инференса модели для получения индекса и ближайших видео [module](ml_algo/ml_algo.py) 
6. Функция, реализующая конечный функционал ручки api - скачивание видео, проверку и формирование результата [module](ml_algo/check_duplicate.py) 

## Установка

1. Клонируйте репозиторий:

```shell
git clone git@github.com:ShadowP1e/lct-hackathon.git
```

2. Перейдите в папку проекта

```shell
cd lct-hackathon
```

3. Переименуйте файл .env.sample в .env

```shell
mv .env.sample .env
```

4. Запустите сервисы:

```shell
docker compose up --build
```

## Swager

#### Swager API лежит по [этой](http://localhost:5000/docs) ссылке после развертывания приложения 

## Команда

Проект разработан командой Selling Pandas в рамках хакатона Цифровой прорыв, кейс Yappy.