<p align="center"><img align=center src="images/README_IMAGES/selling_pandas_LOGO.png" width="508" alt="Selling Pandas"/></p>
<h1 align="center">Сервис по определению дубликатов видео</h1>

## Основные функции

- `Проверка видеофайлов на наличие нарушений авторских прав.`
- `Загрузка видео в базу лицензионных видео.`

## Основная структура

TODO

## Алгоритм

TODO

## Пайплайн API

TODO from MIRO

## Архитектура WEB-Сервиса

![Architecture](images/README_IMAGES/architecture.jpg)

***Проект построен на принципах микросервисной архитектуры с целью обеспечить надежность и дальнейшую масштабируемость.
***

## Используемые технологии

### RabbitMQ

- RabbitMQ является мощной и надежной системой обмена сообщениями. Она обеспечивает асинхронную коммуникацию между
  сервисами, что важно для построения отказоустойчивых и масштабируемых микросервисных архитектур.

### PostgreSQL

- PostgreSQL является одной из самых мощных и надежных реляционных баз данных с открытым исходным кодом.

### Qrand

- Qrand современная векторная база данных, помогающая хранить и поддерживать огромное количество размеченных видео

### FastAPI

- FastAPI используется для разработки бэкенда. Этот фреймворк обеспечивает высокую производительность и быструю
  разработку благодаря асинхронной архитектуре.

### REST API

- REST API обеспечивает стандартизированный подход к построению веб-сервисов, что делает их легко доступными и
  интегрируемыми с различными клиентами и сервисами.

### Docker-compose

- Docker-compose помогает инкапсулированно развертывать приложения на сервере при деплое. Так же дает огромные
  возможности
  в рамках добавления сложных CI/CD инструментов

**Выбор данных технологий обусловлен их надежностью, производительностью и удобством в использовании. FastAPI
обеспечивает создание быстрого, масштабируемого и простого в поддержке микросервиса. Использование REST API и Swagger
улучшает взаимодействие с API и облегчает интеграцию с другими системами. Использование Docker и Docker Compose упрощает
процесс развертывания и управления многокомпонентной системой, обеспечивая консистентность среды и легкость
масштабирования.**

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

## Документация

#### Swager API лежит по [этой](http://localhost:5000/docs) ссылке после развертывания приложения 

## Команда

Проект разработан командой Selling Pandas в рамках хакатона Цифровой прорыв, кейс Yappy.