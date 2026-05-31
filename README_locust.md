# Нагрузочное тестирование API (Locust)

Этот документ описывает, как запускать нагрузочное тестирование для сервиса `AI Adaptive Training Coach` с помощью `Locust` по шаблону `locust-api-template`.

## Что именно тестируется

- Endpoint `GET /health` — возвращает `{"status":"ok"}`
- Endpoint `GET /info` — возвращает `input_type`, `input_schema`, `output_schema`
- Endpoint `POST /run` — принимает `content` (строка или список) + `extra_body`, возвращает `{"status":"success","result":...,"error":null}`

Логика нагрузки находится в `locustfile.py`.

## Предварительные условия

1. Запущен Docker Desktop.
2. Поднят сервис API на `http://localhost:8000`.
3. Установлен Locust в локальном Python-окружении.

### Поднять сервис

```bash
docker compose up --build -d
```

Проверка:

```bash
curl http://localhost:8000/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

## Установка Locust

Если Locust еще не установлен на хосте:

```bash
pip install "locust>=2.0"
```

## Запуск теста

Есть три способа запуска Locust.

### Вариант 1: Locust на хосте (рекомендуется, соответствует заданию)

Сначала поднимите сервис:

```bash
docker compose up --build -d
```

Web-интерфейс:

```bash
locust -f locustfile.py --host http://localhost:8000
```

Открыть UI: `http://localhost:8089`

Headless:

```bash
locust -f locustfile.py --host http://localhost:8000 \
  --headless -u 10 -r 2 -t 60s
```

### Вариант 2: Отдельный Docker-контейнер Locust

Не требует установки Locust на хост. Использует тот же Dockerfile:

```bash
# Headless
docker compose run locust --headless -u 10 -r 2 -t 60s --only-summary

# Web-интерфейс
docker compose run -p 8089:8089 locust
```

### Вариант 3: Через app-контейнер

Если Locust недоступен на хосте и нет отдельного контейнера:

```bash
docker compose -f docker-compose.yaml exec -T app locust -f locustfile.py \
  --host http://127.0.0.1:8000 --headless -u 1 -r 1 -t 20s --only-summary
```

## Рекомендуемые профили запуска

- Smoke: `-u 1 -r 1 -t 60s`
- Short load: `-u 2 -r 1 -t 20s`
- Basic load: `-u 10 -r 2 -t 60s`

## Сохранение лога в файл

PowerShell:

```powershell
locust -f locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 -t 60s --only-summary |
  Tee-Object -FilePath "artifacts/locust/locust_10_users_60s.txt"
```

## Критерии успешного прогона

- В конце прогона `Shutting down (exit code 0)`
- В сводке `# fails = 0` для `GET /info` и `POST /run`
- Нет секции `Error report` или она пустая

## Режимы работы

### LOAD_TEST_MODE=true (без LLM)

В `.env` установлен `LOAD_TEST_MODE=true`. В этом режиме `/run` возвращает стабильный тестовый ответ без вызова внешней LLM. Используется для проверки API-контракта, сериализации и устойчивости сервиса.

### LOAD_TEST_MODE=false (реальный вызов LLM)

Для проверки что реальные LLM-вызовы работают через `/run`, временно переключите:

```bash
# В .env: LOAD_TEST_MODE=false
docker compose up --build -d app
```

Каждый запрос к `/run` вызывает LLM через LiteLLM с tool-calling, что занимает ~30-50 секунд. При 10 пользователях за 60 секунд проходит ~10 запросов (вместо ~460 с заглушкой). Стоимость: ~18₽ за прогон.

После проверки верните `LOAD_TEST_MODE=true` и пересоберите.

## Структура файлов для Locust

| Файл | Назначение |
|---|---|
| `app/main.py` | FastAPI endpoints: `/health`, `/info`, `/run` |
| `app/models.py` | `InfoResponse`, `RunRequest`, `RunResponse`, `InputType`, `ContentPart` |
| `app/adapter.py` | Адаптер Locust payload → CoachRequest |
| `locustfile.py` | Locust-сценарий |


