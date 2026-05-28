# Нагрузочное тестирование API (Locust)

Этот документ описывает, как запускать нагрузочное тестирование для сервиса `AI Adaptive Training Coach` с помощью `Locust`.

## Что именно тестируется

- Endpoint `GET /info` (вес задачи: 1)
- Endpoint `POST /run` (вес задачи: 10)
- Формат ответа `/run`: ожидается JSON со `status = "success"`

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

Если Locust еще не установлен:

```bash
pip install "locust>=2.0"
```

## Запуск теста

### Вариант 1: Web UI

```bash
locust -f locustfile.py --host http://localhost:8000
```

Далее открыть UI: `http://localhost:8089`

### Вариант 2: Headless (рекомендуется для отчетов)

```bash
locust -f locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 -t 60s
```

Где:

- `-u` — число виртуальных пользователей
- `-r` — скорость спауна пользователей в секунду
- `-t` — длительность теста

## Рекомендуемые профили запуска

- Smoke: `-u 1 -r 1 -t 60s`
- Short load: `-u 3 -r 1 -t 120s`
- Basic load: `-u 10 -r 2 -t 60s`
- Endurance light: `-u 10 -r 2 -t 180s`

## Сохранение лога в файл

PowerShell:

```powershell
locust -f locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 -t 60s |
  Tee-Object -FilePath "artifacts/locust/locust_10_users_60s.txt"
```

## Критерии успешного прогона

- В конце прогона `Shutting down (exit code 0)`
- В сводке `# fails = 0` для `GET /info` и `POST /run`
- Нет секции `Error report` или она пустая

## Типовые причины падения

1. `status = "error"` в `/run` из-за лимитов внешней LLM (429/402).
2. Сервис возвращает невалидный JSON.
3. `input_type` в `/info` отличается от ожидаемого (`text`).

Важно: текущий `locustfile.py` проверяет `input_type == "text"`.

## Режим для стабильного нагрузочного теста

Для изоляции от внешней LLM используйте `LOAD_TEST_MODE=true`.

В проекте это уже задано:

- `.env` (`LOAD_TEST_MODE=true`)
- `docker-compose.yaml` для сервиса `app`

В этом режиме `/run` возвращает тестовый ответ и не вызывает внешнюю модель.


