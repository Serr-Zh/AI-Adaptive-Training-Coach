# AI Adaptive Training Coach

AI Adaptive Training Coach - API-сервис для генерации стартовых тренировочных планов и адаптации следующей тренировки по истории пользователя, текущей сессии, признакам восстановления, ограничениям и safety-правилам.

Система построена как LLM-агент с инструментами: сначала собирает проверяемые факты через локальные tools и retrieval, затем формирует структурированный ответ по SGR-схеме (Schema-Guided Reasoning) и возвращает компактный API-ответ `CoachResponse`.

## Что умеет проект

- Генерирует стартовый тренировочный план по цели, уровню подготовки, оборудованию и ограничениям.
- Адаптирует следующую сессию по истории тренировок и текущей тренировке.
- Выявляет признаки перегрузки: недовыполнение подходов, высокий RPE, недостаток сна, высокая усталость.
- Учитывает ограничения пользователя: колени, поясница, плечи, шея и другие текстовые ограничения.
- Оформляет безопасный отказ при признаках медицинского риска: острая боль, травма, резкая боль, онемение и похожие маркеры.
- Использует локальную базу знаний `data/knowledge_base.json` через простой lexical retriever.
- Пишет trace вызовов инструментов и LLM-этапов в Langfuse.
- Поддерживает Docker Compose окружение с FastAPI, LiteLLM, Langfuse и инфраструктурными сервисами.
- Имеет eval-пайплайн, retrieval benchmark, Locust-нагрузочное тестирование и отдельные материалы по fine-tuning.

## Архитектура

| Компонент | Файл | Назначение |
|---|---|---|
| FastAPI API | `main.py` | HTTP endpoints: `/coach`, `/health`, `/schema`, `/info`, `/run` |
| Pydantic-модели | `models.py` | Входной контракт, SGR-схема, итоговый `CoachResponse`, trace-модели |
| LLM runtime | `llm.py` | Двухфазный agent runtime, structured output, Langfuse spans, fallback-логика |
| Prompts | `prompts.py` | Tool orchestration prompt и финальный SGR prompt |
| Tools | `tools.py` | Локальные инструменты анализа контекста, нагрузки, ограничений и medical risk |
| Retriever | `retriever.py` | Лексический поиск по локальной базе знаний |
| Knowledge base | `data/knowledge_base.json` | Доменные знания для training decisions |
| LiteLLM config | `litellm_config.yaml` | Роутинг модели `coach-model` через LiteLLM |
| Docker | `Dockerfile`, `docker-compose.yaml` | Контейнеризация приложения и инфраструктуры |
| Load testing | `locustfile.py`, `README_locust.md` | Нагрузочный тест по unified API `/info` + `/run` |
| Evaluation | `evaluation_pipeline.py`, `scripts/run_agent_eval.py` | Автоматическая оценка качества агента |

## Как работает agent runtime

1. API получает `CoachRequest` на `POST /coach`.
2. `llm.py` запускает tool-calling phase.
3. Модель может вызвать инструменты `build_training_context`, `retrieve_training_knowledge`, `assess_restrictions`, `assess_training_load`, `assess_medical_risk`, `request_confirmation`.
4. Если function calling недоступен или провайдер возвращает ошибку, система запускает локальный fallback pipeline через те же tools.
5. Runtime гарантирует наличие обязательных tool outputs для финального решения.
6. Финальная LLM-фаза формирует `CoachSGRResponse` по строгой SGR-схеме.
7. Ответ нормализуется, валидируется Pydantic-моделями и преобразуется в публичный `CoachResponse`.
8. Langfuse получает trace по этапам `coach_request`, `tool_calling_phase`, `final_response_phase`, `sgr_response_parsing` и tool spans.

## API

| Метод | Endpoint | Назначение |
|---|---|---|
| `GET` | `/` | Краткая информация о сервисе |
| `GET` | `/health` | Health check, возвращает `{"status":"ok"}` |
| `GET` | `/schema` | JSON Schema итогового `CoachResponse` |
| `POST` | `/coach` | Основной endpoint тренировочного coach-агента |
| `GET` | `/info` | Metadata endpoint для Locust/API template |
| `POST` | `/run` | Unified endpoint для Locust/API template |

### Основной запрос `/coach`

Пример стартового плана:

```json
{
  "user_profile": {
    "goal": "hypertrophy",
    "experience_level": "intermediate",
    "equipment": ["штанга", "гантели", "тренажёры", "турник"],
    "restrictions": []
  },
  "session_history": [],
  "current_session": null,
  "temperature": 0.3
}
```

Пример адаптации после тренировки:

```json
{
  "user_profile": {
    "goal": "strength",
    "experience_level": "intermediate",
    "equipment": ["штанга", "силовая рама", "скамья"],
    "restrictions": ["без рывковых движений"]
  },
  "session_history": [],
  "current_session": {
    "date": "2025-01-10",
    "exercises": [
      {
        "name": "Жим лёжа",
        "sets_planned": 5,
        "sets_completed": 4,
        "reps": "5/5/5/3",
        "weight_kg": 80,
        "rpe": 9
      }
    ],
    "sleep_hours": 5.5,
    "fatigue_level": 8,
    "notes": "Сплю плохо вторую неделю, завтра планировал присед"
  },
  "temperature": 0.3
}
```

Публичный ответ имеет форму:

```json
{
  "mode": "initial_plan",
  "session_assessment": null,
  "next_session": {
    "decision": "...",
    "exercise_changes": [],
    "reasoning": "..."
  },
  "long_term_recommendation": "...",
  "safety_warnings": [],
  "refused": false,
  "refuse_reason": null
}
```

## Быстрый старт через Docker Compose

Скопируйте шаблон окружения:

```powershell
Copy-Item .env.example .env
```

Для Linux/macOS:

```bash
cp .env.example .env
```

Заполните в `.env` реальные ключи для LLM-провайдера. Минимально важные переменные:

| Переменная | Для чего нужна |
|---|---|
| `POLZA_AI_API_KEY` | Ключ провайдера, используемый LiteLLM config |
| `LLM_BASE_URL` | OpenAI-compatible endpoint для приложения |
| `LLM_API_KEY` | Ключ для обращения приложения к LiteLLM/OpenAI-compatible API |
| `LLM_MODEL` | Имя модели, например `coach-model` |
| `LLM_MAX_TOKENS` | Лимит токенов ответа |
| `LOAD_TEST_MODE` | Стабовый режим для `/run` в нагрузочных тестах |
| `LANGFUSE_BASE_URL` | URL Langfuse для observability |
| `LANGFUSE_PUBLIC_KEY` | Публичный ключ Langfuse |
| `LANGFUSE_SECRET_KEY` | Секретный ключ Langfuse |
| `LITELLM_MASTER_KEY` | Master key для LiteLLM |
| `LITELLM_DATABASE_URL` | PostgreSQL DSN для LiteLLM |
| `DATABASE_URL` | PostgreSQL DSN, который читает LiteLLM config |

Для проверки `/health`, `/info` и `/run` в `LOAD_TEST_MODE=true` внешний LLM-ключ не нужен. Для реальной работы `/coach` нужен валидный ключ провайдера модели.

Поднимите окружение:

```bash
docker compose up --build -d
```

Проверьте API:

```bash
curl http://localhost:8000/health
```

Swagger UI доступен по адресу `http://localhost:8000/docs`.

Полезные порты:

| Сервис | URL |
|---|---|
| FastAPI app | `http://localhost:8000` |
| LiteLLM | `http://localhost:4000` |
| Langfuse UI | `http://localhost:3000` |
| Locust UI | `http://localhost:8089` при отдельном запуске Locust |

Остановить окружение:

```bash
docker compose down
```

## Локальный запуск без Docker

Создайте окружение и установите зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Запустите API:

```bash
python run.py
```

Локальный запуск требует валидный `.env`. Если вы не используете локальный LiteLLM, укажите прямой OpenAI-compatible `LLM_BASE_URL`, `LLM_API_KEY` и `LLM_MODEL`.

## Примеры запросов

Через готовый JSON-файл:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/coach" -Method Post -ContentType "application/json" -InFile "examples/01_initial_plan.json"
```

Через `curl`:

```bash
curl -X POST http://localhost:8000/coach \
  -H "Content-Type: application/json" \
  --data @examples/02_adaptation.json
```

Примеры входов лежат в `examples/`:

| Файл | Сценарий |
|---|---|
| `examples/01_initial_plan.json` | Генерация стартового плана |
| `examples/02_adaptation.json` | Адаптация после тренировки |
| `examples/03_medical_refusal.json` | Безопасный отказ при medical risk |

## Нагрузочное тестирование

Проект поддерживает задание формата `locust-api-template`: `/info` описывает тип входа, `/run` принимает единый payload `content + extra_body`.

Базовый headless-прогон:

```bash
locust -f locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 -t 60s
```

В текущем проекте `/run` работает с текстовым `content`. При `LOAD_TEST_MODE=true` endpoint `/run` возвращает стабильный тестовый ответ и не вызывает внешнюю LLM. Это удобно для проверки API-контракта, сериализации и устойчивости сервиса без влияния rate limits и бюджета модели.

Подробная инструкция находится в `README_locust.md`.

## Оценка качества

### Retriever benchmark

```bash
python scripts/evaluate_retriever.py
```

Скрипт использует `benchmark/queries.jsonl`, `benchmark/qrels/test.tsv`, сохраняет run в `benchmark/runs/current_lexical.tsv` и метрики в `results/retriever_eval.json`.

### Agent evaluation

```bash
python scripts/run_agent_eval.py --input data/eval_cases_v2.json --output-dir results/agent_eval --retriever-eval results/retriever_eval.json
```

Eval-пайплайн сохраняет:

- `eval_results.csv`
- `eval_summary.json`
- `eval_report.md`
- сырые ответы по кейсам
- trace вызовов инструментов

Подробнее: `README_eval_v2.md`.

## Быстрые проверки

Проверка eval-логики без обращения к LLM:

```bash
python tests/test_eval_pipeline.py
```

Проверка retriever:

```bash
python tests/test_retriever.py
```

Проверка подключения к LLM:

```bash
python scripts/test_llm_connection.py
```

Сценарии tool-calling с сохранением trace:

```bash
python scripts/run_tool_agent_scenarios.py
```

## Данные и артефакты

| Путь | Назначение |
|---|---|
| `data/knowledge_base.json` | База знаний для retrieval |
| `data/eval_cases_v2.json` | Основной набор eval-кейсов для agent runtime |
| `data/validation_cases.json` | Набор для API-eval через `/coach` |
| `data/synthetic_eval_dataset.*` | Синтетический датасет для оценки |
| `benchmark/` | Retrieval benchmark и локальные LLM benchmark-материалы |
| `results/` | Результаты eval и trace-прогонов |
| `artifacts/` | Отчеты и учебные артефакты проекта |
| `finetuning/` | Скрипты и отчеты по fine-tuning/LoRA |

## Структура проекта

```text
.
├── main.py                    # FastAPI app
├── models.py                  # API, SGR and tool schemas
├── llm.py                     # Agent runtime and Langfuse tracing
├── prompts.py                 # Tool and final prompts
├── tools.py                   # Local tool implementations
├── retriever.py               # Lexical retriever over local knowledge base
├── locustfile.py              # Locust load test scenario
├── locust_models.py           # Unified API models for /info and /run
├── locust_adapter.py          # Adapter from Locust payload to CoachRequest
├── data/                      # Knowledge base, eval cases and datasets
├── examples/                  # Ready-to-send API request examples
├── scripts/                   # Eval, benchmark and utility scripts
├── tests/                     # Local checks and scenario tests
├── results/                   # Generated eval outputs and traces
├── artifacts/                 # Reports and project artifacts
├── benchmark/                 # Retriever and inference benchmarks
├── finetuning/                # Fine-tuning scripts and reports
├── Dockerfile
├── docker-compose.yaml
├── litellm_config.yaml
├── README_locust.md
├── README_eval_v2.md
└── requirements.txt
```

## Safety-ограничения

Система не является медицинским сервисом и не предназначена для диагностики, лечения или назначения реабилитации. Если во входе есть признаки острой боли, травмы, онемения или состояния, мешающего движению, агент должен вернуть безопасный отказ (`refused=true`) и не давать тренировочную рекомендацию.

