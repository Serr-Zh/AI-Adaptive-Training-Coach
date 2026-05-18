# Интеграция Langfuse и LiteLLM в AI Adaptive Training Coach

## 1. Итоговая схема интеграции

После настройки система работает через локальный LiteLLM Proxy, а выполнение запроса дополнительно трассируется в Langfuse.

```text
Пользователь
   ↓
FastAPI endpoint POST /coach
   ↓
llm.py
   ↓
Langfuse SDK: trace / spans / generations
   ↓
LiteLLM Proxy: http://localhost:4000/v1
   ↓
OpenRouter
   ↓
qwen/qwen3.5-35b-a3b
```

В приложении используется логическое имя модели:

```env
LLM_MODEL=coach-model
```

В LiteLLM это имя связано с реальной моделью:

```text
coach-model → openrouter/qwen/qwen3.5-35b-a3b
```


---

## 2. Docker Compose для Langfuse и LiteLLM

Для запуска инфраструктуры используется файл `docker-compose.yaml` в корне проекта. В нём объединены сервисы Langfuse и LiteLLM.

Langfuse разворачивается вместе с зависимостями:

```text
langfuse-web
langfuse-worker
postgres
clickhouse
redis
minio
```

LiteLLM разворачивается как отдельный proxy-сервис с собственной PostgreSQL-базой для хранения виртуальных ключей, бюджетов и usage-данных.

Ключевой фрагмент `docker-compose.yaml`, относящийся к LiteLLM:

```yaml
litellm-postgres:
  image: postgres:17
  restart: always
  environment:
    POSTGRES_DB: litellm
    POSTGRES_USER: litellm
    POSTGRES_PASSWORD: litellm_password
  volumes:
    - litellm_postgres_data:/var/lib/postgresql/data
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U litellm -d litellm"]
    interval: 5s
    timeout: 5s
    retries: 10

litellm:
  image: ghcr.io/berriai/litellm:main-stable
  restart: always
  depends_on:
    litellm-postgres:
      condition: service_healthy
  ports:
    - "4000:4000"
  environment:
    LITELLM_MASTER_KEY: ${LITELLM_MASTER_KEY}
    DATABASE_URL: ${LITELLM_DATABASE_URL}
    OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}
  volumes:
    - ./litellm_config.yaml:/app/config.yaml
  command: ["--config", "/app/config.yaml", "--host", "0.0.0.0", "--port", "4000"]
```

В разделе `volumes` также добавлен том:

```yaml
litellm_postgres_data:
  driver: local
```

Запуск выполняется командой:

```bash
docker compose up -d
```

После запуска доступны веб-интерфейсы:

```text
Langfuse UI: http://localhost:3000
LiteLLM UI:  http://localhost:4000/ui
```

---

## 3. Конфигурация LiteLLM

Для LiteLLM используется файл `litellm_config.yaml`.

```yaml
model_list:
  - model_name: coach-model
    litellm_params:
      model: openrouter/qwen/qwen3.5-35b-a3b
      api_key: os.environ/OPENROUTER_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/LITELLM_DATABASE_URL

litellm_settings:
  drop_params: true
  set_verbose: false
```

В этой конфигурации `coach-model` является алиасом модели. Приложение отправляет запросы к `coach-model`, а LiteLLM перенаправляет их к `openrouter/qwen/qwen3.5-35b-a3b`.

Для работы LiteLLM используются переменные окружения:

```env
OPENROUTER_API_KEY=sk-or-...
LITELLM_MASTER_KEY=sk-litellm-master-local-123456789
LITELLM_DATABASE_URL=postgresql://litellm:litellm_password@litellm-postgres:5432/litellm
```

---

## 4. Подключение приложения к LiteLLM

В приложении используется OpenAI-compatible клиент. Его `base_url`, ключ и имя модели берутся из переменных окружения:

```env
LLM_BASE_URL=http://localhost:4000/v1
LLM_API_KEY=sk-virtual-key-from-litellm
LLM_MODEL=coach-model
LLM_MAX_TOKENS=2048
```

Фрагмент `llm.py`, отвечающий за создание клиента:

```python
def get_training_llm_client() -> AsyncOpenAI:
    global _openai_client

    if _openai_client is None:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")

        if not api_key or not base_url:
            raise RuntimeError(
                "Не заданы LLM_API_KEY или LLM_BASE_URL — проверь .env файл"
            )

        _openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    return _openai_client
```

Так как `LLM_BASE_URL` указывает на `http://localhost:4000/v1`, приложение отправляет запросы не напрямую в OpenRouter, а через LiteLLM.

---

## 5. Интеграция Langfuse в `llm.py`

Для сбора трассировок используется Langfuse SDK и OpenAI-compatible wrapper:

```python
from langfuse import get_client, propagate_attributes
from langfuse.openai import AsyncOpenAI
```

Верхний уровень обработки запроса фиксируется observation `coach_request`:

```python
with langfuse.start_as_current_observation(
    as_type="span",
    name="coach_request",
    input=_to_jsonable(request_data),
    metadata={
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    },
) as root_span:
    with propagate_attributes(
        trace_name="coach_request",
        tags=["ai-training-coach", "langfuse", "manual-spans"],
    ):
        ...
```

Фаза вызова инструментов выделена в отдельный span `tool_calling_phase`:

```python
with langfuse.start_as_current_observation(
    as_type="span",
    name="tool_calling_phase",
    input={
        "request_data": _to_jsonable(request_data),
        "initial_messages": _to_jsonable(messages),
    },
    metadata={
        "model": model_name,
        "max_tokens": max_tokens,
        "tool_count": len(tools),
    },
) as phase_span:
    ...
```

Каждый инструмент фиксируется как отдельное observation с именем вида `tool.<tool_name>`:

```python
with langfuse.start_as_current_observation(
    as_type="tool",
    name=f"tool.{tool_name}",
    input=_to_jsonable(parsed_arguments),
    metadata={"source": "model_function_call"},
) as tool_span:
    result_model = execute_tool(tool_name, parsed_arguments)
    result = result_model.model_dump()
    tool_span.update(output=_to_jsonable(result))
```

Финальная генерация ответа вынесена в `final_response_phase`:

```python
with langfuse.start_as_current_observation(
    as_type="span",
    name="final_response_phase",
    input={
        "dialog": _to_jsonable(dialog),
        "tool_outputs": _to_jsonable(tool_outputs),
    },
    metadata={
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    },
) as final_phase_span:
    raw_answer = await _request_model_response(
        client=client,
        model_name=model_name,
        temperature=temperature,
        messages=dialog,
    )
    final_phase_span.update(output={"raw_answer": raw_answer})
```

Парсинг structured-ответа фиксируется через `sgr_response_parsing`:

```python
with langfuse.start_as_current_observation(
    as_type="span",
    name="sgr_response_parsing",
    input={"raw_answer": raw_answer},
) as parsing_span:
    sgr_data = normalize_sgr_response_shape(
        json.loads(extract_json_from_model_answer(raw_answer))
    )
    sgr_response = CoachSGRResponse(**sgr_data)
    parsing_span.update(
        output={
            "sgr_data": _to_jsonable(sgr_data),
            "sgr_response": _to_jsonable(sgr_response),
        }
    )
```

Дополнительно используется `flush`, чтобы события быстрее попадали в интерфейс Langfuse:

```python
def _flush_langfuse() -> None:
    try:
        get_client().flush()
    except Exception as exc:
        logger.warning("Не удалось выполнить Langfuse flush: %s", exc)
```

---

## 6. Обработка превышения бюджета LiteLLM

При превышении бюджета LiteLLM возвращает ошибку с признаками:

```text
budget_exceeded
Budget has been exceeded
max budget
current cost
```

В `main.py` добавлена отдельная функция для определения такой ошибки:

```python
def is_litellm_budget_exceeded_error(error: Exception) -> bool:
    error_text = str(error).lower()

    budget_error_markers = (
        "budget_exceeded",
        "budget has been exceeded",
        "max budget",
        "current cost",
    )

    return any(marker in error_text for marker in budget_error_markers)
```

В endpoint `/coach` эта ошибка обрабатывается отдельно и возвращается HTTP 402:

```python
except Exception as e:
    if is_litellm_budget_exceeded_error(e):
        raise HTTPException(
            status_code=402,
            detail=(
                "Дневной бюджет LiteLLM превышен. "
                f"Исходная ошибка: {str(e)}"
            ),
        )

    raise HTTPException(
        status_code=500,
        detail=f"Ошибка при обращении к LLM: {str(e)}",
    )
```

Проверка выполнялась с виртуальным ключом, у которого установлен минимальный бюджет. После превышения лимита приложение вернуло:

```text
HTTP/1.1 402 Payment Required
```

---

## 7. Скриншоты Langfuse

### 7.1. Список трассировок

На скриншоте показан список трассировок в Langfuse. Видны события `coach request`, `coach response mapping`, `OpenAI-generation`, а также latency, tokens и total cost.

![Список трассировок в Langfuse](../screenshots/langfuse_traces_list.png)

### 7.2. Observations пайплайна

На скриншоте показана вкладка `Observations`. Видны промежуточные этапы обработки запроса: `tool_calling_phase`, `tool.build_training_context`, `tool.retrieve_training_knowledge`, `tool.assess_training_load`, `tool.assess_restrictions`, `tool.assess_medical_risk`, `final_response_phase`, `sgr_response_parsing`.

![Observations пайплайна в Langfuse](../screenshots/langfuse_observations_pipeline.png)

### 7.3. Детали общего запроса `coach_request`

На скриншоте показан общий trace `coach_request`: входной запрос пользователя, итоговый выход, дерево выполнения, latency, prompt/completion tokens и total cost.

![Детали coach_request](../screenshots/langfuse_coach_request_details.png)

### 7.4. Промежуточный инструмент `tool.retrieve_training_knowledge`

На скриншоте показан промежуточный инструмент поиска знаний. Видны входные данные, сформированный query, найденные документы и отформатированный контекст.

![Промежуточный инструмент retrieve_training_knowledge](../screenshots/langfuse_tool_retrieve_training_knowledge.png)

### 7.5. LLM-вызов `OpenAI-generation`

На скриншоте показан непосредственный вызов модели. Видны input, output, latency, prompt tokens, completion tokens, total tokens и стоимость.

![LLM-вызов OpenAI-generation](../screenshots/langfuse_openai_generation_tokens.png)

---

## 8. Скриншоты LiteLLM

### 8.1. Подключение модели в LiteLLM

На скриншоте показана страница `Models + Endpoints`. Видно, что в LiteLLM подключена модель `coach-model`, которая ссылается на `openrouter/qwen/qwen3.5-35b-a3b`.

![Подключённая модель в LiteLLM](../screenshots/litellm_models_endpoints.png)

### 8.2. Настройки виртуального ключа и бюджет

На скриншоте показана вкладка `Settings` виртуального ключа. Видны модель `coach-model`, максимальный бюджет `$0.01` и параметры ключа.

![Настройки виртуального ключа LiteLLM](../screenshots/litellm_virtual_key_settings_budget.png)

### 8.3. Обзор виртуального ключа и расход

На скриншоте показана вкладка `Overview` виртуального ключа. Видны расход `Spend`, привязанная модель `coach-model` и состояние ключа.

![Overview виртуального ключа LiteLLM](../screenshots/litellm_virtual_key_overview_spend_model.png)

### 8.4. Дневной бюджет

На скриншоте показано, что для ключа задан `Max Budget (USD) = 0.01`, а сброс бюджета установлен как `Daily`.

![Дневной бюджет виртуального ключа LiteLLM](../screenshots/litellm_virtual_key_daily_budget_duration.png)

### 8.5. Проверка HTTP 402 при превышении бюджета

На скриншоте показан терминал с запросом к `/coach`, выполненным через ключ с очень маленьким бюджетом. Приложение вернуло `HTTP/1.1 402 Payment Required`, что подтверждает корректную обработку ошибки бюджета.

![HTTP 402 при превышении бюджета LiteLLM](../screenshots/litellm_budget_exceeded_http_402.png)

---

## 9. Итог интеграции

Langfuse и LiteLLM были развернуты через Docker Compose и подключены к `AI Adaptive Training Coach`.

Langfuse фиксирует трассировки выполнения запроса: входные данные, промежуточные tool-вызовы, финальную генерацию, итоговый ответ, latency и usage токенов.

LiteLLM используется как model gateway: приложение обращается к локальному proxy по адресу `http://localhost:4000/v1`, а не напрямую к OpenRouter. В LiteLLM подключена модель `coach-model`, создан виртуальный ключ с дневным бюджетом, а превышение бюджета обрабатывается приложением как HTTP 402.
