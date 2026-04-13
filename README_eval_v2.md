
## Evaluation pipeline v2

Новая версия пайплайна оценки запускается напрямую через текущий agent runtime и сохраняет:
- `eval_results.csv`
- `eval_summary.json`
- `eval_report.md`
- сырые ответы по кейсам
- trace вызовов инструментов

### Подготовка

Проверь, что:
- настроены `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`;
- текущий `llm.py` поддерживает `get_sgr_response_with_trace()`.

### Запуск

```bash
python scripts/run_agent_eval.py --input data/eval_cases_v2.json --output-dir results/agent_eval --retriever-eval results/retriever_eval.json
```

### Что делает скрипт

1. Загружает расширенный набор кейсов.
2. Прогоняет агент напрямую.
3. Сохраняет `CoachSGRResponse`, `CoachResponse` и trace инструментов.
4. Считает:
   - success rate;
   - latency;
   - JSON/schema validity;
   - scenario rule accuracy;
   - safety accuracy;
   - restriction compliance;
   - tool coverage;
   - tool precision.
5. Формирует markdown-отчёт.

### Быстрый тест без обращения к LLM

```bash
python tests/test_eval_pipeline.py
```
