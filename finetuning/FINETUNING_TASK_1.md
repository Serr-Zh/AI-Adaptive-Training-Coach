# Fine-tuning pipeline

## Постановка задачи

В рамках задания реализован пайплайн fine-tuning для задачи Question Answering в домене фитнес-рекомендаций.

Модель получает вопрос пользователя о тренировках, физической активности, прогрессе или восстановлении и должна сгенерировать ответ в стиле AI fitness coach.

Практическая постановка задачи: abstractive question answering / supervised instruction fine-tuning.

## Датасет

Используется датасет `its-myrto/fitness-question-answers`.

Структура исходных данных:

- `Question` — вопрос пользователя о фитнесе;
- `Answer` — целевой ответ.

После подготовки данные преобразуются в формат:

- `prompt` — инструкция и вопрос пользователя;
- `completion` — эталонный ответ;
- `text` — объединённая строка для supervised fine-tuning causal language model.

Данные разделены на обучающую и валидационную выборки в пропорции 90/10 с фиксированным random seed.

Итоговое разделение:

- train: 868 примеров;
- validation: 97 примеров.

## Стратегия экспериментов

Выбрана стратегия прогрессии.

Порядок базовых экспериментов:

1. `Qwen/Qwen2.5-0.5B-Instruct`
2. `Qwen/Qwen2.5-1.5B-Instruct`
3. `HuggingFaceTB/SmolLM2-1.7B-Instruct`

Дополнительно для анализа влияния гиперпараметров были проведены эксперименты на лучшей базовой модели `Qwen/Qwen2.5-1.5B-Instruct`:

1. `qwen_1_5b_lora_r16` — увеличен LoRA-rank с `r=8` до `r=16`;
2. `qwen_1_5b_lora_lr1e4` — уменьшен learning rate с `2e-4` до `1e-4`.

## Метод fine-tuning

Используется PEFT-подход с LoRA.

LoRA выбрана, потому что:

- датасет содержит менее 10 000 примеров;
- full fine-tuning требует существенно больше вычислительных ресурсов;
- PEFT позволяет обучать только небольшое число дополнительных параметров;
- обученные адаптеры сохраняются отдельно от базовой модели.

Базовая LoRA-конфигурация:

```text
r = 8
lora_alpha = 16
lora_dropout = 0.05
target_modules = q_proj, k_proj, v_proj, o_proj
learning_rate = 2e-4
```

## Структура реализации

```text
finetuning/
├── configs/
│   ├── qwen_0_5b_qlora.yaml
│   ├── qwen_1_5b_qlora.yaml
│   ├── smollm2_1_7b_qlora.yaml
│   ├── qwen_1_5b_lora_r16.yaml
│   └── qwen_1_5b_lora_lr1e4.yaml
├── data/
│   └── processed/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── dataset_metadata.json
├── outputs/
│   ├── adapters/
│   ├── metrics/
│   ├── predictions/
│   └── plots/
├── prepare_dataset.py
├── train_lora.py
├── plot_price_quality.py
├── FINETUNING_TASK_1.md
└── FINETUNING_REPORT.md
```

Примечание: часть технических имён файлов содержит суффикс `qlora`, однако фактически в проведённых экспериментах использовалась LoRA без 4-bit quantization: `use_4bit=false`.

## Запуск подготовки данных

```bash
python finetuning/prepare_dataset.py
```

## Запуск обучения

Установка зависимостей:

```bash
pip install -r requirements-finetuning-gpu.txt
```

Запуск базовых экспериментов:

```bash
python finetuning/train_lora.py --config finetuning/configs/qwen_0_5b_qlora.yaml
python finetuning/train_lora.py --config finetuning/configs/qwen_1_5b_qlora.yaml
python finetuning/train_lora.py --config finetuning/configs/smollm2_1_7b_qlora.yaml
```

Запуск дополнительных экспериментов с гиперпараметрами:

```bash
python finetuning/train_lora.py --config finetuning/configs/qwen_1_5b_lora_r16.yaml
python finetuning/train_lora.py --config finetuning/configs/qwen_1_5b_lora_lr1e4.yaml
```

## Результаты

После обучения сохраняются:

```text
finetuning/outputs/adapters/
finetuning/outputs/metrics/
finetuning/outputs/plots/
```

## Ограничения

Локальная машина не имеет NVIDIA GPU и CUDA, поэтому подготовка данных и конфигураций выполнялась локально, а обучение моделей выполнялось в GPU-среде.

Основные вычислительные ограничения:

- GPU: Tesla T4;
- GPU memory: 14.56 GB;
- training precision: fp16;
- PEFT method: LoRA;
- full fine-tuning не использовался из-за повышенных требований к памяти.
