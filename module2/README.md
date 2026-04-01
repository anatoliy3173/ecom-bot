# Module 2: Brand Voice Bot (Shoply)

Бренд-ассистент магазина **Shoply** с единым стилем ответов на базе LangChain.

## Возможности

- **Голос бренда** — правила стиля загружаются из `data/style_guide.yaml`
- **Few-shot примеры** — автоматически подмешиваются из `data/few_shots.jsonl`
- **Структурированный вывод** — Pydantic-модель `BrandReply` (answer, tone, actions)
- **Автооценка стиля** — `style_eval.py` проверяет ответы и сохраняет отчёт в JSON
- **FAQ + заказы** — бот отвечает только по данным из `data/faq.json` и `data/orders.json`

## Быстрый старт

```bash
cd module2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # впишите OPENAI_API_KEY
```

## Запуск

```bash
# Демо цепочки
python src/brand_chain.py

# Чат-бот в стиле бренда
python app_lc.py

# Автооценка стиля (сохранит reports/style_eval.json)
python src/style_eval.py
```

## API

Проект использует **OpenRouter** как OpenAI-совместимый endpoint:

- `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
- `OPENAI_MODEL=openai/gpt-4o-mini`

## Структура

```
module2/
  app_lc.py              # Интерактивный чат-бот
  src/
    brand_chain.py       # Ядро: YAML-конфиг, Pydantic-модель, ask()
    style_eval.py        # Автооценка стиля + JSON-отчёт
  data/
    style_guide.yaml     # Голос бренда (YAML)
    few_shots.jsonl       # Few-shot примеры
    eval_prompts.txt      # 10 тестовых промптов
    faq.json             # FAQ магазина
    orders.json          # Данные заказов
  reports/
    style_eval.json      # Автогенерируемый отчёт
```
