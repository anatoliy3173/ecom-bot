# ecom-bot (Shoply)

Консольный бот поддержки магазина **Shoply** с:

- диалогом с историей;
- RAG по FAQ через embeddings (семантический поиск);
- командой `/order <id>` по `data/orders.json`;
- логированием сессий в `logs/session_*.jsonl` + учётом токенов (`usage`).

## Быстрый старт

```bash
cd ecom-bot
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
python3 app.py
```

## ProxyAPI.ru

Проект по умолчанию настроен на **ProxyAPI.ru** как OpenAI‑совместимый endpoint:

- `OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1`
- `OPENAI_API_KEY=<ваш ключ ProxyAPI>`

Если хотите использовать прямой OpenAI endpoint, поменяйте:

- `OPENAI_BASE_URL=https://api.openai.com/v1`

