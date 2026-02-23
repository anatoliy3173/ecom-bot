from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class Settings:
    openai_api_key: str
    openai_base_url: str
    chat_model: str
    embedding_model: str
    brand_name: str
    max_history_messages: int = 10
    max_completion_tokens: int = 200


def load_settings() -> Settings:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please create a .env file based on .env.example.")

    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
    chat_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    brand_name = os.getenv("BRAND_NAME", "Shoply")

    return Settings(
        openai_api_key=api_key,
        openai_base_url=openai_base_url,
        chat_model=chat_model,
        embedding_model=embedding_model,
        brand_name=brand_name,
    )


def init_openai_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_logs_dir(base_dir: Path) -> Path:
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def create_session_log_file(logs_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"session_{timestamp}.jsonl"


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def embed_text(client: OpenAI, settings: Settings, text: str) -> Tuple[List[float], Dict[str, int]]:
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    embedding = response.data[0].embedding
    usage_data: Dict[str, int] = {
        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
        "completion_tokens": 0,
        "total_tokens": getattr(response.usage, "total_tokens", 0),
    }
    return embedding, usage_data


def prepare_faq_with_embeddings(
    client: OpenAI,
    settings: Settings,
    faq_items: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    items_with_embeddings: List[Dict[str, Any]] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for item in faq_items:
        question = item.get("q", "")
        embedding, usage = embed_text(client, settings, question)
        items_with_embeddings.append(
            {
                "q": question,
                "a": item.get("a", ""),
                "embedding": embedding,
            }
        )
        for key in total_usage:
            total_usage[key] += usage.get(key, 0)

    return items_with_embeddings, total_usage


def retrieve_faq_context(
    client: OpenAI,
    settings: Settings,
    faq_items: List[Dict[str, Any]],
    query: str,
    max_items: int = 3,
    similarity_threshold: float = 0.75,
) -> Tuple[str, List[int], Dict[str, int]]:
    query_embedding, usage = embed_text(client, settings, query)

    scored: List[Tuple[int, float]] = []
    for idx, item in enumerate(faq_items):
        item_embedding = item.get("embedding")
        if not isinstance(item_embedding, list):
            continue
        score = cosine_similarity(query_embedding, item_embedding)
        scored.append((idx, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    top_indices: List[int] = []
    for idx, score in scored[:max_items]:
        if score < similarity_threshold:
            continue
        top_indices.append(idx)

    if not top_indices:
        return "", top_indices, usage

    lines: List[str] = ["Релевантные ответы из FAQ (используй их, если они подходят):"]
    for idx in top_indices:
        item = faq_items[idx]
        lines.append(f"- Вопрос: {item.get('q', '')}")
        lines.append(f"  Ответ: {item.get('a', '')}")

    context_text = "\n".join(lines)
    return context_text, top_indices, usage


def build_order_context(order_id: str, order_data: Optional[Dict[str, Any]]) -> str:
    if order_data is None:
        return f"Информация о заказе: заказа с номером {order_id} нет в системе. Объясни это вежливо пользователю."

    status = order_data.get("status", "unknown")
    parts: List[str] = [f"Информация о заказе {order_id}:"]
    if status == "in_transit":
        eta_days = order_data.get("eta_days")
        carrier = order_data.get("carrier")
        parts.append("Статус: в пути.")
        if isinstance(eta_days, int):
            parts.append(f"Ожидаемый срок доставки: {eta_days} дней.")
        if isinstance(carrier, str):
            parts.append(f"Служба доставки: {carrier}.")
    elif status == "delivered":
        delivered_at = order_data.get("delivered_at")
        parts.append("Статус: доставлен.")
        if isinstance(delivered_at, str):
            parts.append(f"Дата доставки: {delivered_at}.")
    elif status == "processing":
        parts.append("Статус: в обработке.")
        note = order_data.get("note")
        if isinstance(note, str):
            parts.append(f"Комментарий: {note}.")
    else:
        parts.append(f"Статус: {status}.")

    parts.append("Используй только эти данные при ответе на вопросы про этот заказ.")
    return " ".join(parts)


def truncate_history(
    history: List[Dict[str, str]],
    max_messages: int,
) -> List[Dict[str, str]]:
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def generate_answer(
    client: OpenAI,
    settings: Settings,
    history: List[Dict[str, str]],
    user_message: str,
    faq_context: str,
    order_context: str,
) -> Tuple[str, Dict[str, int]]:
    system_parts: List[str] = [
        f"Ты — вежливый и лаконичный бот поддержки интернет-магазина {settings.brand_name}.",
        "Отвечай по-русски, 1–3 коротких предложения.",
        "Используй только информацию из контекстов FAQ и заказов, а также из предыдущего диалога.",
        "Если нужной информации нет, честно скажи, что не знаешь, и предложи обратиться в поддержку, не выдумывай детали.",
    ]

    system_message = {"role": "system", "content": " ".join(system_parts)}

    context_messages: List[Dict[str, str]] = []
    if faq_context:
        context_messages.append(
            {
                "role": "system",
                "content": f"FAQ контекст:\n{faq_context}",
            }
        )
    if order_context:
        context_messages.append(
            {
                "role": "system",
                "content": f"Контекст по заказу:\n{order_context}",
            }
        )

    truncated_history = truncate_history(history, settings.max_history_messages)

    messages: List[Dict[str, str]] = [system_message]
    messages.extend(context_messages)
    messages.extend(truncated_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        max_completion_tokens=settings.max_completion_tokens,
    )

    content = response.choices[0].message.content or ""
    usage_raw = response.usage
    usage: Dict[str, int] = {
        "prompt_tokens": getattr(usage_raw, "prompt_tokens", 0),
        "completion_tokens": getattr(usage_raw, "completion_tokens", 0),
        "total_tokens": getattr(usage_raw, "total_tokens", 0),
    }
    return content.strip(), usage


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    settings = load_settings()
    client = init_openai_client(settings)

    data_dir = base_dir / "data"
    faq_raw: List[Dict[str, str]] = load_json(data_dir / "faq.json")
    orders: Dict[str, Any] = load_json(data_dir / "orders.json")

    logs_dir = ensure_logs_dir(base_dir)
    session_log_path = create_session_log_file(logs_dir)

    usage_total: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    faq_with_embeddings, faq_usage = prepare_faq_with_embeddings(client, settings, faq_raw)
    for key in usage_total:
        usage_total[key] += faq_usage.get(key, 0)

    history: List[Dict[str, str]] = []

    print(f"Добро пожаловать в поддержку {settings.brand_name}!")
    print("Задайте вопрос или используйте команды:")
    print("  /order <id> — статус заказа")
    print("  /exit       — выход")

    try:
        while True:
            try:
                user_input = input("> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in {"/exit", "exit", "quit"}:
                print("Спасибо, что обратились в поддержку. Хорошего дня!")
                break

            order_context = ""
            faq_context = ""
            retrieval_info: Dict[str, Any] = {}

            order_match = re.match(r"^/order\s+(\S+)$", user_input, flags=re.IGNORECASE)
            if order_match:
                order_id = order_match.group(1)
                order_data = orders.get(order_id)
                order_context = build_order_context(order_id, order_data)
                retrieval_info["order_id"] = order_id
                retrieval_info["order_found"] = order_data is not None
            else:
                faq_context, faq_indices, faq_query_usage = retrieve_faq_context(
                    client=client,
                    settings=settings,
                    faq_items=faq_with_embeddings,
                    query=user_input,
                )
                retrieval_info["faq_indices"] = faq_indices
                for key in usage_total:
                    usage_total[key] += faq_query_usage.get(key, 0)

            answer, usage_chat = generate_answer(
                client=client,
                settings=settings,
                history=history,
                user_message=user_input,
                faq_context=faq_context,
                order_context=order_context,
            )

            for key in usage_total:
                usage_total[key] += usage_chat.get(key, 0)

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})

            print(answer)

            exchange_record: Dict[str, Any] = {
                "ts": datetime.now().isoformat(),
                "type": "exchange",
                "user": user_input,
                "assistant": answer,
                "usage": {
                    "chat": usage_chat,
                },
                "retrieval": retrieval_info,
            }
            append_jsonl(session_log_path, exchange_record)

    except KeyboardInterrupt:
        print("\nДиалог завершён по Ctrl+C.")

    summary_record: Dict[str, Any] = {
        "ts": datetime.now().isoformat(),
        "type": "summary",
        "usage_total": usage_total,
    }
    append_jsonl(session_log_path, summary_record)

    print(
        f"Суммарные токены за сессию — prompt: {usage_total['prompt_tokens']}, "
        f"completion: {usage_total['completion_tokens']}, total: {usage_total['total_tokens']}."
    )


if __name__ == "__main__":
    main()

