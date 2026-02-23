from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@dataclass
class Settings:
    openai_api_key: str
    openai_base_url: str
    chat_model: str
    embedding_model: str
    brand_name: str
    max_history_messages: int = 10
    max_completion_tokens: int = 200
    faq_similarity_threshold: float = 0.6


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


def init_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def init_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        max_tokens=settings.max_completion_tokens,
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


def build_faq_vector_store(
    embeddings: OpenAIEmbeddings,
    faq_items: List[Dict[str, str]],
) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embeddings)
    docs: List[Document] = []
    for idx, item in enumerate(faq_items):
        q = item.get("q", "")
        a = item.get("a", "")
        docs.append(
            Document(
                page_content=f"Вопрос: {q}\nОтвет: {a}",
                metadata={"id": idx, "q": q, "a": a},
            )
        )
    store.add_documents(docs)
    return store


def retrieve_faq_context(
    settings: Settings,
    faq_store: InMemoryVectorStore,
    query: str,
    max_items: int = 3,
) -> Tuple[str, List[int]]:
    results = faq_store.similarity_search_with_score(query=query, k=max_items)

    top_indices: List[int] = []
    lines: List[str] = ["Релевантные ответы из FAQ (используй их, если они подходят):"]
    for doc, score in results:
        if score < settings.faq_similarity_threshold:
            continue
        doc_id = doc.metadata.get("id")
        if isinstance(doc_id, int):
            top_indices.append(doc_id)
        q = doc.metadata.get("q", "")
        a = doc.metadata.get("a", "")
        lines.append(f"- Вопрос: {q}")
        lines.append(f"  Ответ: {a}")

    if not top_indices:
        return "", top_indices

    return "\n".join(lines), top_indices


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
    llm: ChatOpenAI,
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

    system_message = SystemMessage(content=" ".join(system_parts))

    context_messages: List[BaseMessage] = []
    if faq_context:
        context_messages.append(SystemMessage(content=f"FAQ контекст:\n{faq_context}"))
    if order_context:
        context_messages.append(SystemMessage(content=f"Контекст по заказу:\n{order_context}"))

    truncated_history = truncate_history(history, settings.max_history_messages)

    lc_history: List[BaseMessage] = []
    for msg in truncated_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            lc_history.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_history.append(AIMessage(content=content))

    messages: List[BaseMessage] = [system_message]
    messages.extend(context_messages)
    messages.extend(lc_history)
    messages.append(HumanMessage(content=user_message))

    ai_message: AIMessage = llm.invoke(messages)
    content = ai_message.content or ""

    response_metadata = getattr(ai_message, "response_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}

    usage: Dict[str, int] = {
        "prompt_tokens": int(token_usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(token_usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(token_usage.get("total_tokens", 0) or 0),
    }

    return content.strip(), usage


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    settings = load_settings()
    embeddings = init_embeddings(settings)
    llm = init_llm(settings)

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

    faq_store = build_faq_vector_store(embeddings, faq_raw)

    history: List[Dict[str, str]] = []
    last_order_id: Optional[str] = None
    last_order_data: Optional[Dict[str, Any]] = None
    turn_index = 0

    print(f"Добро пожаловать в поддержку {settings.brand_name}!")
    print("Задайте вопрос или используйте команды:")
    print("  /order <id> — статус заказа")
    print("  /exit       — выход")

    try:
        while True:
            try:
                user_input = input(f"Вы({turn_index + 1:02d}): ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            user_input_lower = user_input.lower()
            if user_input_lower in {"/exit", "exit", "quit", "стоп", "stop"}:
                turn_index += 1
                print("Бот: До свидания!")
                append_jsonl(
                    session_log_path,
                    {
                        "ts": datetime.now().isoformat(),
                        "type": "exchange",
                        "turn": turn_index,
                        "user_label": f"Вы({turn_index:02d})",
                        "assistant_label": "Бот",
                        "user": user_input,
                        "assistant": "До свидания!",
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "retrieval": {},
                    },
                )
                break

            turn_index += 1
            user_label = f"Вы({turn_index:02d})"
            assistant_label = "Бот"

            order_context = ""
            faq_context = ""
            retrieval_info: Dict[str, Any] = {}

            order_match = re.match(r"^/order\s+(\S+)$", user_input, flags=re.IGNORECASE)
            if order_match:
                order_id = order_match.group(1)
                order_data = orders.get(order_id)
                order_context = build_order_context(order_id, order_data)
                last_order_id = order_id
                last_order_data = order_data
                retrieval_info["order_id"] = order_id
                retrieval_info["order_found"] = order_data is not None
            else:
                faq_context, faq_indices = retrieve_faq_context(
                    settings=settings,
                    faq_store=faq_store,
                    query=user_input,
                )
                retrieval_info["faq_indices"] = faq_indices
                if last_order_id is not None:
                    order_context = build_order_context(last_order_id, last_order_data)
                    retrieval_info["order_id"] = last_order_id
                    retrieval_info["order_found"] = last_order_data is not None

            answer, usage_chat = generate_answer(
                llm=llm,
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

            print(f"{assistant_label}: {answer}")

            exchange_record: Dict[str, Any] = {
                "ts": datetime.now().isoformat(),
                "type": "exchange",
                "turn": turn_index,
                "user_label": user_label,
                "assistant_label": assistant_label,
                "user": user_input,
                "assistant": answer,
                "usage": usage_chat,
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

