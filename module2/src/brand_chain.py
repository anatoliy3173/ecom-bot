from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent

# ── Env ────────────────────────────────────────────────────────────────
load_dotenv(BASE / ".env", override=True)

# ── Style guide (YAML) ────────────────────────────────────────────────
with (BASE / "data" / "style_guide.yaml").open("r", encoding="utf-8") as _f:
    STYLE: dict = yaml.safe_load(_f)

# ── Few-shot examples ─────────────────────────────────────────────────
_few_shots: list[dict[str, str]] = []
with (BASE / "data" / "few_shots.jsonl").open("r", encoding="utf-8") as _f:
    for _line in _f:
        _line = _line.strip()
        if _line:
            _few_shots.append(json.loads(_line))

# ── Data (FAQ + orders) ───────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


_faq: List[Dict[str, str]] = _load_json(BASE / "data" / "faq.json")
_orders: Dict[str, Any] = _load_json(BASE / "data" / "orders.json")

# ── LLM ────────────────────────────────────────────────────────────────
_llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    temperature=0.3,
)


# ── Pydantic model for structured output ───────────────────────────────
class BrandReply(BaseModel):
    answer: str = Field(description="краткий ответ клиенту")
    tone: str = Field(description="контроль: совпадает ли тон (да/нет) + одна фраза почему")
    actions: list[str] = Field(default_factory=list, description="список следующих шагов для клиента (0-3 пункта)")


# ── Helpers ─────────────────────────────────────────────────────────────

def _build_faq_context() -> str:
    """Include all FAQ items in prompt (dataset is small)."""
    lines = ["Данные из FAQ магазина (используй только их для ответа на вопросы):"]
    for item in _faq:
        lines.append(f"- Вопрос: {item['q']}")
        lines.append(f"  Ответ: {item['a']}")
    return "\n".join(lines)


def _build_order_context(order_id: str) -> str:
    order_data = _orders.get(order_id)
    if order_data is None:
        return f"Заказа с номером {order_id} нет в системе. Вежливо сообщи об этом."

    status = order_data.get("status", "unknown")
    parts = [f"Информация о заказе {order_id}:"]

    if status == "in_transit":
        parts.append("Статус: в пути.")
        if eta := order_data.get("eta_days"):
            parts.append(f"Ожидаемый срок доставки: {eta} дней.")
        if carrier := order_data.get("carrier"):
            parts.append(f"Служба доставки: {carrier}.")
    elif status == "delivered":
        parts.append("Статус: доставлен.")
        if dt := order_data.get("delivered_at"):
            parts.append(f"Дата доставки: {dt}.")
    elif status == "processing":
        parts.append("Статус: в обработке.")
        if note := order_data.get("note"):
            parts.append(f"Комментарий: {note}.")
    else:
        parts.append(f"Статус: {status}.")

    parts.append("Используй только эти данные при ответе про заказ.")
    return " ".join(parts)


def _build_system_prompt(order_id: Optional[str] = None) -> str:
    tone = STYLE["tone"]
    brand = STYLE["brand"]
    fallback = STYLE["fallback"]["no_data"]

    parts = [
        f"Ты — бот поддержки интернет-магазина {brand}.",
        f"Характер: {tone['persona']}.",
        f"Отвечай строго на русском языке.",
        f"Максимум {tone['sentences_max']} предложения в ответе.",
        "Всегда обращайся к клиенту на «Вы».",
        "Никогда не используй разговорных сокращений, даже если клиент их употребляет.",
    ]

    if tone.get("bullets"):
        parts.append("Используй маркированные списки для шагов и вариантов.")

    if tone.get("avoid"):
        parts.append(f"Строго избегай: {', '.join(tone['avoid'])}.")

    if tone.get("must_include"):
        parts.append(f"Обязательно включай: {', '.join(tone['must_include'])}.")

    parts.append(f'Если нет данных для ответа, говори: "{fallback}"')

    # FAQ context
    parts.append("")
    parts.append(_build_faq_context())

    # Order context
    if order_id:
        parts.append("")
        parts.append(_build_order_context(order_id))

    return "\n".join(parts)


# ── Main ask() function ───────────────────────────────────────────────

def ask(
    question: str,
    order_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> BrandReply:
    system_text = _build_system_prompt(order_id)
    messages: list[BaseMessage] = [SystemMessage(content=system_text)]

    # Few-shot examples
    for shot in _few_shots:
        messages.append(HumanMessage(content=shot["user"]))
        messages.append(AIMessage(content=shot["assistant"]))

    # Chat history (last 10 turns)
    if history:
        for msg in history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    structured_llm = _llm.with_structured_output(BrandReply, method="function_calling")
    return structured_llm.invoke(messages)


# ── Demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_questions = [
        "Как оформить возврат?",
        "Какой статус заказа 55555?",
    ]
    for q in demo_questions:
        order_id = None
        if "55555" in q:
            order_id = "55555"
        reply = ask(q, order_id=order_id)
        print(f"В: {q}")
        print(f"Ответ: {reply.answer}")
        print(f"Тон: {reply.tone}")
        print(f"Действия: {reply.actions}")
        print()
