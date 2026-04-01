from __future__ import annotations

import json
import os
import re
import statistics
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from brand_chain import BASE, STYLE, BrandReply, ask

load_dotenv(BASE / ".env", override=True)
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)


# ── Rule-based checks (before LLM) ────────────────────────────────────

def rule_checks(text: str) -> int:
    score = 100
    # 1) No emoji
    if re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", text):
        score -= 20
    # 2) No screaming "!!!"
    if "!!!" in text:
        score -= 10
    # 3) Length guard
    if len(text) > 600:
        score -= 10
    return max(score, 0)


# ── LLM-based grading ─────────────────────────────────────────────────

class Grade(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Оценка от 0 до 100")
    notes: str = Field(..., description="Краткое пояснение оценки")


_grade_llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    temperature=0,
)

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ты — строгий ревьюер соответствия голосу бренда {brand}.\n"
     "Тон: {persona}.\n"
     "Избегать: {avoid}.\n"
     "Обязательно: {must_include}.\n"
     "Максимум предложений: {max_sentences}.\n"
     "Обращение к клиенту должно быть на «Вы».\n\n"
     "Оцени ответ ассистента по шкале 0-100 и дай краткие заметки."),
    ("human",
     "Вопрос клиента:\n{question}\n\nОтвет ассистента:\n{answer}"),
])


def llm_grade(question: str, answer: str) -> Grade:
    chain = GRADE_PROMPT | _grade_llm.with_structured_output(Grade, method="function_calling")
    return chain.invoke({
        "brand": STYLE["brand"],
        "persona": STYLE["tone"]["persona"],
        "avoid": ", ".join(STYLE["tone"]["avoid"]),
        "must_include": ", ".join(STYLE["tone"]["must_include"]),
        "max_sentences": STYLE["tone"]["sentences_max"],
        "question": question,
        "answer": answer,
    })


# ── Batch evaluation ──────────────────────────────────────────────────

def eval_batch(prompts: List[str]) -> dict:
    results = []
    for p in prompts:
        # Detect order ID in prompt
        order_id = None
        m = re.search(r"[Зз]аказ[а-я]*\s+(\d+)", p)
        if m:
            order_id = m.group(1)

        reply: BrandReply = ask(p, order_id=order_id)
        rule = rule_checks(reply.answer)
        g = llm_grade(p, reply.answer)
        final = int(0.4 * rule + 0.6 * g.score)

        results.append({
            "prompt": p,
            "answer": reply.answer,
            "actions": reply.actions,
            "tone_model": reply.tone,
            "rule_score": rule,
            "llm_score": g.score,
            "final": final,
            "notes": g.notes,
        })
        print(f"  [{final:3d}] {p}")

    mean_final = round(statistics.mean(r["final"] for r in results), 2)
    out = {"mean_final": mean_final, "items": results}
    report_path = REPORTS / "style_eval.json"
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    eval_prompts = (
        (BASE / "data" / "eval_prompts.txt")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    print(f"Evaluating {len(eval_prompts)} prompts...\n")
    report = eval_batch(eval_prompts)
    print(f"\nСредний балл: {report['mean_final']}")
    print(f"Отчёт: {REPORTS / 'style_eval.json'}")
