from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def load_examples(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt_static(
    examples: list[dict[str, str]],
    example_prompt: PromptTemplate,
) -> FewShotPromptTemplate:
    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=(
            "Ты — бот-помощник по программированию на Python. "
            "Давай полезные советы и всегда показывай примеры кода.\n"
            "Вот примеры хороших ответов:"
        ),
        suffix="Вопрос: {question}\n",
        input_variables=["question"],
        example_separator="\n---\n",
    )


def build_prompt_with_selector(
    examples: list[dict[str, str]],
    example_prompt: PromptTemplate,
    api_key: str,
    base_url: str,
) -> FewShotPromptTemplate:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url=base_url,
    )

    selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings=embeddings,
        vectorstore_cls=InMemoryVectorStore,
        k=2,
    )

    return FewShotPromptTemplate(
        example_selector=selector,
        example_prompt=example_prompt,
        prefix=(
            "Ты — бот-помощник по программированию на Python. "
            "Давай полезные советы и всегда показывай примеры кода.\n"
            "Вот примеры хороших ответов:"
        ),
        suffix="Вопрос: {question}\n",
        input_variables=["question"],
        example_separator="\n---\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot Python tips bot")
    parser.add_argument(
        "--use-selector",
        action="store_true",
        help="Использовать SemanticSimilarityExampleSelector (выбирает 2 самых релевантных примера)",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Create a .env file based on .env.example.")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    base_dir = Path(__file__).resolve().parent
    examples = load_examples(base_dir / "examples.yaml")

    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="Вопрос: {question}\n{answer}",
    )

    if args.use_selector:
        print("Режим: SemanticSimilarityExampleSelector (k=2)")
        few_shot_prompt = build_prompt_with_selector(
            examples, example_prompt, api_key, base_url,
        )
    else:
        print(f"Режим: все примеры ({len(examples)} шт.)")
        few_shot_prompt = build_prompt_static(examples, example_prompt)

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=500,
    )

    print("Бот программных советов (Python)")
    print("Задайте вопрос о Python. Введите 'exit' для выхода.\n")

    try:
        while True:
            try:
                user_input = input("Вы: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "выход"}:
                print("До свидания!")
                break

            formatted = few_shot_prompt.format(question=user_input)
            response = llm.invoke(formatted)
            print(f"\nБот:\n{response.content}\n")

    except KeyboardInterrupt:
        print("\nДо свидания!")


if __name__ == "__main__":
    main()
