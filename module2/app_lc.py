from __future__ import annotations

import re
import sys
from pathlib import Path

# Add src/ to path so brand_chain is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from brand_chain import STYLE, BrandReply, ask


def main() -> None:
    brand = STYLE.get("brand", "Shoply")
    print(f"Добро пожаловать в поддержку {brand}!")
    print("Команды: /order <id> — статус заказа, /exit — выход\n")

    history: list[dict[str, str]] = []
    last_order_id: str | None = None

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "exit", "quit", "стоп", "stop"}:
            print("До свидания!")
            break

        # Parse /order command
        order_id: str | None = None
        order_match = re.match(r"^/order\s+(\S+)$", user_input, re.IGNORECASE)
        if order_match:
            order_id = order_match.group(1)
            last_order_id = order_id
        elif last_order_id and re.search(r"заказ", user_input, re.IGNORECASE):
            order_id = last_order_id

        reply: BrandReply = ask(user_input, order_id=order_id, history=history)

        print(f"\nОтвет: {reply.answer}")
        print(f"Тон:   {reply.tone}")
        if reply.actions:
            print("Действия:")
            for action in reply.actions:
                print(f"  - {action}")
        print()

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply.answer})


if __name__ == "__main__":
    main()
