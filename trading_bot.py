import argparse
import json
from pathlib import Path
from typing import Dict

import requests


def load_positions(path: str | None) -> Dict[str, int]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {ticker: int(quantity) for ticker, quantity in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Безопасная paper-trading утилита: строит план ребалансировки, но не отправляет реальные заявки."
    )
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL FastAPI сервиса")
    parser.add_argument("--cash", type=float, default=100000.0, help="Размер портфеля в рублях")
    parser.add_argument("--positions", help="JSON-файл текущих позиций, например {'SBER': 10}")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--top-pool", type=int, default=30)
    parser.add_argument("--target-assets", type=int, default=8)
    parser.add_argument("--method", default="ensemble")
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--universe-limit", type=int, default=60)
    args = parser.parse_args()

    payload = {
        "cash": args.cash,
        "current_positions": load_positions(args.positions),
        "lookback_days": args.lookback_days,
        "top_pool": args.top_pool,
        "target_assets": args.target_assets,
        "method": args.method,
        "max_weight": args.max_weight,
        "universe_limit": args.universe_limit,
    }

    response = requests.post(f"{args.api_url.rstrip('/')}/moex/rebalance-plan", json=payload, timeout=120)
    response.raise_for_status()
    plan = response.json()

    print("\nВАЖНО: это paper-trading план. MOEX ISS API не умеет выставлять реальные заявки.")
    print(f"Капитал: {plan['cash']} руб.")
    print(f"Оценочно инвестировано: {plan['estimated_invested']} руб.")
    print(f"Остаток денег: {plan['estimated_cash_left']} руб.\n")
    print("План ребалансировки:")
    for order in plan["orders"]:
        print(
            f"{order['ticker']:8} {order['action']:4} "
            f"delta={order['delta_lots']:5} target={order['target_lots']:5} "
            f"price={order['last_price']:10.4f} weight={order['target_weight']:.4f}"
        )


if __name__ == "__main__":
    main()
