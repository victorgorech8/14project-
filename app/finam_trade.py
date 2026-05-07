import os
from typing import Any, Dict, List


class FinamConfigError(RuntimeError):
    pass


def is_finam_configured() -> bool:
    return bool(os.getenv("FINAM_TOKEN") and os.getenv("FINAM_ACCOUNT_ID"))


def get_finam_status() -> Dict[str, Any]:
    return {
        "configured": is_finam_configured(),
        "account_id_present": bool(os.getenv("FINAM_ACCOUNT_ID")),
        "token_present": bool(os.getenv("FINAM_TOKEN")),
        "trading_enabled": os.getenv("ENABLE_REAL_TRADING", "false").lower() == "true",
        "symbol_format": "TICKER@MISX, например SBER@MISX",
        "safety_note": "Реальные заявки отправляются только если ENABLE_REAL_TRADING=true и confirm_real_trading=true.",
    }


async def get_finam_account_info() -> Dict[str, Any]:
    client, account_id = await _get_client()
    info = await client.account.get_account_info(account_id)
    return _to_plain(info)


async def place_finam_orders(orders: List[Dict[str, Any]], confirm_real_trading: bool = False) -> Dict[str, Any]:
    trading_enabled = os.getenv("ENABLE_REAL_TRADING", "false").lower() == "true"
    if not trading_enabled or not confirm_real_trading:
        return {
            "mode": "dry_run",
            "submitted": False,
            "reason": "Для реальной отправки заявок нужны ENABLE_REAL_TRADING=true и confirm_real_trading=true.",
            "orders": orders,
        }

    client, account_id = await _get_client()
    submitted = []
    for order in orders:
        quantity = int(order.get("quantity", 0))
        if quantity <= 0:
            continue
        side = str(order.get("side", "BUY")).upper()
        ticker = str(order["ticker"]).upper()
        price = order.get("price")
        response = await _place_order(
            client=client,
            account_id=account_id,
            ticker=ticker,
            side=side,
            quantity=quantity,
            price=price,
        )
        submitted.append({"ticker": ticker, "side": side, "quantity": quantity, "response": _to_plain(response)})

    return {"mode": "real_trading", "submitted": True, "orders": submitted}


async def _get_client():
    token = os.getenv("FINAM_TOKEN")
    account_id = os.getenv("FINAM_ACCOUNT_ID")
    if not token or not account_id:
        raise FinamConfigError("Нужны переменные окружения FINAM_TOKEN и FINAM_ACCOUNT_ID")

    try:
        from finam_trade_api import Client, TokenManager
    except ImportError as exc:
        raise FinamConfigError("Установите зависимость: pip install finam-trade-api") from exc

    client = Client(TokenManager(token))
    await client.access_tokens.set_jwt_token()
    return client, account_id


async def _place_order(client: Any, account_id: str, ticker: str, side: str, quantity: int, price: Any):
    from finam_trade_api.base_client import FinamDecimal
    from finam_trade_api.base_client.models import Side
    from finam_trade_api.order import Order, OrderType, TimeInForce

    order_type = OrderType.MARKET if price is None else OrderType.LIMIT
    order = Order(
        account_id=account_id,
        symbol=f"{ticker}@MISX",
        quantity=FinamDecimal(value=str(quantity)),
        side=Side.BUY if side == "BUY" else Side.SELL,
        type=order_type,
        time_in_force=TimeInForce.DAY,
    )
    if price is not None:
        order.limit_price = FinamDecimal(value=str(price))
    return await client.orders.place_order(order)


def _to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value
