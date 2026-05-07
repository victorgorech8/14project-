from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import requests

from .portfolio_advanced import AdvancedPortfolioPipeline


MOEX_BASE_URL = "https://iss.moex.com/iss"
DEFAULT_CACHE_TTL_SECONDS = 30 * 60

_cache: Dict[str, Tuple[datetime, Any]] = {}
_http = requests.Session()


def get_top_stocks(
    lookback_days: int = 90,
    limit: int = 10,
    universe_limit: int = 40,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> Dict[str, Any]:
    cache_key = f"top:{lookback_days}:{limit}:{universe_limit}"
    cached = _get_cached(cache_key, cache_ttl_seconds)
    if cached is not None:
        return cached

    candidates = _get_liquid_tickers(universe_limit=universe_limit)
    ranked = []
    histories = _load_histories(candidates, lookback_days=lookback_days)
    for ticker, history in histories.items():
        metrics = _calculate_stock_metrics(ticker, history)
        if metrics is not None:
            ranked.append(metrics)

    ranked.sort(key=lambda item: item["total_return_pct"], reverse=True)
    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "lookback_days": lookback_days,
        "universe_limit": universe_limit,
        "limit": limit,
        "stocks": ranked[:limit],
        "source": "MOEX ISS API",
    }
    _set_cached(cache_key, result)
    return result


def build_dynamic_portfolio(
    lookback_days: int = 180,
    top_pool: int = 30,
    target_assets: int = 8,
    method: str = "ensemble",
    max_weight: float = 0.25,
    universe_limit: int = 60,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> Dict[str, Any]:
    cache_key = f"portfolio:{lookback_days}:{top_pool}:{target_assets}:{method}:{max_weight}:{universe_limit}"
    cached = _get_cached(cache_key, cache_ttl_seconds)
    if cached is not None:
        return cached

    top = get_top_stocks(
        lookback_days=lookback_days,
        limit=top_pool,
        universe_limit=universe_limit,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    pool = [stock["ticker"] for stock in top["stocks"]]
    histories = _load_histories(pool, lookback_days=lookback_days)
    tensor, tickers, dates = _build_tensor(histories)

    if len(tickers) < 2:
        raise ValueError("Недостаточно акций с полной историей для динамического портфеля")

    pipeline = AdvancedPortfolioPipeline(
        method=method.lower(),
        n_clusters=min(8, len(tickers)),
        target_assets=min(target_assets, len(tickers)),
        max_weight=max_weight,
    ).fit(tensor, np.array(tickers))

    weights = pipeline.get_weights()
    performance = _portfolio_performance(histories, dates, weights)
    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "MOEX ISS API",
        "rebalance_policy": "Пересчет при каждом запросе с кешем; для Grafana обычно достаточно обновления раз в 30-60 минут в торговое время.",
        "lookback_days": lookback_days,
        "top_pool": top_pool,
        "target_assets": target_assets,
        "method": method.lower(),
        "portfolio": pipeline.predict(),
        "weights": weights,
        "diagnostics": pipeline.get_diagnostics(),
        "clusters": pipeline.get_clusters(),
        "performance": performance,
        "top_stocks": top["stocks"],
    }
    _set_cached(cache_key, result)
    return result


def get_dynamic_portfolio_weights(**kwargs: Any) -> Dict[str, Any]:
    portfolio = build_dynamic_portfolio(**kwargs)
    rows = [
        {"ticker": ticker, "weight": round(float(weight), 6), "weight_pct": round(float(weight) * 100, 4)}
        for ticker, weight in portfolio["weights"].items()
    ]
    rows.sort(key=lambda item: item["weight"], reverse=True)
    return {
        "generated_at": portfolio["generated_at"],
        "method": portfolio["method"],
        "weights": rows,
        "source": portfolio["source"],
    }


def build_rebalance_plan(
    cash: float,
    current_positions: Dict[str, int] | None = None,
    **portfolio_kwargs: Any,
) -> Dict[str, Any]:
    if cash <= 0:
        raise ValueError("cash должен быть положительным")

    current_positions = current_positions or {}
    portfolio = build_dynamic_portfolio(**portfolio_kwargs)
    last_prices = {stock["ticker"]: stock["last_price"] for stock in portfolio["top_stocks"]}
    plan = []
    invested = 0.0

    for ticker, target_weight in portfolio["weights"].items():
        price = last_prices.get(ticker)
        if not price or price <= 0:
            continue
        target_value = cash * target_weight
        target_lots = int(target_value // price)
        current_lots = int(current_positions.get(ticker, 0))
        delta_lots = target_lots - current_lots
        invested += target_lots * price
        action = "BUY" if delta_lots > 0 else "SELL" if delta_lots < 0 else "HOLD"
        plan.append(
            {
                "ticker": ticker,
                "action": action,
                "current_lots": current_lots,
                "target_lots": target_lots,
                "delta_lots": delta_lots,
                "last_price": round(float(price), 4),
                "target_weight": round(float(target_weight), 6),
                "target_value": round(float(target_value), 2),
            }
        )

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "mode": "paper_trading_rebalance_plan",
        "important_notice": "MOEX ISS API не выставляет заявки. Это безопасный расчетный план; для реальной торговли нужен API брокера.",
        "cash": round(float(cash), 2),
        "estimated_invested": round(float(invested), 2),
        "estimated_cash_left": round(float(cash - invested), 2),
        "portfolio": portfolio["portfolio"],
        "weights": portfolio["weights"],
        "orders": plan,
    }


def _get_liquid_tickers(universe_limit: int) -> List[str]:
    url = f"{MOEX_BASE_URL}/engines/stock/markets/shares/boards/TQBR/securities.json"
    response = _http.get(url, timeout=8)
    response.raise_for_status()
    payload = response.json()

    securities = _rows_to_dicts(payload.get("securities", {}))
    marketdata = _rows_to_dicts(payload.get("marketdata", {}))
    liquidity_by_ticker = {
        row.get("SECID"): _safe_float(row.get("VALTODAY") or row.get("VALUE") or row.get("NUMTRADES"))
        for row in marketdata
    }

    tickers = []
    for row in securities:
        ticker = row.get("SECID")
        if not ticker or not ticker.isalnum():
            continue
        tickers.append({"ticker": ticker, "liquidity": liquidity_by_ticker.get(ticker, 0.0)})

    tickers.sort(key=lambda item: item["liquidity"], reverse=True)
    return [item["ticker"] for item in tickers[:universe_limit]]


def _get_history(ticker: str, lookback_days: int) -> List[Dict[str, Any]]:
    cache_key = f"history:{ticker}:{lookback_days}:{date.today().isoformat()}"
    cached = _get_cached(cache_key, DEFAULT_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    from_date = (date.today() - timedelta(days=max(lookback_days * 2, 30))).isoformat()
    url = f"{MOEX_BASE_URL}/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
    rows = []
    start = 0
    while True:
        response = _http.get(
            url,
            params={"from": from_date, "start": start},
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
        batch = _rows_to_dicts(payload.get("history", {}))
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 100:
            break
        start += 100

    prepared = []
    for row in rows:
        close = _safe_float(row.get("CLOSE"))
        trade_date = row.get("TRADEDATE")
        if close <= 0 or not trade_date:
            continue
        prepared.append(
            {
                "date": trade_date,
                "close": close,
                "volume": _safe_float(row.get("VOLUME")),
                "numtrades": _safe_float(row.get("NUMTRADES")),
            }
        )
    prepared = prepared[-lookback_days:]
    _set_cached(cache_key, prepared)
    return prepared


def _load_histories(tickers: List[str], lookback_days: int) -> Dict[str, List[Dict[str, Any]]]:
    histories: Dict[str, List[Dict[str, Any]]] = {}
    if not tickers:
        return histories

    max_workers = min(8, len(tickers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_get_history, ticker, lookback_days): ticker for ticker in tickers}
        for future in as_completed(futures, timeout=max(20, len(tickers) * 3)):
            ticker = futures[future]
            try:
                histories[ticker] = future.result(timeout=1)
            except Exception:
                histories[ticker] = []
    return histories


def _calculate_stock_metrics(ticker: str, history: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if len(history) < 10:
        return None
    closes = np.array([row["close"] for row in history], dtype=np.float64)
    returns = closes[1:] / closes[:-1] - 1.0
    if len(returns) == 0 or not np.all(np.isfinite(returns)):
        return None

    cumulative = closes / closes[0]
    drawdown = cumulative / np.maximum.accumulate(cumulative) - 1.0
    return {
        "ticker": ticker,
        "start_date": history[0]["date"],
        "end_date": history[-1]["date"],
        "last_price": round(float(closes[-1]), 4),
        "total_return_pct": round(float((closes[-1] / closes[0] - 1.0) * 100), 4),
        "volatility_annual_pct": round(float(np.std(returns) * np.sqrt(252) * 100), 4),
        "max_drawdown_pct": round(float(np.min(drawdown) * 100), 4),
        "avg_volume": round(float(np.mean([row["volume"] for row in history])), 2),
    }


def _build_tensor(histories: Dict[str, List[Dict[str, Any]]]) -> Tuple[np.ndarray, List[str], List[str]]:
    by_ticker = {}
    for ticker, history in histories.items():
        if len(history) < 10:
            continue
        by_ticker[ticker] = {row["date"]: row for row in history}

    if not by_ticker:
        raise ValueError("MOEX API не вернул достаточно исторических данных")

    common_dates = set.intersection(*(set(rows.keys()) for rows in by_ticker.values()))
    dates = sorted(common_dates)
    if len(dates) < 3:
        raise ValueError("Недостаточно общих торговых дат для выбранных акций")

    tickers = sorted(by_ticker.keys())
    tensor = np.zeros((len(dates), len(tickers), 3), dtype=np.float64)
    for ticker_idx, ticker in enumerate(tickers):
        for date_idx, trade_date in enumerate(dates):
            row = by_ticker[ticker][trade_date]
            tensor[date_idx, ticker_idx, 0] = row["close"]
            tensor[date_idx, ticker_idx, 1] = row["volume"]
            tensor[date_idx, ticker_idx, 2] = row["numtrades"]
    return tensor, tickers, dates


def _portfolio_performance(
    histories: Dict[str, List[Dict[str, Any]]],
    dates: List[str],
    weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    selected = sorted(weights.keys())
    rows_by_ticker = {
        ticker: {row["date"]: row for row in histories[ticker]}
        for ticker in selected
        if ticker in histories
    }

    value = 100.0
    result = [{"time": f"{dates[0]}T00:00:00Z", "portfolio_value": round(value, 4)}]
    for current_date, previous_date in zip(dates[1:], dates[:-1]):
        daily_return = 0.0
        for ticker in selected:
            current = rows_by_ticker[ticker][current_date]["close"]
            previous = rows_by_ticker[ticker][previous_date]["close"]
            daily_return += weights[ticker] * (current / previous - 1.0)
        value *= 1.0 + daily_return
        result.append({"time": f"{current_date}T00:00:00Z", "portfolio_value": round(float(value), 4)})
    return result


def _rows_to_dicts(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = section.get("columns", [])
    return [dict(zip(columns, row)) for row in section.get("data", [])]


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        result = float(value)
        return result if np.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _get_cached(key: str, ttl_seconds: int) -> Any | None:
    cached = _cache.get(key)
    if cached is None:
        return None
    created_at, value = cached
    if (datetime.utcnow() - created_at).total_seconds() > ttl_seconds:
        _cache.pop(key, None)
        return None
    return value


def _set_cached(key: str, value: Any) -> None:
    _cache[key] = (datetime.utcnow(), value)
