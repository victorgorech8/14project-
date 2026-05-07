from typing import Any, Dict, List

import numpy as np

from .moex_dynamic import build_dynamic_portfolio


def get_softs_analysis() -> Dict[str, Any]:
    return {
        "paper": "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion",
        "authors": "Lu Han, Xu-Yang Chen, Han-Jia Ye, De-Chuan Zhan",
        "core_idea": "STAR Aggregate-Redistribute: все ряды агрегируются в общий core, затем core перераспределяется обратно в представления отдельных рядов.",
        "why_relevant_for_portfolio": [
            "акции — это многомерный временной ряд с большим числом каналов",
            "канальная независимость устойчива к distribution drift, но теряет межрыночные связи",
            "attention улавливает связи, но дорог и может переобучаться на нестабильных корреляциях",
            "SOFTS/STAR дает компромисс: общий рыночный core + индивидуальная динамика каждой акции",
        ],
        "implementation_in_project": "Добавлен легкий SOFTS-inspired forecast score: общий market core строится как робастное среднее доходностей, затем каждая акция оценивается по смеси собственного momentum и связи с core.",
        "limitations": "Это не полное обучение оригинальной SOFTS-нейросети, а интерпретируемый исследовательский слой, вдохновленный STAR. Для полной модели нужен PyTorch-модуль и backtesting.",
    }


def build_softs_signal(
    lookback_days: int = 180,
    top_pool: int = 30,
    horizon_days: int = 5,
    universe_limit: int = 60,
) -> Dict[str, Any]:
    portfolio = build_dynamic_portfolio(
        lookback_days=lookback_days,
        top_pool=top_pool,
        target_assets=min(8, top_pool),
        universe_limit=universe_limit,
    )
    top_stocks = portfolio["top_stocks"][:top_pool]
    scores = []

    returns = np.array([stock["total_return_pct"] for stock in top_stocks], dtype=np.float64) / 100.0
    volatility = np.array([stock["volatility_annual_pct"] for stock in top_stocks], dtype=np.float64) / 100.0
    drawdown = np.abs(np.array([stock["max_drawdown_pct"] for stock in top_stocks], dtype=np.float64) / 100.0)

    market_core = float(np.median(returns))
    dispersion = float(np.std(returns) + 1e-12)

    for idx, stock in enumerate(top_stocks):
        local_signal = returns[idx] / max(volatility[idx], 1e-6)
        core_alignment = 1.0 - abs(returns[idx] - market_core) / (dispersion + 1e-12)
        risk_penalty = drawdown[idx]
        score = 0.60 * local_signal + 0.30 * core_alignment - 0.10 * risk_penalty
        forecast_return = returns[idx] * horizon_days / max(lookback_days, 1)
        scores.append(
            {
                "ticker": stock["ticker"],
                "softs_score": round(float(score), 6),
                "forecast_return_pct": round(float(forecast_return * 100), 4),
                "core_alignment": round(float(core_alignment), 6),
                "local_return_pct": stock["total_return_pct"],
                "volatility_annual_pct": stock["volatility_annual_pct"],
            }
        )

    scores.sort(key=lambda item: item["softs_score"], reverse=True)
    return {
        "method": "SOFTS-inspired STAR core signal",
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "market_core_return": round(market_core, 6),
        "signals": scores,
        "analysis": get_softs_analysis(),
    }


def tsururu_example_code() -> Dict[str, Any]:
    code = '''
import pandas as pd
from tsururu.dataset import Pipeline, TSDataset
from tsururu.model_training.trainer import MLTrainer
from tsururu.model_training.validator import KFoldCrossValidator
from tsururu.models.boost import CatBoost
from tsururu.strategies import RecursiveStrategy

# df columns: date, id, value. id = ticker, value = close или return.
df = pd.DataFrame({
    "date": ["2025-01-01", "2025-01-02"],
    "id": ["SBER", "SBER"],
    "value": [290.0, 292.0],
})

dataset_params = {
    "target": {"columns": ["value"]},
    "date": {"columns": ["date"]},
    "id": {"columns": ["id"]},
}

dataset = TSDataset(data=df, columns_params=dataset_params)
pipeline = Pipeline.easy_setup(
    dataset_params,
    {"target_lags": 20, "date_lags": 7},
    multivariate=False,
)
trainer = MLTrainer(model=CatBoost, validator=KFoldCrossValidator)
strategy = RecursiveStrategy(horizon=5, history=60, trainer=trainer, pipeline=pipeline)

fit_time, _ = strategy.fit(dataset)
forecast_time, prediction = strategy.predict(dataset)
print(prediction)
'''.strip()
    return {
        "package": "tsururu",
        "install": "pip install -U tsururu[catboost]",
        "purpose": "Альтернативный forecasting pipeline для сравнения стратегий Recursive/Direct/MIMO/FlatWideMIMO на акциях MOEX.",
        "example_code": code,
    }
