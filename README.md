# 14project-
Дипломный проект студентов программы ИИ2025
Тема: Биржевая и внебиржевая торговля, оптимизация портфеля

Руководители: Ковалева Александра

Уровень сложности: base/middle
	
Пререквизиты: понимание специфики работы с финансовыми рядами, понимание отличий от классических табличных данных (структура OHLC, хронология). Для base интерес к финансовой тематике и готовность осваивать новое.

Описание:
- оптимизация и диверсификация портфелем ценных бумаг (asset management)
- алгоритмы биржевой торговли

## Продвинутый модуль составления портфелей

В API добавлен модуль `AdvancedPortfolioPipeline`, который расширяет базовую кластеризацию до ансамбля современных ML/DL-подходов к portfolio construction:

- `hrp` — Hierarchical Risk Parity: иерархическая диверсификация по корреляционному дереву с shrinkage-ковариацией.
- `black_litterman` — Black-Litterman с ML-style views: ожидаемая доходность оценивается по EWMA-моментуму и уверенности, зависящей от волатильности.
- `cvar` — оптимизация хвостового риска CVaR 95% через cross-entropy policy search.
- `graph_attention` — графовый attention-подход: акции рассматриваются как сеть корреляций, а системно центральные инструменты штрафуются для снижения концентрации риска.
- `ensemble` — метод по умолчанию; объединяет HRP, Black-Litterman, CVaR и graph attention в один робастный портфель.

Научная новизна проекта: не один классический Markowitz-оптимизатор, а гибридная модель, объединяющая робастную оценку риска, хвостовую оптимизацию, графовую структуру рынка и ансамблирование решений. Это ближе к свежим исследовательским подходам, где deep learning идеи используются не только для прогноза цены, но и для управления структурным и системным риском портфеля.

### API

Список методов:

```http
GET /methods
```

Построение портфеля:

```http
POST /forward
```

Пример тела запроса:

```json
{
  "tensor_3d": [[[100.0], [120.0]], [[101.0], [119.0]], [[102.0], [121.0]]],
  "tickers": ["SBER", "GAZP"],
  "method": "ensemble",
  "target_assets": 8,
  "max_weight": 0.25
}
```

Ответ содержит не только список тикеров, но и веса портфеля, кластеры и диагностические метрики: ожидаемую дневную доходность, волатильность, Sharpe, CVaR 95% и effective number of assets.

### Динамический портфель по данным MOEX

Для автоматического обновления портфеля добавлены endpoint'ы, которые сами получают свежие дневные данные из MOEX ISS API:

```http
GET /moex/top-stocks?lookback_days=90&limit=10&universe_limit=60
```

Возвращает динамический пул самых прибыльных акций за выбранный период.

```http
GET /moex/dynamic-portfolio?lookback_days=180&top_pool=30&target_assets=8&method=ensemble&max_weight=0.25&universe_limit=60
```

Пересобирает портфель из актуального пула акций, возвращает веса, кластеры, диагностику и временной ряд доходности портфеля. По умолчанию включен кеш на 30 минут, чтобы не перегружать API МосБиржи.

Для дневных данных разумная частота ребалансировки — 1 раз в день после закрытия торгов. Для демонстрационного dashboard в Grafana можно обновлять данные каждые 30-60 минут.

### Grafana

В папке `grafana/` добавлены:

- `README.md` — инструкция по подключению Grafana к FastAPI через JSON API datasource.
- `dashboard_dynamic_portfolio.json` — заготовка dashboard с графиком доходности портфеля и панелью самых прибыльных акций.

### Docker-запуск

Проект завернут в Docker Compose: FastAPI + Grafana + volume для SQLite.

```bash
docker compose up --build
```

После запуска:

- цельное веб-приложение: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Grafana: `http://localhost:3000`
- логин/пароль Grafana: `admin` / `admin`

Grafana автоматически получает datasource `MOEX Portfolio API` и dashboard `MOEX Dynamic Portfolio`.

### Структура проекта

- `app/` — FastAPI-приложение, портфельные модели, MOEX/Finam интеграции и веб-интерфейс.
- `analysis/notebooks/` — Jupyter-ноутбуки EDA.
- `analysis/reports/` — HTML-отчеты аналитики, доступны через `/eda1` и `/eda2`.
- `analysis/charts/` — сгенерированные графики аналитики.
- `scripts/` — вспомогательные скрипты загрузки данных и построения графиков.
- `docs/` — презентация, методология, Finam/SOFTS документация.
- `models/` — сохраненные ML-модели.
- `data/` — локальные базы и runtime-данные вне Docker volume.
- `grafana/` — dashboard и provisioning Grafana.
- `moex_data/` — локальный датасет MOEX в parquet.

### Paper-trading утилита

Через MOEX ISS API нельзя выставлять реальные заявки — это API рыночных данных. Для реальной торговли нужен API брокера. Поэтому добавлена безопасная утилита `trading_bot.py`, которая строит только расчетный план ребалансировки:

```bash
python trading_bot.py --cash 100000 --api-url http://localhost:8000
```

Если есть текущие позиции, их можно передать JSON-файлом:

```json
{"SBER": 10, "GAZP": 5}
```

```bash
python trading_bot.py --cash 100000 --positions positions.json
```

Также доступен API endpoint:

```http
POST /moex/rebalance-plan
```

### Finam Trade API

Добавлен адаптер под Finam Trade API. По умолчанию он работает безопасно: реальные заявки не отправляются, пока явно не включить торговлю.

Документация: `docs/FINAM.md`.

Основные endpoint'ы:

```http
GET /finam/status
GET /finam/account
POST /finam/orders
```

Для реальной торговли нужны переменные окружения:

```env
FINAM_TOKEN=...
FINAM_ACCOUNT_ID=...
ENABLE_REAL_TRADING=true
```

И в запросе на отправку заявок нужно указать `confirm_real_trading=true`.

### SOFTS и Tsururu

Добавлена исследовательская линия по статье `SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion` и пример альтернативного forecasting pipeline на `tsururu`.

Endpoint'ы:

```http
GET /research/softs
GET /research/softs-signal
GET /research/tsururu-example
```

Документация: `docs/SOFTS_TSURURU.md`.

Команда проекта:
Бреусова Юлия Евгеньевна
Виноградов Илья Романович
Горюнов Виктор Алексеевич
