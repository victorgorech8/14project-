# Grafana для динамического портфеля MOEX

Grafana может отображать данные из API проекта через JSON API datasource.

## Быстрый запуск через Docker Compose

Из корня проекта:

```bash
docker compose up --build
```

После запуска откройте Grafana: `http://localhost:3000`.

Логин/пароль: `admin` / `admin`.

Datasource и dashboard подключаются автоматически через provisioning.

## Что уже добавлено в FastAPI

- `GET /moex/top-stocks` — динамический пул самых прибыльных акций за выбранное окно.
- `GET /moex/dynamic-portfolio` — динамически пересобранный портфель, веса и ряд доходности.
- `GET /moex/dynamic-portfolio/weights` — веса портфеля в удобном для Grafana формате массива.
- `POST /moex/dynamic-portfolio` — то же самое, но параметры передаются JSON-телом.

Данные берутся напрямую из MOEX ISS API. Внутри FastAPI есть кеш по умолчанию на 30 минут, чтобы не перегружать API МосБиржи и не ждать долгую загрузку при каждом обновлении панели.

## Рекомендуемая частота обновления

Для дневных свечей портфель логично пересобирать:

- для учебного демо: каждые 30-60 минут;
- для реального дневного портфеля: 1 раз в день после закрытия торгов;
- для презентации: можно поставить refresh 5-15 минут, но данные дневные, поэтому смысл чаще ограничен.

Параметр `cache_ttl_seconds` управляет кешем. Например, `1800` — пересчет не чаще одного раза в 30 минут.

## Установка JSON API datasource

В Grafana установите плагин:

```bash
grafana-cli plugins install marcusolsson-json-datasource
```

И перезапустите Grafana.

## Datasource

Создайте datasource типа `JSON API`:

- URL: `http://localhost:8000`
- Access: `Server`

Если FastAPI запущен в Docker-сети рядом с Grafana, вместо `localhost` укажите имя сервиса, например `http://api:8000`.

## Панель 1. Доходность портфеля

Endpoint:

```text
GET /moex/dynamic-portfolio?lookback_days=180&top_pool=30&target_assets=8&method=ensemble&max_weight=0.25&universe_limit=60
```

Поле данных:

```text
performance
```

Поля:

- время: `time`
- значение: `portfolio_value`

Тип панели: `Time series`.

## Панель 2. Самые прибыльные акции

Endpoint:

```text
GET /moex/top-stocks?lookback_days=90&limit=10&universe_limit=60
```

Поле данных:

```text
stocks
```

Поля:

- категория: `ticker`
- значение: `total_return_pct`

Тип панели: `Bar chart` или `Table`.

## Панель 3. Веса портфеля

Endpoint:

```text
GET /moex/dynamic-portfolio/weights?lookback_days=180&top_pool=30&target_assets=8&method=ensemble&max_weight=0.25&universe_limit=60
```

Поле данных: `weights`.

Поля:

- категория: `ticker`
- значение: `weight_pct`

Тип панели: `Pie chart`, `Bar chart` или `Table`.
