# Finam Trade API

## Важное предупреждение

Реальная торговля связана с финансовым риском. По умолчанию приложение работает безопасно: заявки в Finam не отправляются.

Чтобы включить реальную отправку заявок, нужны одновременно два условия:

1. В `.env` указано `ENABLE_REAL_TRADING=true`.
2. В запросе к `/finam/orders` указано `confirm_real_trading=true`.

Если хотя бы одно условие не выполнено, endpoint вернет `dry_run`.

## Настройка

Скопируйте пример окружения:

```bash
cp .env.example .env
```

Заполните:

```env
FINAM_TOKEN=ваш_токен
FINAM_ACCOUNT_ID=ваш_account_id
ENABLE_REAL_TRADING=false
```

Для первого теста оставьте `ENABLE_REAL_TRADING=false`.

## Проверка статуса

```http
GET /finam/status
```

## Информация об аккаунте

```http
GET /finam/account
```

## Dry-run отправки заявок

```http
POST /finam/orders
```

```json
{
  "confirm_real_trading": false,
  "orders": [
    {"ticker": "SBER", "side": "BUY", "quantity": 1, "price": 290.0}
  ]
}
```

## Реальная отправка заявок

Только после проверки на минимальных объемах:

```env
ENABLE_REAL_TRADING=true
```

И тело запроса:

```json
{
  "confirm_real_trading": true,
  "orders": [
    {"ticker": "SBER", "side": "BUY", "quantity": 1, "price": 290.0}
  ]
}
```

Тикеры отправляются в формате Finam `SBER@MISX` внутри адаптера. В API приложения достаточно передавать `SBER`.

