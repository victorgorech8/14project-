# SOFTS и Tsururu в проекте

## SOFTS

Статья: **SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion**.

Авторы: Lu Han, Xu-Yang Chen, Han-Jia Ye, De-Chuan Zhan.

Основная идея статьи — модуль **STAR Aggregate-Redistribute**:

1. Модель берет много временных рядов, например акции.
2. Агрегирует их в общий глобальный `core`.
3. Возвращает этот core обратно в локальные представления отдельных рядов.
4. Получается компромисс между channel independence и учетом межканальных связей.

Почему это полезно для акций:

- акции имеют индивидуальную динамику;
- рынок имеет общий фактор/режим;
- обычный attention может быть тяжелым и нестабильным;
- STAR дешевле и устойчивее при большом числе каналов.

В проект добавлен endpoint:

```http
GET /research/softs
```

И SOFTS-inspired сигнал:

```http
GET /research/softs-signal
```

Это не полная реализация оригинальной нейросети SOFTS, а исследовательский слой, вдохновленный идеей STAR: строится рыночный core и оценивается соответствие каждой акции этому core с учетом локального momentum и риска.

## Tsururu

Tsururu — Python-библиотека для стратегий прогнозирования временных рядов. Она полезна как альтернативный forecasting baseline: можно сравнивать Recursive, Direct, MIMO и FlatWideMIMO стратегии.

Endpoint с примером:

```http
GET /research/tsururu-example
```

Установка:

```bash
pip install -U tsururu[catboost]
```

Для диплома это можно оформить как дополнительную исследовательскую линию: сравнить текущий ансамбль портфеля с прогнозами Tsururu и SOFTS-inspired сигналом.

