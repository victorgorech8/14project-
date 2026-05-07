from fastapi.responses import HTMLResponse


def render_dashboard() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MOEX Portfolio App</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #0f172a; color: #e5e7eb; }
    header { padding: 18px 28px; background: #111827; border-bottom: 1px solid #334155; }
    h1 { margin: 0; font-size: 22px; }
    nav { display: flex; gap: 10px; padding: 14px 28px; background: #0b1220; }
    button { border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; background: #1d4ed8; color: white; }
    button.secondary { background: #334155; }
    main { padding: 22px 28px; }
    .tab { display: none; }
    .tab.active { display: block; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .card { background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 16px; }
    .wide { grid-column: 1 / -1; }
    input, select, textarea { width: 100%; box-sizing: border-box; margin: 6px 0 12px; padding: 9px; border-radius: 8px; border: 1px solid #475569; background: #0f172a; color: #e5e7eb; }
    pre { white-space: pre-wrap; background: #020617; padding: 12px; border-radius: 8px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    th, td { padding: 8px; border-bottom: 1px solid #334155; text-align: left; }
    th { color: #bfdbfe; }
    .bar { height: 14px; border-radius: 999px; background: #2563eb; min-width: 3px; }
    .ok { color: #86efac; }
    .error { color: #fca5a5; }
    .loading { color: #fde68a; }
    .muted { color: #94a3b8; }
    a { color: #93c5fd; }
  </style>
</head>
<body>
  <header>
    <h1>MOEX Portfolio App</h1>
    <div class="muted">Динамический портфель, Grafana, paper/Finam bot, SOFTS-анализ</div>
  </header>
  <nav>
    <button onclick="showTab('portfolio')">Портфель</button>
    <button onclick="showTab('bot')">Бот</button>
    <button onclick="showTab('softs')">SOFTS / Tsururu</button>
    <button onclick="showTab('docs')">Материалы</button>
  </nav>
  <main>
    <section id="portfolio" class="tab active">
      <div class="grid">
        <div class="card wide">
          <button onclick="loadPortfolio()">Обновить портфель</button>
          <span class="muted">данные берутся из MOEX ISS API</span>
          <div id="portfolioStatus" class="loading"></div>
          <div id="portfolioChart"></div>
        </div>
        <div class="card"><h3>Веса</h3><div id="weightsChart"></div></div>
        <div class="card"><h3>Самые прибыльные акции</h3><div id="topChart"></div></div>
        <div class="card wide"><h3>JSON</h3><pre id="portfolioJson"></pre></div>
      </div>
    </section>
    <section id="bot" class="tab">
      <div class="grid">
        <div class="card">
          <h3>План ребалансировки</h3>
          <label>Капитал, руб.</label><input id="cash" value="100000" />
          <label>Текущие позиции JSON</label><textarea id="positions" rows="5">{}</textarea>
          <button onclick="buildPlan()">Сформировать план</button>
        </div>
        <div class="card">
          <h3>Finam status</h3>
          <button onclick="loadFinamStatus()">Проверить</button>
          <pre id="finamStatus"></pre>
        </div>
        <div class="card wide"><h3>Результат</h3><pre id="botResult"></pre></div>
      </div>
    </section>
    <section id="softs" class="tab">
      <div class="grid">
        <div class="card wide">
          <button onclick="loadSofts()">SOFTS-inspired сигнал</button>
          <button class="secondary" onclick="loadTsururu()">Пример Tsururu</button>
          <pre id="softsResult"></pre>
        </div>
      </div>
    </section>
    <section id="docs" class="tab">
      <div class="card">
        <p><a href="/docs" target="_blank">Swagger API</a></p>
        <p><a href="/methodology" target="_blank">PDF методология</a></p>
        <p><a href="http://localhost:3000" target="_blank">Grafana</a></p>
      </div>
    </section>
  </main>
  <script>
    function showTab(id) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.getElementById(id).classList.add('active');
    }
    async function getJson(url, options) {
      const r = await fetch(url, options);
      if (!r.ok) throw new Error(await r.text());
      return await r.json();
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>'"]/g, ch => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;'}[ch]));
    }
    function renderLineChart(elementId, points) {
      if (!points.length) return;
      const width = 900, height = 280, pad = 30;
      const values = points.map(p => p.portfolio_value);
      const min = Math.min(...values), max = Math.max(...values);
      const span = Math.max(max - min, 0.0001);
      const path = points.map((p, i) => {
        const x = pad + i * (width - pad * 2) / Math.max(points.length - 1, 1);
        const y = height - pad - (p.portfolio_value - min) * (height - pad * 2) / span;
        return `${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
      }).join(' ');
      document.getElementById(elementId).innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" style="width:100%;height:320px;background:#020617;border-radius:10px">
          <path d="${path}" fill="none" stroke="#60a5fa" stroke-width="3" />
          <text x="${pad}" y="22" fill="#e5e7eb">Последнее значение: ${values[values.length - 1].toFixed(2)}</text>
          <text x="${pad}" y="${height - 8}" fill="#94a3b8">${escapeHtml(points[0].time.slice(0,10))}</text>
          <text x="${width - 120}" y="${height - 8}" fill="#94a3b8">${escapeHtml(points[points.length - 1].time.slice(0,10))}</text>
        </svg>`;
    }
    function renderWeights(weights) {
      const rows = Object.entries(weights).sort((a, b) => b[1] - a[1]).map(([ticker, weight]) => `
        <tr><td>${escapeHtml(ticker)}</td><td>${(weight * 100).toFixed(2)}%</td><td><div class="bar" style="width:${Math.max(weight * 100, 2)}%"></div></td></tr>
      `).join('');
      document.getElementById('weightsChart').innerHTML = `<table><tr><th>Тикер</th><th>Вес</th><th></th></tr>${rows}</table>`;
    }
    function renderTopStocks(stocks) {
      const maxAbs = Math.max(...stocks.map(s => Math.abs(s.total_return_pct)), 1);
      const rows = stocks.slice(0, 10).map(s => `
        <tr><td>${escapeHtml(s.ticker)}</td><td>${s.total_return_pct.toFixed(2)}%</td><td><div class="bar" style="width:${Math.max(Math.abs(s.total_return_pct) / maxAbs * 100, 2)}%; background:${s.total_return_pct >= 0 ? '#22c55e' : '#ef4444'}"></div></td></tr>
      `).join('');
      document.getElementById('topChart').innerHTML = `<table><tr><th>Тикер</th><th>Доходность</th><th></th></tr>${rows}</table>`;
    }
    async function loadPortfolio() {
      const status = document.getElementById('portfolioStatus');
      status.className = 'loading';
      status.textContent = 'Загружаю данные MOEX и считаю портфель... первый расчет может занять 20-60 секунд.';
      try {
        const data = await getJson('/moex/dynamic-portfolio?lookback_days=90&top_pool=12&target_assets=6&method=ensemble&max_weight=0.30&universe_limit=18');
        document.getElementById('portfolioJson').textContent = JSON.stringify(data, null, 2);
        renderLineChart('portfolioChart', data.performance || []);
        renderWeights(data.weights || {});
        renderTopStocks(data.top_stocks || []);
        status.className = 'ok';
        status.textContent = 'Готово. Портфель пересчитан.';
      } catch (e) {
        status.className = 'error';
        status.textContent = 'Ошибка загрузки: ' + e.message;
        document.getElementById('portfolioJson').textContent = e.stack || e.message;
      }
    }
    async function buildPlan() {
      const out = document.getElementById('botResult');
      out.textContent = 'Считаю план...';
      try {
        const payload = {cash: Number(document.getElementById('cash').value), current_positions: JSON.parse(document.getElementById('positions').value)};
        const data = await getJson('/moex/rebalance-plan', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
        out.textContent = JSON.stringify(data, null, 2);
      } catch (e) {
        out.textContent = 'Ошибка: ' + e.message;
      }
    }
    async function loadFinamStatus() {
      const data = await getJson('/finam/status');
      document.getElementById('finamStatus').textContent = JSON.stringify(data, null, 2);
    }
    async function loadSofts() {
      const data = await getJson('/research/softs-signal');
      document.getElementById('softsResult').textContent = JSON.stringify(data, null, 2);
    }
    async function loadTsururu() {
      const data = await getJson('/research/tsururu-example');
      document.getElementById('softsResult').textContent = JSON.stringify(data, null, 2);
    }
    loadPortfolio();
  </script>
</body>
</html>
"""
    return HTMLResponse(html)
