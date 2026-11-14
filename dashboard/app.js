/* MoE Prefetch Lab: renders every chart from window.DATA (data.js) with d3 v7. No build step. */
(function () {
  'use strict';
  if (!window.DATA) { document.body.insertAdjacentHTML('afterbegin', '<p style="padding:16px">data.js did not load.</p>'); return; }
  if (!window.d3) { document.body.insertAdjacentHTML('afterbegin', '<p style="padding:16px">d3 did not load (needs internet access for cdnjs.cloudflare.com).</p>'); return; }

  // ---- data ------------------------------------------------------------------------------------
  const D = {};
  Object.entries(window.DATA).forEach(([k, v]) => { D[k] = v.rows.map((r) => { const o = {}; v.cols.forEach((c, i) => { o[c] = r[i]; }); return o; }); });

  // ---- state -----------------------------------------------------------------------------------
  const DEF = { model: 'Mixtral-8x7B', batch: '4', link: 'C2C-900', cache: '25%', ctxcache: 'physical', metric: 'overlap', h100: 'calibrated', more: 'ctx' };
  const S = Object.assign({}, DEF);
  try { const h = new URLSearchParams(location.search); Object.keys(DEF).forEach((k) => { if (h.get(k)) S[k] = h.get(k); }); } catch (e) { /* ignore */ }
  function set(patch) { Object.assign(S, patch); renderAll(); }

  // ---- helpers ---------------------------------------------------------------------------------
  const C = ['var(--c0)', 'var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c4)'];
  const num = (v) => (v === null || v === undefined || v === '') ? NaN : +v;
  const fx = (v) => isFinite(v) ? (v >= 10 ? v.toFixed(1) : v.toFixed(2)) + '×' : '–';
  const fpct = (v) => isFinite(v) ? (v * 100).toFixed(v < 0.1 ? 1 : 0) + ' %' : '–';
  const fms = (v) => isFinite(v) ? (v >= 100 ? d3.format(',.0f')(v) : v.toFixed(1)) + ' ms' : '–';
  const fJ = (v) => isFinite(v) ? (v >= 10 ? v.toFixed(1) : v.toFixed(2)) + ' J' : '–';
  const $ = (id) => document.getElementById(id);
  function mk(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined && text !== null) e.textContent = text; return e; }
  function empty(el, msg) { el.replaceChildren(mk('p', 'empty', msg)); }
  function makeTip(wrap) {
    const t = mk('div', 'tip'); t.hidden = true; wrap.append(t);
    return {
      show(head, rows, x, y) {
        t.replaceChildren(mk('div', 'tip-h', head), ...rows.map(([a, b]) => { const r = mk('div', 'tip-row'); r.append(mk('span', null, a), mk('span', null, b)); return r; }));
        t.hidden = false;
        const W = wrap.clientWidth, tw = t.offsetWidth, th = t.offsetHeight;
        let left = x + 14; if (left + tw > W) left = Math.max(0, x - 14 - tw);
        t.style.left = left + 'px'; t.style.top = Math.max(0, y - th / 2) + 'px';
      },
      hide() { t.hidden = true; }
    };
  }
  const rel = (e, wrap) => { const r = wrap.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  function hover(row, wrap, tip, head, rowsFn) {
    const on = (e) => { const [x, y] = rel(e, wrap); tip.show(head, rowsFn(), x, y); };
    row.addEventListener('pointermove', on);
    row.addEventListener('pointerdown', on);
    row.addEventListener('pointerleave', () => tip.hide());
  }

  const FULL = '9 + value gate = full design', OD = '1 on-demand, no cache', LRUO = 'LRU + miss overlap (no prediction)', ORA = '10 oracle predictor';
  const LADDER = ['1 on-demand, no cache', '2 + HBM cache (LRU)', '3 + temporal predictor', '4 + cross-layer + popularity', '5 + chooser', '6 + adaptive budget', '7 + lookahead + preemption', '8 + need eviction + miss overlap', FULL, ORA, '11 oracle + Belady'];
  const REFS = [LRUO, 'LFU (no prediction)', 'LFU + miss overlap (no prediction)', 'prediction for eviction only (no prefetch)', 'full design, no retention', 'oracle, no retention'];
  const cfgRows = () => D.ladder.filter((r) => r.model === S.model && r.batch === S.batch && r.link === S.link && r.cache === S.cache);
  const byRung = (rows, rung) => rows.find((r) => r.rung === rung);
  const LINKNAME = { 'PCIe 4.0': 'PCIe 4', 'PCIe 5.0': 'PCIe 5', 'C2C-450': 'C2C 450', 'C2C-900': 'C2C 900' };
  const cfgText = () => `${S.model} · batch ${S.batch} · ${LINKNAME[S.link] || S.link} · ${S.cache} cache`;

  // ---- controls --------------------------------------------------------------------------------
  document.querySelectorAll('select[data-param]').forEach((s) => s.addEventListener('change', () => set({ [s.dataset.param]: s.value })));
  document.querySelectorAll('[data-set]').forEach((b) => b.addEventListener('click', () => { const i = b.dataset.set.indexOf(':'); set({ [b.dataset.set.slice(0, i)]: b.dataset.set.slice(i + 1) }); }));
  $('reset-btn').addEventListener('click', () => set(Object.assign({}, DEF)));
  function syncControls() {
    document.querySelectorAll('select[data-param]').forEach((s) => { s.value = S[s.dataset.param]; });
    document.querySelectorAll('[data-set]').forEach((b) => { const i = b.dataset.set.indexOf(':'); b.setAttribute('aria-pressed', String(S[b.dataset.set.slice(0, i)] === b.dataset.set.slice(i + 1))); });
    document.querySelectorAll('[data-more]').forEach((p) => { p.hidden = p.dataset.more !== S.more; });
    try { const q = new URLSearchParams(); Object.keys(DEF).forEach((k) => { if (S[k] !== DEF[k]) q.set(k, S[k]); }); const s = q.toString(); history.replaceState(null, '', location.pathname + (s ? '?' + s : '') + location.hash); } catch (e) { /* file:// */ }
  }

  // theme toggle: auto -> light -> dark
  const themes = ['auto', 'light', 'dark'];
  let theme = 'auto';
  try { theme = localStorage.getItem('moe-theme') || 'auto'; } catch (e) { /* ignore */ }
  function applyTheme() {
    if (theme === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', theme);
    $('theme-btn').textContent = 'Theme: ' + theme;
  }
  $('theme-btn').addEventListener('click', () => { theme = themes[(themes.indexOf(theme) + 1) % 3]; try { localStorage.setItem('moe-theme', theme); } catch (e) { /* ignore */ } applyTheme(); });
  applyTheme();

  // ---- architecture stepper --------------------------------------------------------------------
  let cur = 0, timer = null;
  function step(n) {
    cur = n;
    document.querySelectorAll('.arch').forEach((s) => s.classList.toggle('stepping', n > 0));
    document.querySelectorAll('.arch [data-steps]').forEach((e) => e.classList.toggle('on', n > 0 && e.getAttribute('data-steps').split(' ').includes(String(n))));
    const cap = $('stepcap'), li = document.querySelector(`[data-cap="${n}"]`);
    if (cap) { if (n > 0 && li) { const b = mk('b', null, String(n)); cap.replaceChildren(b, document.createTextNode(li.textContent)); } else cap.textContent = 'Blue = predictions · grey = expert weights'; }
  }
  function stopPlay() { if (timer) { clearInterval(timer); timer = null; } document.querySelectorAll('[data-act="play"]').forEach((b) => b.setAttribute('aria-pressed', 'false')); }
  const nextN = () => (cur >= 7 ? 1 : cur + 1);
  document.querySelectorAll('[data-act]').forEach((b) => b.addEventListener('click', () => {
    const a = b.dataset.act;
    if (a === 'play') {
      if (timer) { stopPlay(); return; }
      b.setAttribute('aria-pressed', 'true'); step(nextN()); timer = setInterval(() => step(nextN()), 2200); return;
    }
    stopPlay();
    if (a === 'next') step(nextN()); else if (a === 'prev') step(cur <= 1 ? 7 : cur - 1); else step(0);
  }));

  // ---- takeaway and tiles ----------------------------------------------------------------------
  function renderTiles() {
    const rows = cfgRows(), f = byRung(rows, FULL), lo = byRung(rows, LRUO), or = byRung(rows, ORA);
    const v = (r, k, fmt) => (r ? fmt(num(r[k])) : '–');
    const put = (id, text, cls) => { const e = $(id); if (e) { e.textContent = text; if (cls !== undefined) e.className = cls; } };
    const g = f ? num(f.vs_lru_overlap) : NaN, gc = g >= 1.005 ? 'good' : (g < 0.995 ? 'bad' : '');
    const accRow = (pred, traces) => D.accuracy.find((z) => z.model === S.model && z.batch === S.batch && z.predictor === pred && z.traces === traces && z.distance === 1);
    const ours = accRow('ours (full, adaptive budget)', 'public'), simple = accRow('previous token', 'public'), hs = accRow('hidden-state (H100)', 'h100');
    put('tk-acc', v(ours, 'overlap', fpct));
    put('t-acc', v(ours, 'overlap', fpct));
    put('t-acc-s', hs ? 'Hidden state on H100: ' + v(hs, 'overlap', fpct) : 'Repeat last token: ' + v(simple, 'overlap', fpct));
    const od = byRung(rows, OD);
    const ideal = od ? num(od.token_latency_ms) * (1 - num(od.stall_fraction)) : NaN;   // no copying at all: compute only
    put('t-ideal', f && isFinite(ideal) ? fpct(ideal / num(f.token_latency_ms)) : '–');
    put('t-ideal-s', or && isFinite(ideal) ? 'Perfect prediction: ' + fpct(ideal / num(or.token_latency_ms)) + ' · ideal = all experts in GPU' : '');
    put('ladder-ideal', od && isFinite(ideal) ? `Ideal, if every expert fitted in GPU memory: ${fx(num(od.token_latency_ms) / ideal)} (${fms(ideal)} per token). Not possible on one GPU, shown as the ceiling.` : '');
    put('tk-config', cfgText());
    put('tk-speed', v(f, 'speedup', fx));
    put('tk-lru', v(f, 'vs_lru_overlap', fx), gc);
    put('t-speed', v(f, 'speedup', fx));
    put('t-speed-s', 'With perfect prediction: ' + v(or, 'speedup', fx));
    put('t-lru', v(f, 'vs_lru_overlap', fx), 'tile-v ' + gc);
    put('t-lru-s', g >= 1.005 ? 'faster than LRU' : (g < 0.995 ? 'slower than LRU' : 'same as LRU'));
    put('t-hit', v(f, 'hit_rate', fpct));
    put('t-hit-s', lo && num(lo.hit_rate) < 0.005 ? 'LRU: 0 % (it evicts what comes next)' : 'LRU: ' + v(lo, 'hit_rate', fpct));
    put('t-energy', v(f, 'energy_J_per_token', fJ));
    put('t-energy-s', 'LRU: ' + v(lo, 'energy_J_per_token', fJ));
  }

  // ---- ladder ----------------------------------------------------------------------------------
  const SHORT = {
    '1 on-demand, no cache': 'On demand (no cache)', '2 + HBM cache (LRU)': '+ GPU cache (LRU)', '3 + temporal predictor': '+ predictor: recent tokens',
    '4 + cross-layer + popularity': '+ layer links, popularity', '5 + chooser': '+ pick best predictor', '6 + adaptive budget': '+ adaptive fetch size',
    '7 + lookahead + preemption': '+ look 4 layers ahead', '8 + need eviction + miss overlap': '+ smart eviction',
    '9 + value gate = full design': '+ cost check = our design', '10 oracle predictor': 'Perfect prediction'
  };
  function renderLadder() {
    const el = $('ladder-chart'), rows = cfgRows();
    const items = LADDER.slice(0, 10).map((n) => byRung(rows, n)).filter(Boolean);
    el.replaceChildren();
    if (!items.length) return empty(el, 'No runs for this setup.');
    const max = d3.max(items, (r) => num(r.speedup)) * 1.04, lro = byRung(rows, LRUO), tip = makeTip(el);
    items.forEach((r) => {
      const n = LADDER.indexOf(r.rung) + 1;
      const row = mk('div', 'bar' + (r.rung === FULL ? ' hl' : '') + (n === 10 ? ' sub sep' : ''));
      const t = mk('div', 'bar-t'), f = mk('div', 'bar-f');
      f.style.width = (100 * num(r.speedup) / max) + '%';
      f.style.background = n <= 2 ? 'var(--chart-muted)' : (n === 10 ? 'var(--c2)' : 'var(--c0)');
      t.append(f);
      if (lro) { const ref = mk('div', 'bar-ref'); ref.style.left = (100 * num(lro.speedup) / max) + '%'; t.append(ref); }
      row.append(mk('div', 'bar-l', SHORT[r.rung] || r.rung), t, mk('div', 'bar-v', fx(num(r.speedup))));
      hover(row, el, tip, SHORT[r.rung] || r.rung, () => [
        ['Token latency', fms(num(r.token_latency_ms))], ['Cache hit rate', fpct(num(r.hit_rate))], ['Time waiting on link', fpct(num(r.stall_fraction))],
        ['Data copied per token', num(r.GB_per_token).toFixed(2) + ' GB'], ['Wasted prefetches', num(r.wasted_GB_per_token).toFixed(2) + ' GB']]);
      el.append(row);
    });
  }

  // ---- heatmap ---------------------------------------------------------------------------------
  const MODELS = ['Mixtral-8x7B', 'Grok-1', 'Llama-4 Maverick', 'DeepSeek-R1'], BATCHES = ['1', '4', '16'], LINKS = ['PCIe 4.0', 'PCIe 5.0', 'C2C-450', 'C2C-900'];
  function renderHeat() {
    const el = $('heat'), data = D.ladder.filter((r) => r.rung === FULL && r.cache === S.cache);
    const get = (m, b, l) => data.find((r) => r.model === m && r.batch === b && r.link === l);
    const span = Math.max(0.05, d3.max(data, (r) => Math.abs(num(r.vs_lru_overlap) - 1)) || 0.05);
    const tbl = mk('table', 'heat'), thead = mk('thead'), h1 = mk('tr'), h2 = mk('tr');
    h1.append(mk('th', null, ''));
    BATCHES.forEach((b) => { const th = mk('th', null, 'Batch ' + b); th.colSpan = LINKS.length; h1.append(th); });
    h2.append(mk('th', 'row', 'Model'));
    BATCHES.forEach(() => LINKS.forEach((l) => h2.append(mk('th', null, l.replace('PCIe ', 'PCIe').replace('C2C-', 'C2C ')))));
    thead.append(h1, h2); tbl.append(thead);
    const tb = mk('tbody');
    MODELS.forEach((m) => {
      const tr = mk('tr'); tr.append(mk('th', 'row', m));
      BATCHES.forEach((b) => LINKS.forEach((l) => {
        const td = mk('td'), r = get(m, b, l);
        if (!r) { td.append(mk('span', 'muted', '–')); tr.append(td); return; }
        const v = num(r.vs_lru_overlap), bt = mk('button');
        bt.type = 'button'; bt.title = `${m}, batch ${b}, ${l}`;
        bt.setAttribute('aria-label', `${m}, batch ${b}, ${l}: ${v.toFixed(2)}`);
        if (m === S.model && b === S.batch && l === S.link) bt.classList.add('sel');
        const bg = mk('span', 'hb'); bg.style.background = v >= 1 ? 'var(--ok)' : 'var(--bad)'; bg.style.opacity = String(0.1 + 0.55 * Math.min(1, Math.abs(v - 1) / span));
        bt.append(bg, mk('span', 'hn', v.toFixed(2)));
        bt.addEventListener('click', () => set({ model: m, batch: b, link: l }));
        td.append(bt); tr.append(td);
      }));
      tb.append(tr);
    });
    tbl.append(tb);
    el.replaceChildren(tbl);
  }

  // ---- line chart ------------------------------------------------------------------------------
  function lineChart(el, series, o) {
    el.replaceChildren();
    const leg = mk('div', 'legend');
    series.forEach((s) => { const it = mk('span', 'leg'), sw = mk('span', 'sw' + (s.dashed ? ' dashed' : '')); sw.style.borderTopColor = s.color; it.append(sw, mk('span', null, s.name)); leg.append(it); });
    el.append(leg);
    const box = mk('div'); box.style.position = 'relative'; el.append(box);
    const W = Math.max(280, el.clientWidth || 600), H = o.height || 280;
    const svg = d3.select(box).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`).attr('role', 'img').attr('aria-label', o.aria);
    const all = series.flatMap((s) => s.points.map((q) => q.y)).filter(isFinite);
    const y = d3.scaleLinear().domain(o.yDomain || [0, (d3.max(all) || 1) * 1.06]).nice();
    const yt = y.ticks(5);
    const probe = svg.append('g');
    let mw = 0; yt.forEach((t) => { const n = probe.append('text').text(o.yFmt(t)).node(); mw = Math.max(mw, n.getComputedTextLength() || String(o.yFmt(t)).length * 6.5); });
    probe.remove();
    const m = { l: mw + 12, r: 16, t: 8, b: 40 };
    const x = (o.log ? d3.scaleLog().base(2) : d3.scaleLinear()).domain(o.xDomain).range([m.l, W - m.r]);
    y.range([H - m.b, m.t]);
    svg.append('g').attr('class', 'axis grid').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(y).tickValues(yt).tickFormat(o.yFmt).tickSize(-(W - m.l - m.r)).tickPadding(6)).call((g) => g.select('.domain').remove());
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues(o.xTicks).tickFormat(o.xFmt).tickSizeOuter(0));
    svg.append('text').attr('x', (m.l + W - m.r) / 2).attr('y', H - 6).attr('text-anchor', 'middle').text(o.xLabel);
    const line = d3.line().defined((q) => isFinite(q.y)).x((q) => x(q.x)).y((q) => y(q.y));
    series.forEach((s) => {
      svg.append('path').attr('d', line(s.points)).attr('fill', 'none').style('stroke', s.color).attr('stroke-width', s.width || 2).attr('stroke-dasharray', s.dashed ? '6 4' : null);
      svg.append('g').selectAll('circle').data(s.points.filter((q) => isFinite(q.y))).join('circle').attr('cx', (q) => x(q.x)).attr('cy', (q) => y(q.y)).attr('r', 3.2).style('fill', s.color);
    });
    const tip = makeTip(box);
    const rule = svg.append('line').attr('y1', m.t).attr('y2', H - m.b).style('stroke', 'var(--ref)').attr('stroke-dasharray', '3 3').attr('visibility', 'hidden');
    const xs = Array.from(new Set(series.flatMap((s) => s.points.map((q) => q.x)))).sort((a, b) => a - b);
    const move = (e) => {
      const vx = d3.pointer(e, svg.node())[0];
      const near = xs.reduce((a, b) => (Math.abs(x(b) - vx) < Math.abs(x(a) - vx) ? b : a), xs[0]);
      rule.attr('x1', x(near)).attr('x2', x(near)).attr('visibility', 'visible');
      const rows = series.map((s) => { const q = s.points.find((z) => z.x === near); return [s.name, q && isFinite(q.y) ? o.tipFmt(q.y) : '–', q && isFinite(q.y) ? q.y : -Infinity]; })
        .sort((a, b) => b[2] - a[2]).map(([a, b]) => [a, b]);
      const [cx, cy] = rel(e, box);
      tip.show(o.xFmtLong(near), rows, cx, cy);
    };
    svg.append('rect').attr('x', m.l).attr('y', m.t).attr('width', W - m.l - m.r).attr('height', H - m.t - m.b).attr('fill', 'transparent')
      .on('pointermove', move).on('pointerdown', move).on('pointerleave', () => { rule.attr('visibility', 'hidden'); tip.hide(); });
  }

  // ---- prediction accuracy ---------------------------------------------------------------------
  function renderAcc() {
    const el = $('acc-chart');
    const pts = (pred, traces) => [1, 2, 3, 4].map((d) => { const r = D.accuracy.find((z) => z.model === S.model && z.batch === S.batch && z.predictor === pred && z.traces === traces && z.distance === d); return { x: d, y: r ? num(r[S.metric]) : NaN }; });
    const series = [
      { name: 'Simple: repeat last token', color: 'var(--chart-muted)', width: 2, points: pts('previous token', 'public') },
      { name: 'Ours (expert IDs only)', color: 'var(--c0)', width: 3, points: pts('ours (full, adaptive budget)', 'public') }
    ];
    if (S.model === 'Mixtral-8x7B') series.push({ name: 'Hidden state (real H100)', color: 'var(--ok)', width: 3, dashed: true, points: pts(S.metric === 'overlap' ? 'hidden-state (H100)' : 'hidden-state + our budget (H100)', 'h100') });
    const label = { overlap: 'Overlap', recall: 'Recall', precision: 'Precision' }[S.metric] || 'Overlap';
    const shown = series.filter((s) => s.points.some((q) => isFinite(q.y)));
    if (!shown.length) return empty(el, 'No data for this model and batch.');
    lineChart(el, shown, {
      aria: `${label} by layers ahead for ${S.model}, batch ${S.batch}`, xDomain: [1, 4], xTicks: [1, 2, 3, 4], xFmt: (d) => d + (d === 1 ? ' layer' : ' layers'),
      xFmtLong: (d) => `${label}, ${d} layer${d > 1 ? 's' : ''} ahead`, xLabel: 'How far ahead', yDomain: [0, 1],
      yFmt: (v) => Math.round(v * 100) + '%', tipFmt: (v) => (v * 100).toFixed(0) + ' %', height: 250
    });
  }

  // ---- context length --------------------------------------------------------------------------
  const CTXPOL = [['on-demand', 'On demand'], ['LRU + overlap', 'LRU'], ['full design', 'Our design'], ['oracle', 'Perfect prediction']];
  const altLink = (l) => ((l === 'PCIe 4.0' || l === 'PCIe 5.0') ? 'PCIe 4.0' : 'C2C-900');
  function renderCtx() {
    const el = $('ctx-chart'), l = altLink(S.link);
    $('ctx-note').textContent = l === S.link ? '' : `(Shown for ${LINKNAME[l]}: this test used PCIe 4 and C2C 900 only.)`;
    const data = D.context.filter((r) => r.model === S.model && r.batch === S.batch && r.link === l && (r.cache === S.ctxcache || r.policy === 'on-demand'));
    const col = ['var(--chart-muted)', 'var(--c1)', 'var(--c0)', 'var(--c2)'];
    const series = CTXPOL.map(([pol, name], i) => ({ name, color: col[i], width: pol === 'full design' ? 3 : 2, dashed: pol === 'oracle',
      points: [256, 512, 1024, 2048, 4096].map((c) => { const r = data.find((z) => z.policy === pol && z.context === c); return { x: c, y: r ? num(r.token_latency_ms) : NaN }; }) }))
      .filter((s) => s.points.some((q) => isFinite(q.y)));
    if (!series.length) return empty(el, 'No runs for this configuration.');
    lineChart(el, series, { aria: `Token latency by context length for ${S.model}`, log: true, xDomain: [256, 4096], xTicks: [256, 512, 1024, 2048, 4096], xFmt: (d) => d,
      xFmtLong: (d) => `Context ${d} tokens`, xLabel: 'Prompt length (tokens)', yFmt: (v) => d3.format(',')(v), tipFmt: fms, height: 250 });
  }

  // ---- energy ----------------------------------------------------------------------------------
  const EN = [[OD, 'On demand'], [LRUO, 'LRU'], [FULL, 'Our design'], [ORA, 'Perfect prediction']];
  function renderEnergy() {
    const el = $('energy-chart'), rows = cfgRows(), items = EN.map(([k, n]) => [byRung(rows, k), n]).filter(([r]) => r);
    el.replaceChildren();
    if (!items.length) return empty(el, 'No runs for this configuration.');
    const max = d3.max(items, ([r]) => num(r.energy_J_per_token)) * 1.02, tip = makeTip(el);
    items.forEach(([r, n]) => {
      const row = mk('div', 'bar' + (r.rung === FULL ? ' hl' : '')), t = mk('div', 'bar-t');
      [['transfer_J_per_token', 'var(--c0)'], ['dynamic_J_per_token', 'var(--c1)'], ['background_J_per_token', 'var(--chart-muted)']].forEach(([f, c], i) => {
        const s = mk('div', 'bar-f'); s.style.width = (100 * num(r[f]) / max) + '%'; s.style.background = c; s.style.borderRadius = i === 0 ? '3px 0 0 3px' : (i === 2 ? '0 3px 3px 0' : '0'); t.append(s);
      });
      row.append(mk('div', 'bar-l', n), t, mk('div', 'bar-v', fJ(num(r.energy_J_per_token))));
      hover(row, el, tip, n, () => [['Total', fJ(num(r.energy_J_per_token))], ['Copying', fJ(num(r.transfer_J_per_token))], ['Computing', fJ(num(r.dynamic_J_per_token))], ['GPU waiting', fJ(num(r.background_J_per_token))], ['Traffic per token', num(r.GB_per_token).toFixed(2) + ' GB'], ['Token latency', fms(num(r.token_latency_ms))]]);
      el.append(row);
    });
  }

  // ---- bonus -----------------------------------------------------------------------------------
  const BPOL = [['LRU + overlap', 'LRU'], ['full design', 'Our design'], ['oracle', 'Perfect prediction']];
  function renderBonus() {
    const el = $('bonus-chart'), note = $('bonus-note'), l = altLink(S.link);
    el.replaceChildren();
    if (S.model === 'Grok-1') { note.textContent = ''; return empty(el, 'Grok-1 has no public trace. Pick another model.'); }
    note.textContent = l === S.link ? '' : `(Shown for ${LINKNAME[l]}.)`;
    const data = D.bonus.filter((r) => r.model === S.model && r.batch === S.batch && r.link === l);
    if (!data.length) return empty(el, 'No runs for this configuration.');
    const max = d3.max(data, (r) => num(r.speedup)) * 1.04, tip = makeTip(el);
    BPOL.forEach(([pol, pname]) => {
      const tr = data.find((r) => r.policy === pol && r.routing === 'trace'), un = data.find((r) => r.policy === pol && r.routing === 'uniform');
      if (!tr && !un) return;
      const row = mk('div', 'bar' + (pol === 'full design' ? ' hl' : '')), t = mk('div', 'bar-t'), v = mk('div', 'bar-v');
      [[tr, 'var(--c0)'], [un, 'var(--chart-muted)']].forEach(([r, c]) => { const f = mk('div', 'bar-f'); f.style.width = r ? (100 * num(r.speedup) / max) + '%' : '0'; f.style.background = c; t.append(f); v.append(mk('span', null, r ? fx(num(r.speedup)) : '–')); });
      row.append(mk('div', 'bar-l', pname), t, v);
      hover(row, el, tip, pname, () => [['Real routing', tr ? fx(num(tr.speedup)) : '–'], ['Uniform routing', un ? fx(num(un.speedup)) : '–'], ['Traffic, real', tr ? num(tr.GB_per_token).toFixed(2) + ' GB' : '–'], ['Traffic, uniform', un ? num(un.GB_per_token).toFixed(2) + ' GB' : '–'], ['Hit rate, real', tr ? fpct(num(tr.hit_rate)) : '–'], ['Hit rate, uniform', un ? fpct(num(un.hit_rate)) : '–']]);
      el.append(row);
    });
  }

  // ---- H100 ------------------------------------------------------------------------------------
  function renderH100() {
    const el = $('h100-chart'), f = S.h100 === 'default' ? 'sim_default_token_latency_ms' : 'sim_cal_token_latency_ms';
    const runs = D.h100_validation.filter((r) => num(r.real_token_latency_ms) > 0 && num(r[f]) > 0);
    const med = d3.median(runs, (r) => Math.abs(num(r[f]) - num(r.real_token_latency_ms)) / num(r.real_token_latency_ms));
    $('h100-err').textContent = isFinite(med) ? 'median error ' + (med * 100).toFixed(1) + ' %' : '–';
    el.replaceChildren();
    if (!runs.length) return empty(el, 'No H100 runs.');
    const bc = { '1': 'var(--c0)', '4': 'var(--c1)', '16': 'var(--c2)' };
    const leg = mk('div', 'legend');
    Object.entries(bc).forEach(([b, c]) => { const it = mk('span', 'leg'), sw = mk('span', 'sw dot'); sw.style.background = c; it.append(sw, mk('span', null, 'Batch ' + b)); leg.append(it); });
    const it = mk('span', 'leg'), sw = mk('span', 'sw dashed'); sw.style.borderTopColor = 'var(--ref)'; it.append(sw, mk('span', null, 'Simulated = real')); leg.append(it);
    el.append(leg);
    const box = mk('div'); box.style.position = 'relative'; el.append(box);
    const W = Math.max(280, el.clientWidth || 500), H = 290;
    const ext = d3.extent(runs.flatMap((r) => [num(r.real_token_latency_ms), num(r[f])]));
    const dom = [ext[0] * 0.85, ext[1] * 1.1];
    const ticks = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000].filter((t) => t >= dom[0] && t <= dom[1]);
    const svg = d3.select(box).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`).attr('role', 'img').attr('aria-label', 'Simulated against real token latency for every H100 run');
    const probe = svg.append('g'); let mw = 0; ticks.forEach((t) => { const n = probe.append('text').text(d3.format(',')(t)).node(); mw = Math.max(mw, n.getComputedTextLength() || 30); }); probe.remove();
    const m = { l: mw + 12, r: 14, t: 22, b: 40 };
    const x = d3.scaleLog().domain(dom).range([m.l, W - m.r]), y = d3.scaleLog().domain(dom).range([H - m.b, m.t]);
    svg.append('g').attr('class', 'axis grid').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).tickValues(ticks).tickFormat(d3.format(',')).tickSize(-(W - m.l - m.r)).tickPadding(6)).call((g) => g.select('.domain').remove());
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues(ticks).tickFormat(d3.format(',')).tickSizeOuter(0));
    svg.append('text').attr('x', (m.l + W - m.r) / 2).attr('y', H - 6).attr('text-anchor', 'middle').text('Real token latency (ms), log scale');
    svg.append('text').attr('x', m.l).attr('y', 10).text('Simulated (ms)');
    svg.append('line').attr('x1', x(dom[0])).attr('y1', y(dom[0])).attr('x2', x(dom[1])).attr('y2', y(dom[1])).style('stroke', 'var(--ref)').attr('stroke-dasharray', '5 4');
    const tip = makeTip(box);
    const show = (e, r) => { const [cx, cy] = rel(e, box), real = num(r.real_token_latency_ms), sim = num(r[f]); tip.show(`${r.spec_item} · batch ${r.batch} · ${Math.round(num(r.cache_ratio) * 100)} % cache`, [['Real', fms(real)], ['Simulated', fms(sim)], ['Error', ((sim - real) / real * 100).toFixed(1) + ' %'], ['Batch', String(r.batch)]], cx, cy); };
    svg.append('g').selectAll('circle').data(runs).join('circle').attr('cx', (r) => x(num(r.real_token_latency_ms))).attr('cy', (r) => y(num(r[f]))).attr('r', 4.8)
      .style('fill', (r) => bc[r.batch] || 'var(--c0)').style('fill-opacity', 0.78).style('stroke', 'var(--panel)').attr('stroke-width', 1)
      .on('pointermove', show).on('pointerdown', show).on('pointerleave', () => tip.hide());
  }
  function renderH100Acc() {
    const el = $('h100acc-chart'), data = D.accuracy.filter((r) => r.traces === 'h100' && r.distance === 1);
    el.replaceChildren();
    const tip = makeTip(el);
    BATCHES.forEach((b) => {
      const id = data.find((r) => r.batch === b && r.predictor === 'ours, ID-only (H100 tokens)'), hs = data.find((r) => r.batch === b && r.predictor === 'hidden-state (H100)');
      const row = mk('div', 'bar'), t = mk('div', 'bar-t'), v = mk('div', 'bar-v');
      [[id, 'var(--chart-muted)'], [hs, 'var(--c0)']].forEach(([r, c]) => { const f = mk('div', 'bar-f'); f.style.width = r ? (100 * num(r.overlap)) + '%' : '0'; f.style.background = c; t.append(f); v.append(mk('span', null, r ? fpct(num(r.overlap)) : '–')); });
      row.append(mk('div', 'bar-l', 'Batch ' + b), t, v);
      hover(row, el, tip, 'Batch ' + b + ', next layer', () => [['ID-only overlap', id ? fpct(num(id.overlap)) : '–'], ['Hidden-state overlap', hs ? fpct(num(hs.overlap)) : '–'], ['Hidden-state recall', hs ? fpct(num(hs.recall)) : '–'], ['Hidden-state precision', hs ? fpct(num(hs.precision)) : '–']]);
      el.append(row);
    });
  }
  const SPOL = [['LRU + overlap', 'LRU'], ['full design [ID-only (ours)]', 'Ours, expert IDs'], ['full design [hidden-state (H100)]', 'Ours, hidden state'], ['oracle', 'Perfect prediction']];
  function renderH100Speed() {
    $('h100speed').replaceChildren(...SPOL.map(([k, n]) => {
      const tr = mk('tr'); if (k === 'full design [hidden-state (H100)]') tr.className = 'hl';
      tr.append(mk('td', null, n));
      BATCHES.forEach((b) => { const r = D.h100_speed.find((z) => z.policy === k && z.batch === b); tr.append(mk('td', null, r ? fx(num(r.speedup)) : '–')); });
      return tr;
    }));
  }

  // ---- render loop -----------------------------------------------------------------------------
  const RENDER = [renderTiles, renderLadder, renderHeat, renderAcc, renderCtx, renderEnergy, renderBonus, renderH100, renderH100Acc, renderH100Speed];
  function renderAll() {
    syncControls();
    RENDER.forEach((fn) => { try { fn(); } catch (e) { console.error(fn.name, e); } });
  }
  let lastW = 0, rt = null;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { if (Math.abs(window.innerWidth - lastW) > 20) { lastW = window.innerWidth; renderAll(); } }, 150); });
  lastW = window.innerWidth;
  renderAll();
  if (location.hash) { const t = document.getElementById(location.hash.slice(1)); if (t) t.scrollIntoView(); }
})();
