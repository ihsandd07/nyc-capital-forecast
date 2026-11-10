"""Build reports/dashboard.html: one self-contained page (data, charts and code inline; fonts from Google Fonts).

Every number comes from results.json or the saved prediction tables; each tile names its source field.
"""
import base64
import html
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config, figures
from .report import LIMITS, pct, spct

matplotlib.rcParams["svg.fonttype"] = "path"

TITLE = "NYC Capital Forecasts"

CSS = """
:root{
  /* Direction: a civic engineering report. Cool paper greys, one construction-orange accent, blue for model results. */
  --bg:#f3f5f4; --surface:#ffffff; --ink:#18212b; --muted:#566170; --line:#d9dfe3; --soft:#eaeef0;
  --accent:#c9611a; --accent-ink:#ffffff; --blue:#2a6fdb; --good:#1f8a4c; --bad:#b83227; --paper:#ffffff;
  --head:"Space Grotesk",system-ui,sans-serif; --body:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#11161b; --surface:#1a2129; --ink:#e7ecf1; --muted:#9aa6b4; --line:#2b343e; --soft:#222b34;
    --accent:#f08a3e; --accent-ink:#11161b; --blue:#78a7ff; --good:#4cc283; --bad:#ff7b6e; color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --bg:#11161b; --surface:#1a2129; --ink:#e7ecf1; --muted:#9aa6b4; --line:#2b343e; --soft:#222b34;
  --accent:#f08a3e; --accent-ink:#11161b; --blue:#78a7ff; --good:#4cc283; --bad:#ff7b6e; color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--body);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:0 48px}
.wrap{max-width:1040px;margin:0 auto}
header{padding-block:28px 12px}
h1{font:600 clamp(26px,5vw,38px)/1.1 var(--head);letter-spacing:-.01em;margin:0 0 8px;text-wrap:balance}
.sub{color:var(--muted);max-width:68ch;margin:0}
nav{display:flex;gap:4px;overflow-x:auto;border-bottom:1px solid var(--line);margin-top:20px;position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);z-index:5}
nav a{padding:10px 14px;color:var(--muted);text-decoration:none;white-space:nowrap;border-bottom:2px solid transparent;font-weight:500}
nav a[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--accent)}
nav a:hover{color:var(--ink)}
nav a:focus-visible,button:focus-visible,select:focus-visible,input:focus-visible,.th:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
section.panel{padding-top:24px}
h2{font:600 22px/1.2 var(--head);margin:0 0 10px;text-wrap:balance}
h3{font:600 16px/1.3 var(--head);margin:28px 0 8px}
p{margin:0 0 12px;max-width:72ch}
.lede{font-size:17px;max-width:66ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0 8px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 14px 12px;min-width:0}
.tile .k{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.tile .v{font:600 30px/1.15 var(--head);margin:6px 0 4px;font-variant-numeric:tabular-nums}
.tile .v.good{color:var(--good)} .tile .v.warn{color:var(--accent)}
.tile .d{font-size:13px;color:var(--muted)}
.src{font:11px/1.4 var(--mono);color:var(--muted);margin-top:8px;overflow-wrap:anywhere}
.figs{display:grid;gap:14px;margin:12px 0}
.figs.two{grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr))}
figure{margin:0;min-width:0}
figure .paper{background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:8px}
figure img{display:block;width:100%;height:auto;max-width:100%}
figcaption{font-size:13px;color:var(--muted);margin-top:6px;max-width:70ch}
.note{border-left:3px solid var(--accent);padding:2px 0 2px 12px;color:var(--ink);margin:14px 0;max-width:72ch}
ul.plain{padding-left:20px;margin:0 0 12px;max-width:72ch} ul.plain li{margin-bottom:6px}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{background:var(--soft);font-weight:600;white-space:nowrap}
td.n,th.n{text-align:right}
tbody tr:last-child td{border-bottom:0}
.th{cursor:pointer;user-select:none} .th[aria-sort="ascending"]::after{content:" \\25B2";font-size:10px} .th[aria-sort="descending"]::after{content:" \\25BC";font-size:10px}
.controls{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}
.controls label{display:flex;flex-direction:column;gap:3px;font-size:12px;color:var(--muted);min-width:0}
select,input[type=search]{font:inherit;font-size:14px;padding:7px 9px;border:1px solid var(--line);border-radius:5px;background:var(--surface);color:var(--ink);min-width:150px;max-width:100%}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:8px 0}
.seg button{font:inherit;font-size:14px;padding:8px 14px;border:0;background:var(--surface);color:var(--muted);cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--bg)}
button.more{font:inherit;font-size:14px;margin-top:10px;padding:8px 14px;border:1px solid var(--line);border-radius:5px;background:var(--surface);color:var(--ink);cursor:pointer}
.count{font-size:13px;color:var(--muted);margin:8px 0 6px}
.pname{font-weight:500} .pag{display:block;color:var(--muted);font-size:12px}
code{font:12.5px var(--mono);background:var(--soft);padding:1px 5px;border-radius:4px}
pre{font:12.5px/1.5 var(--mono);background:var(--soft);padding:12px;border-radius:6px;overflow-x:auto;margin:0 0 12px}
.bar{display:inline-block;height:8px;border-radius:2px;background:var(--blue);vertical-align:middle;margin-right:6px}
@media (max-width:520px){h2{font-size:20px} .tile .v{font-size:26px}}
@media (prefers-reduced-motion:no-preference){nav a{transition:color .15s,border-color .15s}}
"""

JS = r"""
(function(){
const D = JSON.parse(document.getElementById('data').textContent);
const $ = (s, r=document) => r.querySelector(s);
const TABS = ['overview','schedule','cost','completion','drivers','projects','method'];
function show(id){
  if(!TABS.includes(id)) id = 'overview';
  TABS.forEach(t => { $('#panel-'+t).hidden = t !== id; $('#tab-'+t).setAttribute('aria-selected', String(t === id)); });
}
window.addEventListener('hashchange', () => show(location.hash.slice(1)));
show(location.hash.slice(1));

const fmtInt = n => Math.round(n).toLocaleString('en-US');
const fmtMoney = n => n >= 1e9 ? '$' + (n/1e9).toFixed(2) + 'B' : n >= 1e6 ? '$' + (n/1e6).toFixed(1) + 'M' : '$' + fmtInt(n/1e3) + 'K';
const fmtDate = (iso, add=0) => {
  const d = new Date(iso + 'T00:00:00Z'); d.setUTCDate(d.getUTCDate() + Math.round(add));
  return d.toLocaleDateString('en-US', {month:'short', year:'numeric', timeZone:'UTC'});
};

function makeExplorer(cfg){
  const root = $(cfg.root);
  const state = {sort: cfg.sort, dir: cfg.dir, shown: 100, q: '', f: {}};
  root.innerHTML = '';
  const controls = document.createElement('div'); controls.className = 'controls'; root.appendChild(controls);
  cfg.filters.forEach(f => {
    const lab = document.createElement('label'); lab.textContent = f.label;
    let el;
    if(f.type === 'search'){ el = document.createElement('input'); el.type = 'search'; el.placeholder = f.placeholder || ''; }
    else { el = document.createElement('select'); f.options.forEach(o => { const op = document.createElement('option'); op.value = o[0]; op.textContent = o[1]; el.appendChild(op); }); }
    el.id = cfg.root.slice(1) + '-' + f.id;
    el.addEventListener('input', () => { state.f[f.id] = el.value; state.shown = 100; render(); });
    lab.appendChild(el); controls.appendChild(lab);
  });
  const count = document.createElement('div'); count.className = 'count'; root.appendChild(count);
  const wrap = document.createElement('div'); wrap.className = 'tablewrap'; root.appendChild(wrap);
  const more = document.createElement('button'); more.className = 'more'; more.textContent = 'Show 100 more';
  more.addEventListener('click', () => { state.shown += 100; render(); }); root.appendChild(more);

  function render(){
    let rows = cfg.rows.filter(r => cfg.filters.every(f => f.test(r, state.f[f.id] || '')));
    const col = cfg.cols.find(c => c.key === state.sort);
    rows = rows.slice().sort((a, b) => {
      const x = col.val(a), y = col.val(b);
      const c = (typeof x === 'string') ? x.localeCompare(y) : x - y;
      return state.dir === 'asc' ? c : -c;
    });
    count.textContent = 'Showing ' + fmtInt(Math.min(state.shown, rows.length)) + ' of ' + fmtInt(rows.length) + ' ' + cfg.noun;
    const table = document.createElement('table');
    const thead = table.createTHead().insertRow();
    cfg.cols.forEach(c => {
      const th = document.createElement('th'); th.textContent = c.label; th.className = 'th' + (c.num ? ' n' : '');
      th.tabIndex = 0; th.setAttribute('role', 'columnheader');
      th.setAttribute('aria-sort', c.key === state.sort ? (state.dir === 'asc' ? 'ascending' : 'descending') : 'none');
      const go = () => { if(state.sort === c.key) state.dir = state.dir === 'asc' ? 'desc' : 'asc'; else { state.sort = c.key; state.dir = c.num ? 'desc' : 'asc'; } render(); };
      th.addEventListener('click', go); th.addEventListener('keydown', e => { if(e.key === 'Enter' || e.key === ' '){ e.preventDefault(); go(); } });
      thead.appendChild(th);
    });
    const tb = table.createTBody();
    rows.slice(0, state.shown).forEach(r => {
      const tr = tb.insertRow();
      cfg.cols.forEach(c => { const td = tr.insertCell(); if(c.num) td.className = 'n'; c.cell(td, r); });
    });
    wrap.innerHTML = ''; wrap.appendChild(table);
    more.hidden = state.shown >= rows.length;
  }
  render();
}

// ---- schedule explorer
const S = D.schedule_rows;   // [agency, name, phase, forecast_iso, slip, lo, hi, p12]
const uniq = (rows, i) => Array.from(new Set(rows.map(r => r[i]))).sort();
function schedule(){
  makeExplorer({root:'#explore-schedule', rows:S, noun:'open projects', sort:'slip', dir:'desc',
    filters:[
      {id:'agency', label:'Agency', options:[['','All agencies']].concat(uniq(S,0).map(a => [a,a])), test:(r,v) => !v || r[0] === v},
      {id:'phase', label:'Phase', options:[['','All phases']].concat(uniq(S,2).map(a => [a,a])), test:(r,v) => !v || r[2] === v},
      {id:'q', label:'Search project name', type:'search', placeholder:'e.g. firehouse', test:(r,v) => !v || (r[1] || '').toLowerCase().includes(v.toLowerCase())},
    ],
    cols:[
      {key:'name', label:'Project', val:r => r[1] || '', cell:(td,r) => { const s = document.createElement('span'); s.className = 'pname'; s.textContent = r[1] || '(unnamed)'; const a = document.createElement('span'); a.className = 'pag'; a.textContent = r[0]; td.append(s, a); }},
      {key:'phase', label:'Phase', val:r => r[2], cell:(td,r) => td.textContent = r[2]},
      {key:'fc', label:'Agency forecast', val:r => r[3], cell:(td,r) => td.textContent = fmtDate(r[3])},
      {key:'exp', label:'Model expects', val:r => Date.parse(r[3]) + r[4]*864e5, cell:(td,r) => td.textContent = fmtDate(r[3], r[4])},
      {key:'slip', label:'Slip (days)', num:true, val:r => r[4], cell:(td,r) => td.textContent = (r[4] > 0 ? '+' : '') + fmtInt(r[4])},
      {key:'rng', label:'80% range (days)', num:true, val:r => r[6] - r[5], cell:(td,r) => td.textContent = fmtInt(r[5]) + ' to ' + fmtInt(r[6])},
      {key:'p12', label:'Completes within 12 months', num:true, val:r => r[7], cell:(td,r) => { const b = document.createElement('span'); b.className = 'bar'; b.style.width = Math.max(1, r[7]*60) + 'px'; td.append(b, document.createTextNode((r[7]*100).toFixed(0) + '%')); }},
    ]});
}
// ---- budget explorer
const C = D.cost_rows;       // [agency, fms_id, total, p_up, years_ahead]
function cost(){
  makeExplorer({root:'#explore-cost', rows:C, noun:'budget lines', sort:'p', dir:'desc',
    filters:[
      {id:'agency', label:'Agency', options:[['','All agencies']].concat(uniq(C,0).map(a => [a,a])), test:(r,v) => !v || r[0] === v},
      {id:'min', label:'Planned budget', options:[['0','Any size'],['1e6','$1M and up'],['1e7','$10M and up'],['1e8','$100M and up']], test:(r,v) => r[2] >= Number(v || 0)},
      {id:'q', label:'Search FMS ID', type:'search', placeholder:'e.g. HW', test:(r,v) => !v || r[1].toLowerCase().includes(v.toLowerCase())},
    ],
    cols:[
      {key:'id', label:'Budget line', val:r => r[1], cell:(td,r) => { const s = document.createElement('span'); s.className = 'pname'; s.textContent = r[1]; const a = document.createElement('span'); a.className = 'pag'; a.textContent = r[0]; td.append(s, a); }},
      {key:'tot', label:'Planned budget', num:true, val:r => r[2], cell:(td,r) => td.textContent = fmtMoney(r[2])},
      {key:'yrs', label:'Plan years ahead', num:true, val:r => r[4], cell:(td,r) => td.textContent = String(r[4])},
      {key:'p', label:'Chance of +10% within 12 months', num:true, val:r => r[3], cell:(td,r) => { const b = document.createElement('span'); b.className = 'bar'; b.style.width = Math.max(1, r[3]*120) + 'px'; td.append(b, document.createTextNode((r[3]*100).toFixed(0) + '%')); }},
    ]});
}
schedule(); cost();
const seg = document.querySelectorAll('.seg button');
seg.forEach(b => b.addEventListener('click', () => {
  seg.forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  $('#explore-schedule').hidden = b.dataset.t !== 'schedule'; $('#explore-cost').hidden = b.dataset.t !== 'cost';
  $('#explain-schedule').hidden = b.dataset.t !== 'schedule'; $('#explain-cost').hidden = b.dataset.t !== 'cost';
}));
})();
"""


def _svg_uri(fig, salt):
    matplotlib.rcParams["svg.hashsalt"] = salt
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode()


def _fig(uri, caption, alt):
    return (f'<figure><div class="paper"><img src="{uri}" alt="{html.escape(alt)}"></div>'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def _tile(label, value, detail, source, cls=""):
    return (f'<div class="tile"><div class="k">{html.escape(label)}</div><div class="v {cls}">{html.escape(value)}</div>'
            f'<div class="d">{html.escape(detail)}</div><div class="src">{html.escape(source)}</div></div>')


def _table(df, num_cols=()):
    head = "".join(f'<th class="{"n" if c in num_cols else ""}">{html.escape(str(c))}</th>' for c in df.columns)
    body = "".join("<tr>" + "".join(f'<td class="{"n" if c in num_cols else ""}">{html.escape(str(v))}</td>'
                                    for c, v in zip(df.columns, row)) + "</tr>" for row in df.itertuples(index=False))
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def build(R=None, path=None, fragment=False):
    R = R or figures.load_results()
    F = figures.make_all(R)
    uri = {k: _svg_uri(f, salt=k) for k, f in F.items()}
    t = config.REPORTS / "tables"
    cs = pd.read_csv(t / "current_schedule_forecasts.csv")
    cc = pd.read_csv(t / "current_cost_forecasts.csv")
    data = {
        "schedule_rows": [[r.agency, None if pd.isna(r.name) else str(r.name), r.phase, r.forecast_date, round(float(r.pred)), round(float(r.lo)),
                           round(float(r.hi)), round(float(r.p_complete_12m), 4)] for r in cs.itertuples()],
        "cost_rows": [[r.agency, str(r.unit).split("|", 1)[1], round(float(__import__("math").exp(r.log_total))), round(float(r.p_big_up_12m), 4),
                       int(r.yrs_ahead)] for r in cc.itertuples()],
    }
    c, s, sv, cb = R["cost"], R["schedule"], R["survival"], R["cost_big_up"]
    lm = {m["model"]: m for m in sv["landmark"]["metrics"]}
    cbm = {m["model"]: m for m in cb["metrics"]}
    cg, sg = c["mae_change_vs_zero"]["gbm"], s["mae_change_vs_zero"]["gbm"]
    dq, meta = R["data_quality"], R["meta"]
    d = sv["landmark"]["auc_diff_gbm_vs_rule"]
    _mon = lambda code: pd.Timestamp(f"{str(code)[:4]}-{str(code)[4:6]}-01").strftime("%B %Y")
    first, last = _mon(meta["periods"][0]), _mon(meta["periods"][-1])
    cur = R["current"]

    tiles = "".join([
        _tile("Schedule slip error", spct(sg["point"]), f"vs 'the date holds'. 95% CI {spct(sg['lo'])} to {spct(sg['hi'])}", "results.json > schedule.mae_change_vs_zero.gbm", "good"),
        _tile("Schedule 80% ranges", f"{pct(s['coverage']['cqr']['coverage'], 0)}", f"of held-out projects fell inside (CQR; split conformal {pct(s['coverage']['sc']['coverage'], 0)}). Target 80%", "results.json > schedule.coverage", "warn"),
        _tile("Budget change error", spct(cg["point"]), f"vs 'no change'. 95% CI {spct(cg['lo'])} to {spct(cg['hi'])}. Tiny", "results.json > cost.mae_change_vs_zero.gbm"),
        _tile("Budget up 10%+", f"AUC {cbm['gbm']['auc']:.2f}", f"95% CI {cb['auc_ci']['gbm'][0]:.2f} to {cb['auc_ci']['gbm'][1]:.2f}. {pct(cb['base_rate_test'], 0)} of lines rise this much", "results.json > cost_big_up"),
        _tile("Done within 12 months", pct(sv["p_complete_3"]), f"Kaplan-Meier. {sv['cohort']:,} projects, {sv['censored']:,} still open", "results.json > survival.p_complete_3"),
    ])

    agency_cov = pd.DataFrame(s["coverage_by_agency"]).sort_values("n", ascending=False)
    agency_cov = pd.DataFrame({"Agency": agency_cov.agency, "Test rows": agency_cov.n.astype(int), "Coverage (CQR, 80% target)": agency_cov.coverage.map(lambda v: pct(v, 0))})
    lmt = pd.DataFrame(sv["landmark"]["metrics"])
    lmt["Model"] = lmt.model.map(figures.NICE | {"rule": "Own forecast date only"})
    lm_tbl = pd.DataFrame({"Model": lmt.Model, "AUC": lmt.auc.map(lambda v: f"{v:.3f}"), "Brier skill vs base rate": lmt.brier_skill.map(lambda v: f"{v * 100:+.1f}%"),
                           "Log loss": lmt.logloss.map(lambda v: f"{v:.3f}")})
    ds = pd.DataFrame([
        ["Budget and Spend by Fiscal Year", f"{dq['budget_rows_raw']:,}", f"{dq['budget_units']:,} budget lines", "Planned budget by fiscal year at each report; spending for past years"],
        ["Schedule History and Variance", f"{dq['schedule_rows_raw']:,}", f"{dq['schedule_units']:,} projects", "Forecast or actual completion date at each report, phase, reason for change"],
        ["Capital Project Schedules and Budgets", f"{dq['sca_rows_raw']:,}", f"{R['sca']['complete_phases']:,} completed phases", "School-construction phases: planned and actual dates, budget and final estimate"]],
        columns=["Dataset", "Rows", "Covers", "Contents"])
    sca = pd.DataFrame(R["sca"]["by_phase"])
    sca_tbl = pd.DataFrame({"Phase": sca.phase, "Completed": sca.phases.astype(int), "With dates": sca.with_dates.astype(int),
                            "Late": sca.share_late.map(lambda v: "n/a" if pd.isna(v) else pct(v, 0)),
                            "Median delay (days)": sca.median_delay_days.map(lambda v: "n/a" if pd.isna(v) else f"{v:.0f}"),
                            "Over budget": sca.share_over_budget.map(lambda v: "n/a" if pd.isna(v) else pct(v, 0))})

    body = f"""
<div class="wrap">
<header>
  <h1>{TITLE}</h1>
  <p class="sub">Budget growth and schedule slippage on New York City capital projects, forecast from public NYC Open Data.
  City reports from {first} to {last} (three a year). Every forecast uses only data available at its forecast date, and every result below is from projects held out of training.</p>
  <nav role="tablist" aria-label="Sections">
    <a id="tab-overview" role="tab" href="#overview">Overview</a><a id="tab-schedule" role="tab" href="#schedule">Schedule</a>
    <a id="tab-cost" role="tab" href="#cost">Budget</a><a id="tab-completion" role="tab" href="#completion">Completion</a>
    <a id="tab-drivers" role="tab" href="#drivers">Drivers</a><a id="tab-projects" role="tab" href="#projects">Projects</a>
    <a id="tab-method" role="tab" href="#method">Method</a>
  </nav>
</header>

<section class="panel" id="panel-overview" role="tabpanel">
  <h2>What the held-out data show</h2>
  <p class="lede">Schedule slippage is partly predictable and the models beat "the date holds" by a clear margin. Budget growth is mostly not predictable as a number, but the chance of a large increase is. Uncertainty ranges were honest for budgets and too narrow for schedules.</p>
  <div class="tiles">{tiles}</div>
  <h3>Reading the results</h3>
  <ul class="plain">
    <li><b>Schedule.</b> Open projects' forecast completion dates moved a median {s['test_median_y']:.0f} days later over the next 12 months in the test period. Gradient boosting cut error by {abs(sg['point']) * 100:.1f}% versus assuming no move.</li>
    <li><b>Budget.</b> {pct(c['share_unchanged'], 0)} of test budget lines did not change at all, so "no change" is hard to beat. The planned-budget number here is the city's plan, not final cost.</li>
    <li><b>Uncertainty.</b> The slip rate drifted between the calibration and test periods, so 80% ranges for schedule covered {pct(s['coverage']['cqr']['coverage'], 0)}. Drift matters more than the interval method.</li>
    <li><b>Completion.</b> A project's own forecast date already ranks near-term completions well (AUC {lm['rule']['auc']:.2f}). Gradient boosting adds {d[0]:+.3f} (95% CI {d[1]:+.3f} to {d[2]:+.3f}), which is not a significant gain.</li>
  </ul>
  <p class="src">Test = the last two reports with a 12-month outcome (Jan and May 2025 forecasts). 95% intervals: 1,000 bootstrap resamples of projects.</p>
</section>

<section class="panel" id="panel-schedule" role="tabpanel" hidden>
  <h2>Schedule: how far completion dates slip in 12 months</h2>
  <p>For each open project, the target is how many days its forecast completion date moves over the next three city reports (about 12 months). If a project finishes, its actual completion date is used.
  {s['n_test_rows']:,} project-reports were held out.</p>
  <div class="figs">{_fig(uri['schedule_point'], 'Mean absolute error relative to assuming the forecast date holds. Bars to the left are better. Whiskers are 95% bootstrap intervals.', 'Schedule model error versus baseline')}
  {_fig(uri['slip_drift'], 'The median 12-month slip changes a lot from report to report. Shaded reports are the test forecasts.', 'Median 12-month slip by report')}</div>
  <h3>Prediction ranges</h3>
  <p>An 80% range should contain the true outcome for about 80% of projects. For budgets it does. For schedules it does not, because slippage rose after the calibration period.</p>
  <div class="figs">{_fig(uri['coverage'], 'Share of held-out projects inside the 80% interval, by method. Whiskers are 95% bootstrap intervals.', 'Interval coverage by method')}</div>
  <h3>Coverage by agency</h3>
  {_table(agency_cov, num_cols=('Test rows', 'Coverage (CQR, 80% target)'))}
</section>

<section class="panel" id="panel-cost" role="tabpanel" hidden>
  <h2>Budget: change in planned budget over 12 months</h2>
  <p>The target is the change in a budget line's total planned budget between one city report and the one three reports later (12 months). Spending is only recorded for past fiscal years and the public files do not hold the original approved budget, so this measures plan growth, not final cost overrun.
  {c['n_test_rows']:,} budget-line reports from {c['n_test_units']:,} lines were held out.</p>
  <div class="figs">{_fig(uri['cost_point'], 'Mean absolute error relative to assuming no change. Gradient boosting is better by a very small margin; the other models are worse.', 'Cost model error versus baseline')}</div>
  <h3>Risk of a large increase</h3>
  <p>Predicting the size of a change barely beats "no change". Predicting whether a budget line rises by 10% or more works much better.</p>
  <div class="figs">{_fig(uri['cost_classifier'], 'Left: ranking quality (ROC). Right: whether predicted probabilities match observed frequencies.', 'Large budget increase classifier')}</div>
  <p class="note">This classifier was added after the point forecasts showed almost no skill. It is reported whatever it shows.</p>
</section>

<section class="panel" id="panel-completion" role="tabpanel" hidden>
  <h2>Completion: survival analysis</h2>
  <p>{sv['censored']:,} of {sv['cohort']:,} projects are still open when the data ends. Dropping them would make projects look faster than they are, so Kaplan-Meier treats them as censored at their last report. Time counts reports since a project first appeared as open, about four months each.</p>
  <div class="figs">{_fig(uri['km'], 'Share of projects still open over time, overall (shaded = 95% confidence band) and for the six largest agencies.', 'Kaplan-Meier curves')}</div>
  <p>About {pct(sv['p_complete_3'])} complete within 12 months and {pct(sv['p_complete_9'], 0)} within 36 months. Completion rates differ widely by agency.</p>
  <h3>Will it complete in the next 12 months?</h3>
  <p>{lm['rule']['events']} of {lm['rule']['n']:,} test project-reports completed. The agency's own forecast date is already a strong ranker. Learned models add little, and the simpler models are miscalibrated because completion rates fell between training and test.</p>
  <div class="figs">{_fig(uri['landmark'], 'Left: ranking quality. Right: calibration of the gradient boosting probabilities.', 'Completion within 12 months')}</div>
  {_table(lm_tbl, num_cols=('AUC', 'Brier skill vs base rate', 'Log loss'))}
</section>

<section class="panel" id="panel-drivers" role="tabpanel" hidden>
  <h2>What the models rely on</h2>
  <p>Permutation importance shows how much worse a model gets when one feature is shuffled. Partial dependence shows the model's average prediction as one feature changes. They describe the models, not causes.</p>
  <div class="figs">{_fig(uri['drivers'], 'Permutation importance on the test set, with standard deviation over 10 shuffles.', 'Permutation importance')}
  {_fig(uri['pdp'], 'Average model prediction as one feature varies. The two schedule panels share a y-scale, and so do the two budget panels.', 'Partial dependence')}</div>
  <ul class="plain">
    <li>Schedule: how soon the forecast date is, the project phase and the agency matter most. Projects at or past their date slip the most. Earlier delays add little once those are known.</li>
    <li>Large budget increase: spending relative to budget, the agency and the size of the budget matter most. The pattern with size is not monotonic.</li>
  </ul>
</section>

<section class="panel" id="panel-projects" role="tabpanel" hidden>
  <h2>Project forecasts for the {_mon(cur['report'])} report</h2>
  <p>Model forecasts for every currently open project and budget line. They are not official city numbers.</p>
  <div class="seg" role="group" aria-label="Table"><button type="button" data-t="schedule" aria-pressed="true">Schedule ({cur['n_schedule']:,} open projects)</button><button type="button" data-t="cost" aria-pressed="false">Budget ({cur['n_cost']:,} lines)</button></div>
  <div id="explain-schedule"><p class="note">"Model expects" is the agency's forecast date plus the model's predicted slip. In testing, the 80% range covered {pct(s['coverage']['cqr']['coverage'], 0)} of projects, so treat ranges as too narrow. Click a column heading to sort.</p></div>
  <div id="explain-cost" hidden><p class="note">The public budget file identifies lines by agency and FMS ID only, without names. The probability comes from the large-increase classifier (test AUC {cbm['gbm']['auc']:.2f}). Click a column heading to sort.</p></div>
  <div id="explore-schedule"></div>
  <div id="explore-cost" hidden></div>
</section>

<section class="panel" id="panel-method" role="tabpanel" hidden>
  <h2>Method and limits</h2>
  <h3>Data</h3>
  {_table(ds, num_cols=('Rows',))}
  <ul class="plain" style="margin-top:12px">
    <li>{dq['budget_rows_missing_amount']:,} budget rows with no amount were dropped; {dq['budget_negative_rows']:,} rows hold negative adjustments and were kept.</li>
    <li>{dq['schedule_implausible_dates_removed']} completion dates with impossible years were set to missing; {dq['schedule_missing_completion_date']:,} schedule rows have no forecast date.</li>
    <li>In {dq['sca_planned_end_not_a_date']:,} school-phase rows the planned-end field holds a program label rather than a date.</li>
  </ul>
  <h3>Design</h3>
  <ul class="plain">
    <li>Each project at each report is one row. Features use only reports up to that date.</li>
    <li>Models train on older reports, calibrate on a newer one and test on the last two reports that have a 12-month outcome. A unit test checks that every training label was known at the test date.</li>
    <li>Models: "no change" and agency-median baselines, an L1-penalized median regression, and gradient boosting. Hyperparameters were fixed in advance.</li>
    <li>Intervals: conformalized quantile regression (named primary before the test run) and split conformal.</li>
  </ul>
  <h3>School-construction phases (descriptive only)</h3>
  <p>Only completed phases are observed, so long phases still open are missing, and many programs have no planned-end date. No forecast is made from this file.</p>
  {_table(sca_tbl, num_cols=('Completed', 'With dates', 'Late', 'Median delay (days)', 'Over budget'))}
  <h3>Limitations and analysis log</h3>
  <ul class="plain">{''.join(f'<li>{html.escape(x)}</li>' for x in LIMITS)}</ul>
  <h3>Reproduce</h3>
  <pre>pip install -r requirements.txt
python run_all.py</pre>
</section>
</div>
<script id="data" type="application/json">{json.dumps(data, separators=(",", ":")).replace("</", "<\\/")}</script>
<script>{JS}</script>
"""
    head = (f'<title>{TITLE}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">\n'
            f"<style>{CSS}</style>\n")
    page = head + body
    if not fragment:
        page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
                + head + "</head>\n<body>\n" + body + "</body>\n</html>\n")
    path = path or (config.REPORTS / ("dashboard_fragment.html" if fragment else "dashboard.html"))
    path.write_text(page, encoding="utf-8")
    return path
