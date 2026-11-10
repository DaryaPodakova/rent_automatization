"""Сборка одностраничного дашборда «Ассистент аренды в Петербурге» -> docs/index.html.

Страница — один файл: шаблон + данные, встроенные внутрь HTML (работает и с file://, и на GitHub Pages,
без внешних запросов). Формулы калькулятора берутся из econ_model.json как есть и вычисляются в браузере.

Запуск из корня репозитория:
    python rent_spb/dashboard/build_dashboard.py
    python rent_spb/dashboard/build_dashboard.py --dash data/dashboard_data.json --econ data/econ_model.json \
        --eval data/gold/eval_results.json --out docs/index.html

Файл с метриками разбора (--eval) необязателен: без него блок «правила против LLM» покажет «ждёт прогона».
В страницу попадают только агрегаты: тексты постов и контакты не читаются и не встраиваются.
"""
import argparse
import html as html_lib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # rent_spb/dashboard
REPO = HERE.parents[1]                            # корень репозитория
FINAL = REPO / "ITMO" / "AI" / "final_project" / "data"
DEFAULTS = {
    "dash": FINAL / "dashboard_data.json",
    "econ": FINAL / "econ_model.json",
    "eval": REPO / "rent_spb" / "data" / "gold" / "eval_results.json",
    "out": REPO / "rent_spb" / "docs" / "index.html",
}
REPO_URL = "https://github.com/DaryaPodakova/rent_automatization"
AUTHOR = "Дарья Подакова"

# Факты, которых нет в JSON-файлах (из протокола разметки эталона).
FACTS = {
    "second_annotator": "второй разметчик вслепую на 40 постах: цена, коммуналка (статус и сумма), комиссия, "
                        "залог, метро — совпадение 100%, каппа 1,0; тип сделки — 97,5% (каппа 0,96)",
}

# Поля разбора: ключ в eval_results, подпись, тип метрики.
FIELDS = [
    ("price_rub", "Цена аренды", "num"),
    ("utilities_status", "Коммуналка: статус", "status"),
    ("utilities_rub", "Коммуналка: сумма", "num"),
    ("commission_status", "Комиссия: статус", "status"),
    ("deposit_status", "Залог: статус", "status"),
    ("deal", "Срок: долгая или посуточно", "status"),
    ("move_from", "Дата заезда", "acc"),
    ("metro", "Станция метро", "acc"),
    ("metro_walk_min", "Минут до метро", "num"),
]


def load(path, required=True):
    p = Path(path)
    if not p.exists():
        if required:
            sys.exit(f"нет файла: {p}")
        print(f"[!] нет файла {p.name} — блок покажет «ждёт прогона»")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def kpi(dash):
    """Ключевые цифры шапки считаются из тех же данных, что и графики, — страница не расходится сама с собой."""
    pts = dash["price_vs_full"]["points"]
    xc = dash.get("extra_cost")
    if not xc:
        sys.exit("в dashboard_data.json нет блока extra_cost — перезапустите build_data_nb.py")
    # главная цифра и нижняя граница — из генератора данных (те же, что в отчёте); точки — сверка
    extra = statistics.median(p["est"] - p["price"] for p in pts)
    over10 = sum((p["est"] - p["price"]) / p["price"] > 0.10 for p in pts) / len(pts)
    if len(pts) != xc["n"] or abs(extra - xc["est_median"]) > 1 or abs(over10 - xc["est_over10"]) > 0.001:
        print(f"[!] точки расходятся с extra_cost: {len(pts)}/{xc['n']}, {extra:.0f}/{xc['est_median']:.0f}, "
              f"{over10:.3f}/{xc['est_over10']:.3f}")
    rel = dash["reliability"]
    n = rel["n"]
    no_sum = {s: (rel["util_status_pct"][s].get("не указано", 0) + rel["util_status_pct"][s].get("счётчики сверху", 0))
              / 100 for s in n}
    no_sum_all = sum(n[s] * no_sum[s] for s in n) / sum(n.values())
    top = dash["price_vs_full"]["top10_change"]["total_month_est"]
    return {
        "n_points": xc["n"],
        "median_extra": xc["est_median"],
        "share_over10": xc["est_over10"],
        "median_extra_known": xc["known_median"],
        "share_over10_known": xc["known_over10"],
        "commission_share": xc["commission_share"],
        "no_sum_all": no_sum_all,
        "no_sum_by_source": no_sum,
        "n_by_source": n,
        "n_all": sum(n.values()),
        "top10_dropped": round(top["dropped_pct"] / 10),
        "tau": top["kendall_tau"],
    }


def slim_eval(ev):
    """Из eval_results.json — только агрегаты по полям (регулярки, LLM и гибрид, если есть) и «потолок»."""
    if not ev:
        return {"available": False}
    methods = ev.get("methods", {})
    weights = ev.get("pool_weights", {})
    layers = ev.get("layers", {})
    base = methods.get("baseline", {})

    def denom(lay):   # объявлений в слое: самое большое число «известно» по полям, не больше размера слоя
        f = base.get(lay, {})
        c = [v.get("n_known") for v in f.values() if isinstance(v, dict) and v.get("n_known") is not None]
        c += [v.get("tp", 0) + v.get("fn", 0) for v in f.values() if isinstance(v, dict) and "tp" in v]
        return min(max(c or [0]), layers.get(lay, 0)) or layers.get(lay, 0)

    dens = {lay: denom(lay) for lay in weights}

    def known(m, lay, key, kind):   # (сколько постов с полем, полнота на них) в слое
        f = methods.get(m, {}).get(lay, {})
        if kind == "status":
            v = f.get(f"{key} (известно)") or {}
            return v.get("tp", 0) + v.get("fn", 0), v.get("recall")
        if kind == "num":
            v = f.get(key) or {}
            return v.get("n_known", 0) or 0, v.get("exact_known")
        return 0, None

    def ceiling(key, kind):
        if kind == "acc":
            return None
        tot = sum(weights[lay] * known("baseline", lay, key, kind)[0] / dens[lay] for lay in weights if dens[lay])
        return round(tot, 4)

    def metric(m, key, kind):
        if m not in methods or "взвешенно" not in methods[m]:
            return None
        v = methods[m]["взвешенно"].get(key) or {}
        acc = v.get("accuracy", v.get("exact_all"))
        num = den = 0.0
        for lay, w in weights.items():
            k, r = known(m, lay, key, kind)
            if k and r is not None and dens[lay]:
                num += w * k / dens[lay] * r
                den += w * k / dens[lay]
        rec = round(num / den, 4) if den else None
        return {"acc": None if acc is None else round(acc, 4), "rec": rec}

    fields = [{"key": k, "label": lab, "kind": kind, "ceiling": ceiling(k, kind),
               "regex": metric("baseline", k, kind), "llm": metric("llm", k, kind),
               "hybrid": metric("hybrid", k, kind)} for k, lab, kind in FIELDS]
    routing = ev.get("routing", {})
    part = ev.get("hybrid_llm_part") or {}
    return {
        "available": True,
        "date": ev.get("date"),
        "n": ev.get("n"),
        "layers": layers,
        "weights": weights,
        "n_offers": ev.get("n_offers"),
        "full_price_share": (ev.get("gold_full_price_share") or {}).get("взвешенно"),
        "routing_share": (routing.get("взвешенно") or {}).get("share"),
        "routing_layers": {lay: {"n": v.get("n"), "llm": v.get("llm")} for lay, v in routing.items()
                           if isinstance(v, dict) and "n" in v},
        "llm_posts": part.get("posts"),
        "llm_status": part.get("status"),
        "has_llm": "llm" in methods,
        "has_hybrid": "hybrid" in methods,
        "llm_run": {k: v for k, v in (ev.get("llm_run") or {}).items() if k in ("model", "calls", "invalid_cards")},
        "fields": fields,
        "second_annotator": FACTS["second_annotator"],
    }


def emb(obj):
    """JSON для <script type="application/json">: без закрывающего тега внутри."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def css_theme_blocks():
    light = """color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --raised:#f3f2ee; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s1-soft:#b7d3f6;
  --g-top:#e4e3dc; --g-bot:#b1b0a8; --good:#006300; --bad:#c03030; --good-bg:#e6f3e6; --bad-bg:#fbe8e6;"""
    dark = """color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --raised:#242422; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s1-soft:#104281;
  --g-top:#33332f; --g-bot:#5e5d58; --good:#0ca30c; --bad:#e66767; --good-bg:#102b10; --bad-bg:#331414;"""
    return (f":root{{\n  {light}\n}}\n"
            f"@media (prefers-color-scheme: dark){{\n  :root:not([data-theme=\"light\"]){{\n  {dark}\n  }}\n}}\n"
            f":root[data-theme=\"dark\"]{{\n  {dark}\n}}\n")


CSS = r"""
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--s1)}
.wrap{max-width:1120px;margin:0 auto;padding:0 16px}
header.top{padding:28px 0 8px}
.toprow{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
h1{font-size:28px;line-height:1.2;margin:0 0 8px;font-weight:650;letter-spacing:-.01em}
.lead{margin:0 0 18px;color:var(--ink2);max-width:760px}
.btn,.seg button{font:inherit;font-size:13px;color:var(--ink);background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:6px 10px;cursor:pointer;white-space:nowrap}
.btn:hover,.seg button:hover{background:var(--raised)}
.btn:focus-visible,.seg button:focus-visible,input:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
.seg{display:inline-flex;flex-wrap:wrap;gap:6px}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:0 0 18px}
.chapter-head{padding:44px 0 18px}
.chapter-head .kicker{font-size:13px;color:var(--ink2);text-transform:uppercase;letter-spacing:.06em;margin:0 0 6px}
.chapter-head h2{font-size:26px;line-height:1.25;margin:0 0 10px;font-weight:650}
.chapter-head p{color:var(--ink2);max-width:760px;margin:0}
.chapter-nav{display:flex;flex-direction:column;align-items:center;gap:6px;margin:4px 0 40px;text-align:center}
.chapter-nav a{font:inherit;font-size:16px;font-weight:650;color:#fff;background:var(--s3);border:1px solid var(--s3);
  border-radius:999px;padding:13px 26px;text-decoration:none;box-shadow:0 3px 14px rgba(27,175,122,.3);transition:transform .15s,filter .15s}
.chapter-nav a:hover{filter:brightness(1.08);transform:translateY(-1px)}
.chapter-nav a:active{transform:translateY(0)}
.chapter-nav .teaser{font-size:12px;color:var(--ink2)}
.chapter-nav a.secondary{color:var(--ink);background:var(--surface);border-color:var(--border);box-shadow:none;font-weight:600;font-size:14px}
.chapter-nav a.secondary:hover{background:var(--raised)}
.hero-story{border-left:3px solid var(--s3);background:var(--raised);padding:14px 16px;border-radius:0 8px 8px 0;margin:12px 0 0;font-size:14px}
.time-note{margin:18px 0 0;padding:14px 16px;border:1px dashed var(--border);border-radius:10px;font-size:14px;color:var(--ink2)}
.scroll-rows{max-height:440px;overflow-y:auto;border:1px solid var(--border);border-radius:10px}
.be-badge{display:flex;flex-wrap:wrap;gap:4px 20px;align-items:baseline;padding:14px 18px;border-radius:12px;margin:0 0 16px;font-size:14px;border:1px solid var(--border)}
.be-badge.good{background:var(--good-bg)}
.be-badge.bad{background:var(--bad-bg)}
.be-badge b{font-size:18px;font-variant-numeric:tabular-nums}
.be-badge.good b{color:var(--good)}
.be-badge.bad b{color:var(--bad)}
table.bd{border-collapse:collapse;width:100%;font-size:13px;margin-top:4px}
table.bd td,table.bd th{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:left}
table.bd td.n,table.bd th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.bd tr.grp td{font-weight:600;padding-top:14px;border-bottom:1px solid var(--axis)}
table.bd tr.sub td:first-child{padding-left:22px;color:var(--ink2)}
table.bd tr.total td{font-weight:650;border-top:2px solid var(--axis);border-bottom:0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.tile .lab{font-size:13px;color:var(--ink2);line-height:1.3}
.tile .val{font-size:30px;font-weight:650;line-height:1.15;margin:6px 0 4px}
.tile .sub{font-size:12px;color:var(--ink2);line-height:1.35}
section.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px 22px;margin:0 0 20px}
section.card h2{font-size:20px;line-height:1.25;margin:0 0 6px;font-weight:650}
.sub2{color:var(--ink2);margin:0 0 14px;max-width:820px}
.note{font-size:13px;color:var(--ink2);margin:10px 0 0;max-width:860px}
.callout{border-left:3px solid var(--s1);background:var(--raised);padding:10px 14px;border-radius:0 8px 8px 0;margin:14px 0 0;font-size:14px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:0 0 8px;font-size:13px;color:var(--ink2)}
.legend span.it{display:inline-flex;align-items:center;gap:6px}
.legend svg{flex:none}
.chart{width:100%;position:relative}
.chart svg{display:block;width:100%;overflow:visible}
svg text{font-family:inherit}
.ax{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.axl{fill:var(--ink2);font-size:12px}
.gl{stroke:var(--grid);stroke-width:1}
.bl{stroke:var(--axis);stroke-width:1}
.lab{fill:var(--ink);font-size:13px}
.lab2{fill:var(--ink2);font-size:12px}
.valt{fill:var(--ink);font-size:13px;font-weight:600;font-variant-numeric:tabular-nums}
.f1{fill:var(--s1)}.f2{fill:var(--s2)}.f3{fill:var(--s3)}.fsoft{fill:var(--s1-soft)}.fgt{fill:var(--g-top)}.fgb{fill:var(--g-bot)}
.st1{stroke:var(--s1)}.st2{stroke:var(--s2)}.st3{stroke:var(--s3)}
.ring{stroke:var(--surface);stroke-width:2}
.hot{opacity:.78}
.row-hit{fill:transparent;cursor:default}
details.tv{margin-top:12px;font-size:13px}
details.tv summary{cursor:pointer;color:var(--ink2)}
table.t{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
table.t th,table.t td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
table.t td.n,table.t th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.t code{font-size:12px;word-break:break-word}
.tscroll{max-width:100%;overflow-x:auto}
/* разбор полей */
.ext{display:grid;grid-template-columns:minmax(150px,1.3fr) repeat(3,minmax(0,1fr));gap:0;font-size:13px}
.ext>div{padding:8px 8px;border-bottom:1px solid var(--grid)}
.ext .h{color:var(--ink2);font-weight:600;font-size:12px}
.ext .f{font-weight:600;color:var(--ink)}
.meter{height:8px;border-radius:4px;background:var(--raised);overflow:hidden;margin:4px 0 2px}
.meter>span{display:block;height:100%;border-radius:4px}
.mv{font-variant-numeric:tabular-nums;font-weight:600}
.ms{color:var(--ink2);font-size:12px}
.wait{color:var(--ink2);font-style:italic}
/* калькулятор */
.calc-bar{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:0 0 14px}
.calc-bar .lbl{font-size:13px;color:var(--ink2);margin-right:4px}
.otiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:0 0 18px}
.otiles .tile{padding:12px 14px}
.otiles .val{font-size:24px}
.otiles .val.txt{font-size:18px;line-height:1.25;padding-top:4px}
.calc-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:22px;align-items:start}
.calc-side{position:sticky;top:12px}
details.grp{border:1px solid var(--border);border-radius:10px;margin:0 0 10px;background:var(--surface)}
details.grp>summary{cursor:pointer;padding:10px 14px;font-weight:600;font-size:14px;list-style-position:inside}
.grp-body{padding:2px 14px 12px}
.prm{padding:8px 0;border-top:1px solid var(--grid)}
.prm:first-child{border-top:0}
.prm-top{display:flex;flex-wrap:wrap;justify-content:space-between;gap:2px 10px;align-items:baseline}
.prm-top label{font-size:13px;line-height:1.3;flex:1 1 190px}
.prm-top output{font-size:13px;font-weight:600;white-space:nowrap;font-variant-numeric:tabular-nums;margin-left:auto}
.prm-top output small{font-weight:400;color:var(--ink2);margin-left:4px}
.prm input[type=range]{width:100%;margin:6px 0 0;accent-color:var(--s1)}
.info{border:0;background:none;color:var(--ink2);cursor:pointer;font-size:12px;padding:0 0 0 4px}
.prm-src{font-size:12px;color:var(--ink2);margin-top:4px}
.num-in{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}
.num-in label{font-size:13px;display:flex;flex-direction:column;gap:3px}
.num-in input{font:inherit;font-size:14px;width:130px;padding:5px 8px;border:1px solid var(--axis);border-radius:6px;background:var(--page);color:var(--ink)}
.derived{font-size:12px;color:var(--ink2);margin-top:6px}
#model-bar .num-in{margin:0}
#model-bar .num-in input{width:110px}
.need{margin:16px 0 0;padding:14px 16px;border:1px solid var(--border);border-radius:10px;background:var(--raised)}
.need h3{font-size:15px;margin:0 0 8px}
.need ol{margin:0;padding-left:20px}
.need li{margin:4px 0;font-size:14px}
.need .na{color:var(--ink2)}
.sticky-sum{position:sticky;bottom:0;z-index:5;margin:16px -22px -20px;padding:10px 22px;background:var(--surface);
  border-top:1px solid var(--border);border-radius:0 0 14px 14px;font-size:14px;display:flex;flex-wrap:wrap;gap:4px 18px}
.sticky-sum b{font-variant-numeric:tabular-nums}
h3.ch{font-size:15px;margin:18px 0 4px}
.ch-sub{font-size:13px;color:var(--ink2);margin:0 0 8px}
/* подсказка */
#tip{position:fixed;z-index:50;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);
  box-shadow:0 4px 18px rgba(0,0,0,.14);border-radius:8px;padding:8px 10px;font-size:13px;max-width:280px;line-height:1.35}
#tip .tt-title{color:var(--ink2);font-size:12px;margin-bottom:4px}
#tip .tt-row{display:flex;align-items:center;gap:6px;margin:2px 0}
#tip .tt-key{width:12px;height:2px;border-radius:1px;flex:none}
#tip strong{font-variant-numeric:tabular-nums}
#tip .tt-lab{color:var(--ink2)}
footer{padding:8px 0 40px;color:var(--ink2);font-size:13px}
footer p{margin:6px 0}
@media (max-width:900px){
  .calc-grid{grid-template-columns:1fr}
  .calc-side{position:static}
}
@media (max-width:600px){
  h1{font-size:23px}
  section.card{padding:16px 16px;border-radius:12px}
  .sticky-sum{margin:16px -16px -16px;padding:10px 16px}
  .tile .val{font-size:24px}
  .otiles .val{font-size:20px}
  .otiles .val.txt{font-size:16px}
  .sticky-sum{font-size:13px;gap:2px 12px}
  .ext{grid-template-columns:repeat(3,minmax(0,1fr))}
  .ext .h{display:none}
  .ext .f{grid-column:1/-1;border-bottom:0;padding-bottom:0;padding-top:14px}
  .ext>div[data-l]::before{content:attr(data-l);display:block;font-size:11px;color:var(--ink2)}
  .ext .cl{grid-column:1/-1}
}
@media (forced-colors: active){ .meter>span{background:CanvasText!important} }
"""

BODY_HEAD = r"""
<header class="top wrap">
  <div class="toprow">
    <div>
      <h1>Ассистент аренды в Петербурге</h1>
      <p class="lead">Бот читает объявления Telegram и Циана, считает полную цену в месяц (аренда + коммуналка +
        комиссия) и поднимает выше квартиры рядом с местами, важными человеку: метро, зал, садик, каток.</p>
    </div>
    <button class="btn" id="theme-btn" type="button" aria-label="Переключить тему">Тема: авто</button>
  </div>
</header>

<main class="wrap">
  <section id="chapter-1">
    <div class="kpis" id="kpis"></div>
    <div class="hero-story" id="hero-story"></div>
    <p class="time-note" id="time-note"></p>
    <nav class="chapter-nav">
      <a href="#chapter-2">Дальше: сколько вы на самом деле переплатите →</a>
      <p class="teaser">Покажем квартиры с самой большой доплатой и места, которые объявление скрывает</p>
    </nav>
  </section>

  <section class="chapter-head" id="chapter-2">
    <p class="kicker">Глава 2</p>
    <h2>Сюрприз доплаты и что город прячет рядом с домом</h2>
    <p>Как на самом деле устроена цена аренды и какие места рядом с квартирой объявление никогда не упомянет.</p>
  </section>

  <section class="card" id="s-surprise">
    <h2>Сюрприз доплаты</h2>
    <p class="sub2" id="scatter-sub"></p>
    <div class="legend" id="scatter-legend"></div>
    <div class="scroll-rows"><div class="chart" id="scatter-chart" role="img"></div></div>
    <p class="note">Циан, долгая аренда, студии и 1-к в одном районе. У каждой квартиры — цена из объявления (левая
      точка) и полная цена в месяц (правая точка): аренда + коммуналка + комиссия / 11 месяцев. Где коммуналка не
      указана, она оценена медианой указанных сумм по формату, тонкий отрезок — разброс оценки p25–p75.</p>
    <div class="callout" id="top10-callout"></div>
    <details class="tv"><summary>Таблица к графику (сводка)</summary><div class="tscroll" id="scatter-table"></div></details>
    <details class="tv"><summary>Все объявления по доплате</summary><div class="tscroll" id="scatter-table-full"></div></details>
  </section>

  <section class="card" id="s-drivers">
    <h2>Что делают драйверы интереса</h2>
    <p class="sub2" id="drv-sub"></p>
    <div class="calc-bar"><span class="lbl">Каток:</span>
      <div class="seg" id="ice-seg" role="group" aria-label="Какие катки учитывать">
        <button type="button" data-v="all" aria-pressed="true">все объекты</button>
        <button type="button" data-v="named" aria-pressed="false">только с названием</button>
      </div>
    </div>
    <div class="chart" id="drv-chart" role="img"></div>
    <p class="note">Баллы — по формуле поиска (раздел 5.1 отчёта): близость 0 при ≥ 20 мин пешком, максимум — при
      ≤ 5 мин; место, выбранное единственным после метро, весит 0,5 (= 50 баллов из 100). Левая точка — 0 (в
      объявлении это место никак не названо), правая — медиана баллов среди квартир, которые и правда рядом.
      Минуты — пешком по OpenStreetMap, только у объявлений с точкой на доме.</p>
    <details class="tv"><summary>Таблица к графику</summary><div class="tscroll" id="drv-table"></div></details>
  </section>

  __FLOW_SECTION__
  <nav class="chapter-nav">
    <a href="#chapter-3">Дальше: сойдётся ли тут экономика →</a>
    <p class="teaser">Покажем, где ИИ справляется сам, а где нет, и честный расчёт прибыли</p>
  </nav>

  <section class="chapter-head" id="chapter-3">
    <p class="kicker">Глава 3</p>
    <h2>Как ИИ читает объявления и сходится ли экономика</h2>
    <p>Что из текста объявления достаёт правило, а что — языковая модель, и выдерживает ли всё это юнит-экономику.</p>
  </section>

  <section class="card" id="s-extract">
    <h2>Разбор объявлений: правила против LLM</h2>
    <p class="sub2" id="ext-sub"></p>
    <div class="legend" id="ext-legend"></div>
    <div class="ext" id="ext"></div>
    <p class="note" id="ext-note"></p>
  </section>

  <section class="card" id="calc">
    <h2>Калькулятор юнит-экономики</h2>
    <p class="sub2">Все формулы — из модели проекта (econ_model.json), считаются прямо в браузере. Двигайте
      ползунки: плашка, таблица и подбор «чтобы окупиться» пересчитываются сразу. Один платящий — один поиск
      квартиры. Сценарий меняет рынок, а модель и курс оставляет как выбраны.</p>
    <div class="be-badge" id="be-badge"></div>
    <div class="calc-bar">
      <span class="lbl">Сценарий:</span>
      <div class="seg" id="scn-seg" role="group" aria-label="Сценарий"></div>
      <button type="button" class="btn" id="reset-btn">Сброс</button>
    </div>
    <div class="calc-bar" id="model-bar"><span class="lbl">Модель:</span></div>
    <div class="otiles" id="otiles"></div>
    <div class="calc-grid">
      <div id="params"></div>
      <div class="calc-side">
        <h3 class="ch" style="margin-top:0">Доходы и расходы в месяц</h3>
        <p class="ch-sub" id="curve-sub"></p>
        <div class="tscroll"><table class="bd" id="bd-table"></table></div>
        <div class="need" id="need"></div>
      </div>
    </div>
    <details class="tv"><summary>Фиксированные допущения (без ползунков)</summary><div class="tscroll" id="fixed-table"></div></details>
    <details class="tv"><summary>Все формулы модели и текущие значения</summary><div class="tscroll" id="formula-table"></div></details>
    <div class="sticky-sum" id="sticky-sum" aria-live="polite"></div>
  </section>

  <nav class="chapter-nav"><a href="#chapter-1" class="secondary">↑ Сначала</a></nav>
</main>

<footer class="wrap">
  <p><b>Источники.</b> Геоданные и минуты пешком — © участники OpenStreetMap, лицензия ODbL. Объявления Циана
    и публичного Telegram-канала об аренде собраны автором для учебного проекта и обезличены: на странице только
    агрегаты, без адресов и контактов. Цены моделей — каталог OpenRouter.</p>
  <p id="foot-dates"></p>
  <p>__AUTHOR__ · <a href="__REPO_URL__">__REPO_URL__</a></p>
</footer>
"""

# «Сколько подходящих объявлений в месяц» скрыта до экспериментального прогона агента (нужен реальный поиск,
# не только исторический подсчёт по комбинациям драйверов) — решение от 07.11.2026. Вернуть: SHOW_FLOW = True.
SHOW_FLOW = False
FLOW_SECTION = r"""
  <section class="card" id="s-flow">
    <h2>Сколько подходящих объявлений в месяц</h2>
    <p class="sub2" id="flow-sub"></p>
    <div class="legend" id="flow-legend"></div>
    <div class="chart" id="flow-chart" role="img"></div>
    <p class="note" id="flow-note"></p>
    <details class="tv"><summary>Таблица к графику</summary><div class="tscroll" id="flow-table"></div></details>
  </section>
""" if SHOW_FLOW else ""
BODY = BODY_HEAD.replace("__FLOW_SECTION__", FLOW_SECTION)

JS_CORE = r"""
(function (root) {
  'use strict';
  var FUNCS = {
    CEILING: function (x, s) { return Math.ceil(x / s) * s; },
    IF: function (c, a, b) { return c ? a : b; },
    MAX: function (a, b) { return Math.max(a, b); },
    MIN: function (a, b) { return Math.min(a, b); }
  };
  var FNAMES = ['CEILING', 'IF', 'MAX', 'MIN', 'NEVER'];
  var FVALS = [FUNCS.CEILING, FUNCS.IF, FUNCS.MAX, FUNCS.MIN, Infinity];
  // Компилирует формулы econ_model.json по порядку списка. Выражения проверяются белым списком символов
  // и имён: допускаются только допущения, формулы выше по списку и функции CEILING / IF / MAX / MIN / NEVER.
  function compile(econ) {
    var known = econ.params.map(function (p) { return p.key; });
    var steps = econ.formulas.map(function (f) {
      var expr = String(f.expr);
      if (!/^[A-Za-z0-9_\s.+\-*\/(),<>=!?:&|]+$/.test(expr)) throw new Error('недопустимые символы: ' + f.key);
      var ids = expr.replace(/\b\d+(\.\d+)?(e[+-]?\d+)?\b/gi, ' ').match(/[A-Za-z_][A-Za-z0-9_]*/g) || [];
      ids.forEach(function (id) {
        if (known.indexOf(id) < 0 && FNAMES.indexOf(id) < 0) throw new Error('неизвестное имя ' + id + ' в ' + f.key);
      });
      var args = known.slice();
      // без "use strict": в модели есть ключ static — в строгом режиме это зарезервированное слово
      var fn = new Function(args.concat(FNAMES).join(','), 'return (' + expr + ');');
      known.push(f.key);
      return { key: f.key, args: args, fn: fn };
    });
    return function compute(inputs) {
      var v = {};
      econ.params.forEach(function (p) {
        var x = inputs && inputs[p.key];
        v[p.key] = (x === undefined || x === null || x === '') ? p.base : +x;
      });
      steps.forEach(function (s) {
        var a = s.args.map(function (k) { return v[k]; });
        v[s.key] = s.fn.apply(null, a.concat(FVALS));
      });
      return v;
    };
  }
  root.RentCalc = { compile: compile, FUNCS: FUNCS };
})(typeof window !== 'undefined' ? window : globalThis);
"""

JS_UI = r"""
(function () {
'use strict';
var D = JSON.parse(document.getElementById('dash-data').textContent);
var E = JSON.parse(document.getElementById('econ-data').textContent);
var V = JSON.parse(document.getElementById('eval-data').textContent);
var K = JSON.parse(document.getElementById('kpi-data').textContent);
var NS = 'http://www.w3.org/2000/svg';
function $(s) { return document.querySelector(s); }
function S(tag, attrs, parent) {
  var e = document.createElementNS(NS, tag);
  if (attrs) for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function T(parent, x, y, text, attrs) {
  var t = S('text', Object.assign({ x: x, y: y }, attrs || {}), parent); t.textContent = text; return t;
}
function H(tag, attrs, parent, text) {
  var e = document.createElement(tag);
  if (attrs) for (var k in attrs) { if (k === 'class') e.className = attrs[k]; else if (attrs[k] != null) e.setAttribute(k, attrs[k]); }
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
// ---------- числа по-русски ----------
function num(x, d) {
  d = d || 0;
  if (x === null || x === undefined || !isFinite(x)) return '—';
  var neg = x < 0 && Math.abs(x) >= 0.5 * Math.pow(10, -d);
  var s = Math.abs(x).toFixed(d).split('.');
  var i = s[0].replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
  return (neg ? '−' : '') + i + (s[1] ? ',' + s[1] : '');
}
function pct(x, d) { return (x === null || x === undefined || !isFinite(x)) ? '—' : num(x * 100, d || 0) + '%'; }
function rub(x) { return num(x) + ' ₽'; }
function decimals(step) {
  var s = String(step); if (s.indexOf('e-') > 0) return +s.split('e-')[1];
  var p = s.split('.')[1]; return p ? Math.min(p.length, 4) : 0;
}
// ---------- подсказка ----------
var tip = H('div', { id: 'tip', role: 'tooltip' }, document.body); tip.hidden = true;
function placeTip(ev) {
  var pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  var x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (x < 8) x = 8;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  if (y < 8) y = 8;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
function tipShow(ev, title, rows) {
  tip.replaceChildren();
  if (title) H('div', { class: 'tt-title' }, tip, title);
  rows.forEach(function (r) {
    var row = H('div', { class: 'tt-row' }, tip);
    if (r.c) { var k = H('span', { class: 'tt-key' }, row); k.style.background = 'var(' + r.c + ')'; }
    H('strong', null, row, r.v);
    if (r.l) H('span', { class: 'tt-lab' }, row, r.l);
  });
  tip.hidden = false; placeTip(ev);
}
function tipHide() { tip.hidden = true; }
document.addEventListener('pointerdown', tipHide);
window.addEventListener('scroll', tipHide, { passive: true });
function bindTip(node, getContent, hotNode) {
  var hn = hotNode || node;
  function show(ev) { var c = getContent(); tipShow(ev, c.title, c.rows); hn.classList.add('hot'); }
  node.addEventListener('pointerenter', show);
  node.addEventListener('pointermove', function (ev) { if (!tip.hidden) placeTip(ev); });
  node.addEventListener('pointerleave', function (ev) { if (ev.pointerType !== 'touch') { tipHide(); } hn.classList.remove('hot'); });
  node.addEventListener('pointerdown', function (ev) { show(ev); ev.stopPropagation(); });
  node.addEventListener('focus', function () { var r = node.getBoundingClientRect(); show({ clientX: r.left + r.width / 2, clientY: r.top + 4 }); });
  node.addEventListener('blur', function () { tipHide(); hn.classList.remove('hot'); });
}
function table(host, head, rows, numCols) {
  host.replaceChildren();
  var t = H('table', { class: 't' }, host), tr = H('tr', null, H('thead', null, t));
  head.forEach(function (h, i) { H('th', { class: numCols && numCols.indexOf(i) >= 0 ? 'n' : null }, tr, h); });
  var tb = H('tbody', null, t);
  rows.forEach(function (r) {
    var row = H('tr', null, tb);
    r.forEach(function (c, i) { H('td', { class: numCols && numCols.indexOf(i) >= 0 ? 'n' : null }, row, c); });
  });
  return t;
}
function swatch(parent, kind, cls) {
  var s = S('svg', { width: kind === 'line' ? 18 : 12, height: 12, viewBox: kind === 'line' ? '0 0 18 12' : '0 0 12 12', 'aria-hidden': 'true' }, parent);
  if (kind === 'line') S('line', { x1: 1, y1: 6, x2: 17, y2: 6, class: cls, 'stroke-width': 2, 'stroke-linecap': 'round' }, s);
  else if (kind === 'circle') S('circle', { cx: 6, cy: 6, r: 4.5, class: cls }, s);
  else if (kind === 'diamond') S('path', { d: 'M6 0.8 L11.2 6 L6 11.2 L0.8 6 Z', class: cls }, s);
  else if (kind === 'square') S('rect', { x: 1.5, y: 1.5, width: 9, height: 9, rx: 1.5, class: cls }, s);
  else S('rect', { x: 0, y: 1, width: 12, height: 10, rx: 2, class: cls }, s);
  return s;
}
function legend(host, items) {
  host.replaceChildren();
  items.forEach(function (it) { var sp = H('span', { class: 'it' }, host); swatch(sp, it.kind, it.cls); H('span', null, sp, it.label); });
}
// ---------- шапка ----------
function renderKpis() {
  var host = $('#kpis'), f = D.factoid_1k, h = D.hero_apartment;
  var utilUp = f.util_median && f.price_median ? f.util_median / f.price_median : null;
  var tiles = [
    { v: rub(f.price_median), l: 'медианная цена 1-комнатной в объявлении', s: '+ коммуналка ' + rub(f.util_median) + ' = ' + rub(f.price_median + f.util_median) + ' на деле (+' + pct(utilUp) + ')' },
    { v: pct(K.no_sum_all), l: 'объявлений не называют сумму коммуналки', s: 'Telegram ' + pct(K.no_sum_by_source['Telegram']) + ', Циан ' + pct(K.no_sum_by_source['Циан']) },
    { v: '50 из 100', l: 'баллов получает место, выбранное единственным кроме метро, если до него ≤ 5 мин', s: 'столько же весит метро: место не отсеивает квартиры, а поднимает их в списке (глава 2)' }
  ];
  tiles.forEach(function (t) {
    var d = H('div', { class: 'tile' }, host);
    H('div', { class: 'lab' }, d, t.l); H('div', { class: 'val' }, d, t.v); H('div', { class: 'sub' }, d, t.s);
  });
  var near = h.metro ? 'метро ' + h.metro : 'метро';
  $('#hero-story').textContent = 'Самая удачная находка в наборе — ' + (h.fmt === 'студия' ? 'студия' : h.fmt) + ', ' +
    num(h.area, 0) + ' м² (' + h.source + '): ' + num(h.metro_min) + ' мин до ' + near + ' и ' + num(h.driver_min) +
    ' мин до места «' + h.driver_label + '» — балл близости ' + h.score + ' из 100, полная цена ≈ ' + rub(h.price_est) +
    ' в месяц. Ни одно из этих двух удобств само объявление не называет — их видно только по карте.';
  var views = 10, minPer = 5, saved = views * minPer;
  $('#time-note').textContent = 'Чтобы вручную уточнить коммуналку и посчитать минуты до нужных мест у ' + views +
    ' объявлений, нужно примерно по ' + minPer + ' минут на каждое — около ' + saved + ' минут на один подбор квартиры. ' +
    'Дальше — что ещё в этих данных нашлось.';
}

// ---------- 2. точечный график ----------
var BASIS = [
  { k: 'известно', cls: 'f1', st: 'st1', c: '--s1', shape: 'circle', label: 'полная цена известна' },
  { k: 'оценка коммуналки', cls: 'f2', st: 'st2', c: '--s2', shape: 'diamond', label: 'коммуналка не указана — оценка' },
  { k: 'оценка: счётчики', cls: 'f3', st: 'st3', c: '--s3', shape: 'square', label: '«счётчики сверху» — оценка' }
];
function marker(g, shape, x, y, r, cls) {
  if (shape === 'circle') return S('circle', { cx: x, cy: y, r: r, class: cls + ' ring' }, g);
  if (shape === 'diamond') return S('path', { d: 'M' + x + ' ' + (y - r - 1) + 'L' + (x + r + 1) + ' ' + y + 'L' + x + ' ' + (y + r + 1) + 'L' + (x - r - 1) + ' ' + y + 'Z', class: cls + ' ring' }, g);
  return S('rect', { x: x - r, y: y - r, width: 2 * r, height: 2 * r, rx: 1.5, class: cls + ' ring' }, g);
}
var SCATTER_LIMIT = 40;   // полный список виден только мелко: показываем худшие случаи крупно, остальное — в таблице
function renderScatter() {
  var host = $('#scatter-chart'); host.replaceChildren();
  var all = D.price_vs_full.points.slice().sort(function (a, b) { return (b.est - b.price) - (a.est - a.price); });
  var P = all.slice(0, SCATTER_LIMIT);
  var W = host.clientWidth, narrow = W < 560;
  var rowH = narrow ? 30 : 24, m = { l: 6, r: 6, t: 24, b: 6 };
  var Hh = m.t + P.length * rowH + m.b;
  var lo = Math.min.apply(null, all.map(function (p) { return p.price; })) * 0.92;
  var hi = Math.max.apply(null, all.map(function (p) { return p.hi; })) * 1.05;
  var L0 = Math.log(lo), L1 = Math.log(hi), pw = W - m.l - m.r;
  function sx(v) { return m.l + (Math.log(v) - L0) / (L1 - L0) * pw; }
  var svg = S('svg', { viewBox: '0 0 ' + W + ' ' + Hh, height: Hh }, host);
  host.setAttribute('aria-label', 'Дамбелл-диаграмма: цена из объявления и полная цена в месяц, ' + P.length +
    ' объявлений с самой большой доплатой из ' + all.length + ' (Циан)');
  var ticks = (narrow ? [30000, 50000, 100000, 150000] : [25000, 30000, 40000, 50000, 70000, 100000, 150000]).filter(function (t) { return t >= lo && t <= hi; });
  ticks.forEach(function (t) {
    S('line', { x1: sx(t), x2: sx(t), y1: 16, y2: Hh, class: 'gl' }, svg);
    T(svg, sx(t), 13, num(t / 1000) + ' тыс. ₽', { class: 'ax', 'text-anchor': 'middle' });
  });
  P.forEach(function (p, i) {
    var y = m.t + i * rowH + rowH / 2;
    var b = BASIS.filter(function (x) { return x.k === p.basis; })[0] || BASIS[0];
    var x0 = sx(p.price), x1 = sx(p.est);
    var row = S('g', { tabindex: 0, role: 'listitem' }, svg);
    if (p.hi > p.lo) S('line', { x1: sx(p.lo), x2: sx(p.hi), y1: y, y2: y, class: b.st, 'stroke-width': 2, opacity: 0.4 }, row);
    S('line', { x1: x0, x2: x1, y1: y, y2: y, stroke: 'var(--ink2)', 'stroke-width': 1.6 }, row);
    S('circle', { cx: x0, cy: y, r: 3.5, fill: 'var(--muted)' }, row);
    marker(row, b.shape, x1, y, 5, b.cls);
    S('rect', { x: 0, y: y - rowH / 2, width: W, height: rowH, class: 'row-hit' }, row);
    bindTip(row, function () {
      var extra = p.est - p.price;
      var desc = p.fmt + ', ' + num(p.area, 0) + ' м²' + (p.floor ? ', ' + p.floor + '/' + (p.floors || '?') + ' эт.' : '') +
        (p.metro ? ', у метро ' + p.metro : '') + (p.metro_min != null ? ' (' + num(p.metro_min) + ' мин)' : '');
      var rows = [{ v: rub(p.price), l: 'цена в объявлении' },
        { v: (p.hi > p.lo ? '≈ ' : '') + rub(p.est), l: 'полная цена в месяц', c: b.c }];
      if (p.hi > p.lo) rows.push({ v: rub(p.lo) + ' – ' + rub(p.hi), l: 'разброс оценки' });
      rows.push({ v: '+' + rub(extra) + ' (' + pct(extra / p.price, 1) + ')', l: 'доплата' });
      rows.push({ v: p.util_status, l: 'коммуналка' });
      if (p.commission) rows.push({ v: rub(p.commission), l: 'комиссия' });
      return { title: desc, rows: rows };
    });
  });
}
function scatterStatic() {
  var P = D.price_vs_full.points;
  $('#scatter-sub').textContent = 'Серая точка — цена в объявлении, цветная — сколько на самом деле придётся платить в месяц; ' +
    'показаны ' + SCATTER_LIMIT + ' объявлений с самой большой доплатой из ' + num(P.length) + ' (Циан), остальные — в таблице ниже. ' +
    'Медианная доплата по всем — ' + rub(K.median_extra) + ' в месяц; у ' + pct(K.share_over10) + ' объявлений полная цена выше указанной больше чем на 10%.';
  legend($('#scatter-legend'), BASIS.map(function (b) {
    return { kind: b.shape, cls: b.cls, label: b.label + ' · ' + P.filter(function (p) { return p.basis === b.k; }).length };
  }));
  $('#top10-callout').textContent = 'Честно о порядке: если отсортировать те же объявления по полной цене вместо цены в объявлении, ' +
    'из первой десятки выпадают только ' + K.top10_dropped + ' из 10, тау Кендалла ' + num(K.tau, 2) + '. Порядок почти не меняется — ' +
    'боль не в том, кто дешевле, а в сюрпризе доплаты, которую узнаёшь только на просмотре.';
  function med(a) { a = a.slice().sort(function (x, y) { return x - y; }); var n = a.length; return n ? (n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2) : NaN; }
  var groups = BASIS.map(function (b) { return { label: b.label, P: P.filter(function (p) { return p.basis === b.k; }) }; });
  groups.push({ label: 'все', P: P });
  table($('#scatter-table'), ['Основа полной цены', 'Объявлений', 'Медиана цены', 'Медиана полной', 'Медиана доплаты', 'Доплата > 10%'],
    groups.map(function (g) {
      return [g.label, num(g.P.length), rub(med(g.P.map(function (p) { return p.price; }))), rub(med(g.P.map(function (p) { return p.est; }))),
        rub(med(g.P.map(function (p) { return p.est - p.price; }))), pct(g.P.filter(function (p) { return (p.est - p.price) / p.price > 0.1; }).length / g.P.length)];
    }), [1, 2, 3, 4, 5]);
  var full = P.slice().sort(function (a, b) { return (b.est - b.price) - (a.est - a.price); });
  table($('#scatter-table-full'), ['Формат', 'м²', 'Цена', 'Полная цена', 'Доплата', 'Доплата, %', 'Коммуналка'],
    full.map(function (p) {
      return [p.fmt, num(p.area, 0), rub(p.price), rub(p.est), rub(p.est - p.price), pct((p.est - p.price) / p.price, 1), p.util_status];
    }), [1, 2, 3, 4, 5]);
}

// ---------- 3. драйверы интереса (дамбелл: 0 -> медиана баллов у квартир рядом) ----------
var iceMode = 'all';
function drvRows() {
  return D.drivers_score.filter(function (r) {
    return !/^каток/.test(r.label) || (iceMode === 'named') === /с названием/.test(r.label);
  }).slice().sort(function (a, b) { return b.median_points - a.median_points; });
}
function renderDrivers() {
  var host = $('#drv-chart'); host.replaceChildren();
  var rows = drvRows(), W = host.clientWidth, narrow = W < 520;
  var labW = narrow ? 0 : Math.min(170, W * 0.34), rowH = narrow ? 46 : 32, top = 6, bottom = 20;
  var Hh = top + rows.length * rowH + bottom;
  var x0 = labW + 4, pw = W - x0 - 46, M = 50;
  function sx(v) { return x0 + v / M * pw; }
  var svg = S('svg', { viewBox: '0 0 ' + W + ' ' + Hh, height: Hh }, host);
  host.setAttribute('aria-label', 'Дамбелл: баллы близости по драйверам интереса, шкала 0–50');
  [0, 10, 20, 30, 40, 50].forEach(function (t) {
    S('line', { x1: sx(t), x2: sx(t), y1: top, y2: Hh - bottom, class: t === 0 ? 'bl' : 'gl' }, svg);
    T(svg, sx(t), Hh - 6, num(t), { class: 'ax', 'text-anchor': 'middle' });
  });
  rows.forEach(function (r, i) {
    var y = top + i * rowH, by = narrow ? y + 24 : y + rowH / 2 - 4;
    if (narrow) T(svg, 0, y + 14, r.label, { class: 'lab' });
    else T(svg, labW, by + 4, r.label, { class: 'lab', 'text-anchor': 'end' });
    var g = S('g', { tabindex: 0, role: 'listitem' }, svg);
    S('line', { x1: sx(0), x2: sx(r.median_points), y1: by, y2: by, stroke: 'var(--ink2)', 'stroke-width': 1.5 }, g);
    S('circle', { cx: sx(0), cy: by, r: 3.5, fill: 'var(--muted)' }, g);
    S('circle', { cx: sx(r.median_points), cy: by, r: 5, class: 'f1 ring' }, g);
    T(g, sx(r.median_points) + 9, by + 4, num(r.median_points, 0), { class: 'valt' });
    S('rect', { x: 0, y: y, width: W, height: rowH - 2, class: 'row-hit' }, g);
    bindTip(g, function () {
      return {
        title: r.label, rows: [
          { v: num(r.median_points, 1), l: 'медиана баллов у квартир, которые рядом (≤ 15 мин)' },
          { v: pct(r.share_near15 / 100, 0), l: 'квартир с точным адресом рядом с этим местом' }
        ]
      };
    });
  });
  table($('#drv-table'), ['Место', 'Рядом (≤ 15 мин)', 'Медиана баллов у тех, что рядом'],
    rows.map(function (r) { return [r.label, pct(r.share_near15 / 100, 0), num(r.median_points, 1)]; }), [1, 2]);
}
function drvStatic() {
  $('#drv-sub').textContent = 'В продукте драйверы — веса для порядка выдачи, а не фильтры: место, выбранное единственным кроме ' +
    'метро, весит 0,5 (как метро) и поднимает квартиру в списке, не убирая остальные. Два конца линии — 0 (в объявлении ' +
    'это место никак не названо) и медиана баллов среди квартир, которые и правда рядом.';
}
function bindIce() {
  $('#ice-seg').addEventListener('click', function (ev) {
    var b = ev.target.closest('button'); if (!b) return;
    iceMode = b.dataset.v;
    $('#ice-seg').querySelectorAll('button').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
    renderDrivers();
  });
  var ice = D.drivers_score.filter(function (r) { return /^каток/.test(r.label); });
  $('#ice-seg').querySelectorAll('button').forEach(function (b) {
    var r = ice.filter(function (x) { return (b.dataset.v === 'named') === /с названием/.test(x.label); })[0];
    if (r) b.textContent += ' · ' + pct(r.share_near15 / 100, 0);
  });
}

// ---------- 5. поток ----------
var FLOW = [
  { k: 'метро + фитнес', cls: 'st1', c: '--s1', label: 'метро + фитнес' },
  { k: 'метро + садик + поликлиника', cls: 'st2', c: '--s2', label: 'метро + садик + поликлиника' },
  { k: 'метро + каток с названием', cls: 'st3', c: '--s3', label: 'метро + каток с названием' }
];
var MON = ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек'];
function monLabel(s, full) { var p = s.split('-'); return MON[+p[1] - 1] + (full ? ' ' + p[0] : ''); }
function flowStatic() {
  if (!$('#s-flow')) return;    // «Сколько подходящих объявлений в месяц» скрыта до экспериментального прогона
  var F = D.flow, sm = F.summary, keys = Object.keys(sm);
  var med = sm[keys.find(function (k) { return /медиана/.test(k); })] || {};
  var mx = sm[keys.find(function (k) { return /больше/.test(k); })] || {};
  $('#flow-sub').textContent = 'Telegram-канал по всему городу, долгая аренда квартир. В медианный месяц — ' + num(med['все']) +
    ' объявлений, из них с точкой на доме ' + num(med['точка по дому']) + '. Под узкий профиль подходит единицы: метро + фитнес — ' +
    num(med['метро + фитнес']) + ', метро + садик + поликлиника — ' + num(med['метро + садик + поликлиника']) +
    ', метро + каток с названием — ' + num(med['метро + каток с названием']) + ' (максимум ' + num(mx['метро + каток с названием']) + ' за месяц).';
  var cian = keys.find(function (k) { return /Циан.*30 дней/.test(k); });
  $('#flow-note').textContent = 'Профили: каждое место ≤ 15 минут пешком, только объявления с точкой на доме. ' +
    (cian ? 'Циан — не поток, а один срез в одном районе: за 30 дней до сбора там опубликовано ' + num(sm[cian]['все']) +
      ' студий и 1-к. Поэтому для экономики поток канала умножается на масштаб города (ползунок в калькуляторе).' : '');
  legend($('#flow-legend'), [{ kind: 'rect', cls: 'fgb', label: 'все объявления: с точкой на доме' }, { kind: 'rect', cls: 'fgt', label: 'остальные' }]
    .concat(FLOW.map(function (f) { return { kind: 'line', cls: f.cls, label: f.label }; })));
  var head = ['Месяц', 'Все', 'С точкой на доме'].concat(FLOW.map(function (f) { return f.label; }));
  table($('#flow-table'), head, F.months.map(function (mo, i) {
    return [monLabel(mo, true), num(F.by_month['все'][i]), num(F.by_month['точка по дому'][i])].concat(FLOW.map(function (f) { return num(F.by_month[f.k][i]); }));
  }), [1, 2, 3, 4, 5]);
}
function renderFlow() {
  var host = $('#flow-chart'); if (!host) return;
  host.replaceChildren();
  var F = D.flow, n = F.months.length, W = host.clientWidth, narrow = W < 560;
  var Hh = narrow ? 260 : 300, m = { l: 34, r: 8, t: 10, b: 32 };
  var pw = W - m.l - m.r, ph = Hh - m.t - m.b, all = F.by_month['все'], prec = F.by_month['точка по дому'];
  var ymax = Math.ceil(Math.max.apply(null, all) / 10) * 10;
  var band = pw / n, bwid = Math.min(24, band * 0.68);
  function cx(i) { return m.l + band * (i + 0.5); }
  function sy(v) { return m.t + ph - v / ymax * ph; }
  var svg = S('svg', { viewBox: '0 0 ' + W + ' ' + Hh, height: Hh }, host);
  host.setAttribute('aria-label', 'Объявления Telegram по месяцам и подходящие под три профиля');
  for (var t = 0; t <= ymax; t += 10) {
    S('line', { x1: m.l, x2: m.l + pw, y1: sy(t), y2: sy(t), class: t ? 'gl' : 'bl' }, svg);
    T(svg, m.l - 6, sy(t) + 4, num(t), { class: 'ax', 'text-anchor': 'end' });
  }
  var every = narrow ? 6 : 3;
  F.months.forEach(function (mo, i) {
    if (i % every === 0) T(svg, cx(i), Hh - 12, monLabel(mo, false) + ' ' + mo.slice(2, 4), { class: 'ax', 'text-anchor': 'middle' });
    var x = cx(i) - bwid / 2, yb = sy(prec[i]), yt = sy(all[i]);
    S('path', { d: 'M' + x + ',' + sy(0) + 'V' + yb + 'H' + (x + bwid) + 'V' + sy(0) + 'Z', class: 'fgb' }, svg);
    if (all[i] > prec[i] && yb - yt > 2.5) {
      var h = yb - 2 - yt, r = Math.min(4, h / 2, bwid / 2);
      S('path', { d: 'M' + x + ',' + (yb - 2) + 'V' + (yt + r) + 'a' + r + ',' + r + ' 0 0 1 ' + r + ',' + (-r) + 'H' + (x + bwid - r) +
        'a' + r + ',' + r + ' 0 0 1 ' + r + ',' + r + 'V' + (yb - 2) + 'Z', class: 'fgt' }, svg);
    }
  });
  FLOW.forEach(function (f) {
    var d = F.by_month[f.k].map(function (v, i) { return (i ? 'L' : 'M') + cx(i) + ',' + sy(v); }).join('');
    S('path', { d: d, fill: 'none', class: f.cls, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
    var last = F.by_month[f.k][n - 1];
    S('circle', { cx: cx(n - 1), cy: sy(last), r: 4, class: f.cls.replace('st', 'f') + ' ring' }, svg);
  });
  var cross = S('line', { y1: m.t, y2: m.t + ph, stroke: 'var(--muted)', 'stroke-width': 1, visibility: 'hidden' }, svg);
  var hit = S('rect', { x: m.l, y: m.t, width: pw, height: ph, fill: 'transparent' }, svg);
  function move(ev) {
    var r = svg.getBoundingClientRect(), k = W / r.width, mx = (ev.clientX - r.left) * k;
    var i = Math.max(0, Math.min(n - 1, Math.floor((mx - m.l) / band)));
    cross.setAttribute('x1', cx(i)); cross.setAttribute('x2', cx(i)); cross.setAttribute('visibility', 'visible');
    tipShow(ev, monLabel(F.months[i], true), [{ v: num(all[i]), l: 'все объявления' }, { v: num(prec[i]), l: 'с точкой на доме' }]
      .concat(FLOW.map(function (f) { return { v: num(F.by_month[f.k][i]), l: f.label, c: f.c }; })));
  }
  hit.addEventListener('pointermove', move);
  hit.addEventListener('pointerdown', function (ev) { move(ev); ev.stopPropagation(); });
  hit.addEventListener('pointerleave', function (ev) { if (ev.pointerType !== 'touch') { tipHide(); cross.setAttribute('visibility', 'hidden'); } });
}

// ---------- 6. разбор полей ----------
function renderExtract() {
  var host = $('#ext'); host.replaceChildren();
  if (!V.available) {
    $('#ext-sub').textContent = 'Метрики разбора ещё не собраны: ждёт прогона на размеченном эталоне.';
    return;
  }
  var w = V.weights || {}, layers = V.layers || {}, lk = Object.keys(layers);
  $('#ext-sub').textContent = 'Эталон: ' + num(V.n) + ' постов Telegram, размеченных вручную (' +
    lk.map(function (k) { return k + ' — ' + layers[k]; }).join(', ') + '); метрики взвешены по долям пула (' +
    lk.map(function (k) { return k + ' ' + pct(w[k]); }).join(', ') + '). Точность — доля объявлений, где поле совпало с эталоном, ' +
    'включая случаи «в тексте этого нет». Полнота — сколько найдено там, где поле есть. Потолок — в какой доле постов поле вообще есть: ' +
    'больше этого не извлечёт ни одна модель.';
  legend($('#ext-legend'), [{ kind: 'rect', cls: 'f1', label: 'регулярки' }, { kind: 'rect', cls: 'f2', label: 'LLM' },
    { kind: 'rect', cls: 'f3', label: 'гибрид' }, { kind: 'rect', cls: 'fgb', label: 'потолок: поле есть в посте' }]);
  ['Поле', 'Регулярки', 'LLM / гибрид', 'Потолок'].forEach(function (h) { H('div', { class: 'h' }, host, h); });
  var waitTxt = 'ждёт прогона';
  function meterCell(lab, m, color) {
    var c = H('div', { 'data-l': lab }, host);
    if (!m || m.acc == null) { H('div', { class: 'wait' }, c, '—'); return c; }
    H('span', { class: 'mv' }, c, pct(m.acc, 1));
    var mt = H('div', { class: 'meter' }, c), sp = H('span', null, mt); sp.style.width = (m.acc * 100) + '%'; sp.style.background = 'var(' + color + ')';
    if (m.rec != null) H('div', { class: 'ms' }, c, 'полнота ' + pct(m.rec, 1));
    return c;
  }
  V.fields.forEach(function (f) {
    H('div', { class: 'f' }, host, f.label);
    meterCell('Регулярки', f.regex, '--s1');
    var c = H('div', { 'data-l': 'LLM / гибрид' }, host);
    var any = false;
    [['LLM', f.llm, '--s2'], ['гибрид', f.hybrid, '--s3']].forEach(function (x) {
      if (!x[1] || x[1].acc == null) return; any = true;
      var d = H('div', null, c); H('span', { class: 'mv' }, d, x[0] + ' ' + pct(x[1].acc, 1));
      var mt = H('div', { class: 'meter' }, d), sp = H('span', null, mt); sp.style.width = (x[1].acc * 100) + '%'; sp.style.background = 'var(' + x[2] + ')';
      if (x[1].rec != null) H('div', { class: 'ms' }, d, 'полнота ' + pct(x[1].rec, 1));
    });
    if (!any) H('div', { class: 'wait' }, c, waitTxt);
    var cc = H('div', { 'data-l': 'Потолок' }, host);
    if (f.ceiling == null) H('div', { class: 'wait' }, cc, '—');
    else {
      H('span', { class: 'mv' }, cc, pct(f.ceiling, f.ceiling < 0.1 ? 1 : 0));
      var mt = H('div', { class: 'meter' }, cc), sp = H('span', null, mt); sp.style.width = (f.ceiling * 100) + '%'; sp.style.background = 'var(--g-bot)';
      H('div', { class: 'ms' }, cc, 'постов, где поле есть');
    }
  });
  var bits = [];
  if (V.routing_share != null) {
    var rl = V.routing_layers || {};
    var rls = Object.keys(rl).map(function (k) { return k + ' ' + rl[k].llm + ' из ' + rl[k].n; }).join(', ');
    bits.push(V.routing_share >= 0.999
      ? 'Гибрид работает по полям: по каждой группе полей регулярки пусты у большинства постов, а один вызов разбирает пост целиком, поэтому в модель уходят все посты — ' + pct(V.routing_share) + ' (' + rls + '); экономии на доле постов нет.'
      : 'Архитектура — гибрид: регулярки разбирают шаблонные посты, в LLM уходят свободный текст и посты, не прошедшие проверку, — ' + pct(V.routing_share) + ' пула (' + rls + ').');
  }
  if (!V.has_llm && !V.has_hybrid) bits.push('Метрики LLM и гибрида появятся после прогона модели' + (V.llm_posts ? ' на ' + V.llm_posts + ' постах эталона, которые гибрид отправляет в LLM' : '') + '.');
  if (V.llm_run && V.llm_run.model) bits.push('Прогон: ' + V.llm_run.model + ', вызовов ' + V.llm_run.calls + ', невалидных карточек ' + V.llm_run.invalid_cards + '.');
  bits.push('Потолок для даты заезда и станции не считался («—»).');
  if (V.full_price_share != null) bits.push('Полную цену по самому посту можно посчитать лишь у ' + pct(V.full_price_share, 1) + ' объявлений эталона — остальное оценка.');
  bits.push('Проверка эталона: ' + V.second_annotator + '.');
  $('#ext-note').textContent = bits.join(' ');
}

// ---------- 7. калькулятор ----------
var compute = window.RentCalc.compile(E);
var PARAM = {}; E.params.forEach(function (p) { PARAM[p.key] = p; });
var FORM = {}; E.formulas.forEach(function (f) { FORM[f.key] = f; });
var SLIDERS = E.params.filter(function (p) { return p.slider; });
var GROUP_NAMES = { 1: 'Воронка в месяц', 2: 'Тарифы и доход с платящего', 3: 'Модель и курс', 4: 'Сообщения в диалоге',
  5: 'Разбор объявлений (гибрид)', 6: 'Постоянные затраты', 7: 'Доли от выручки', 8: 'Привлечение', 9: 'Разработка (разово)' };
var GROUP_ORDER = [1, 4, 2, 3, 5, 6, 7, 8, 9];
function baseState() { var s = {}; E.params.forEach(function (p) { s[p.key] = p.base; }); return s; }
var st = baseState();
var BASE_OUT = compute(baseState());
function clampP(p, v) { return Math.min(p.max, Math.max(p.min, v)); }
function fixShares(s, changed) { // доля «месяц» = 1 − день − неделя не может быть меньше нуля
  if (s.sh_day + s.sh_week > 1 + 1e-12) {
    if (changed === 'sh_week') s.sh_day = Math.max(0, 1 - s.sh_week); else s.sh_week = Math.max(0, 1 - s.sh_day);
  }
  return s;
}
function evalWith(base, key, val) { var s = Object.assign({}, base); s[key] = val; fixShares(s, key); return compute(s); }
function trimZ(s) { return s.replace(/,(\d*?)0+$/, function (m, a) { return a ? ',' + a : ''; }); }
function fmtP(p, v, extra) {
  var d = Math.min(decimals(p.step), p.unit === '$' || p.unit === 'доля' ? 4 : 1) + (extra || 0);
  if (p.unit === 'доля') return trimZ(num(v * 100, Math.max(0, d - 2))) + '%';
  if (p.unit === '₽') return trimZ(num(v, d)) + ' ₽';
  if (p.unit === '$') return '$' + num(v, Math.max(d, 2));
  if (p.unit === '×') return '×' + trimZ(num(v, d));
  return trimZ(num(v, d)) + ' ' + p.unit;
}
function short(label) { return label.replace(/\s*\(.*?\)\s*/g, ' ').trim(); }
var inputs = {};
function buildParams() {
  var host = $('#params'); host.replaceChildren();
  var byG = {};
  SLIDERS.forEach(function (p) { var g = parseInt(p.group, 10) || 0; (byG[g] = byG[g] || []).push(p); });
  GROUP_ORDER.concat(Object.keys(byG).map(Number).filter(function (g) { return GROUP_ORDER.indexOf(g) < 0; })).forEach(function (g, gi) {
    var list = byG[g]; if (!list) return;
    var det = H('details', { class: 'grp' }, host); if (gi < 3) det.open = true;
    H('summary', null, det, GROUP_NAMES[g] || list[0].group.replace(/^\d+\.\s*/, ''));
    var body = H('div', { class: 'grp-body' }, det);
    if (g === 3) H('div', { class: 'derived' }, body, 'Модель, цены токенов и курс доллара выбираются над плитками. ' +
      'Цены — каталог OpenRouter, ' + (E.models[0].catalog || '').replace(/^каталог\s*/, '') + '; комиссия OpenRouter при пополнении учтена в формуле.');
    list.forEach(function (p) {
      if (g === 3 && (p.key === 'p_in' || p.key === 'p_out' || p.key === 'usd')) return;
      var row = H('div', { class: 'prm' }, body), top = H('div', { class: 'prm-top' }, row);
      var lab = H('label', { for: 'r-' + p.key }, top, p.label);
      var info = H('button', { class: 'info', type: 'button', 'aria-label': 'Откуда значение', 'aria-expanded': 'false' }, lab, 'ⓘ');
      var out = H('output', { id: 'o-' + p.key, for: 'r-' + p.key }, top);
      var r = H('input', { type: 'range', id: 'r-' + p.key, min: p.min, max: p.max, step: p.step, value: st[p.key] }, row);
      var src = H('div', { class: 'prm-src' }, row, 'Источник: ' + p.source + ' · диапазон ' + fmtP(p, p.min) + ' – ' + fmtP(p, p.max));
      src.hidden = true;
      info.addEventListener('click', function (ev) { ev.preventDefault(); src.hidden = !src.hidden; info.setAttribute('aria-expanded', String(!src.hidden)); });
      r.addEventListener('input', function () { st[p.key] = +r.value; fixShares(st, p.key); schedule(); });
      inputs[p.key] = { r: r, out: out, p: p };
    });
    if (g === 2) H('div', { class: 'derived', id: 'd-shmonth' }, body);
  });
}
var modelBtns = [];
function buildModelBox(body) {
  var seg = H('div', { class: 'seg', role: 'group', 'aria-label': 'Модель' }, body);
  E.models.forEach(function (m) {
    var b = H('button', { type: 'button', title: m.note + ' · ' + m.slug }, seg, m.name);
    b.addEventListener('click', function () { st.p_in = m.p_in; st.p_out = m.p_out; schedule(); });
    modelBtns.push({ b: b, m: m });
  });
  var own = H('button', { type: 'button', disabled: '' }, seg, 'свои цены'); modelBtns.push({ b: own, m: null });
  var box = H('div', { class: 'num-in' }, body);
  [['p_in', 'свои цены: вход, $ за 1 млн'], ['p_out', 'выход, $ за 1 млн'], ['usd', 'курс доллара, ₽']].forEach(function (x) {
    var p = PARAM[x[0]], l = H('label', null, box, x[1]);
    var inp = H('input', { type: 'number', min: p.min, max: p.max, step: 'any', value: st[p.key], id: 'n-' + p.key, inputmode: 'decimal' }, l);
    inp.addEventListener('change', function () {
      var v = parseFloat(String(inp.value).replace(',', '.'));
      st[p.key] = isFinite(v) ? clampP(p, v) : p.base; schedule();
    });
    inputs[p.key] = { num: inp, p: p };
  });
}
function buildScenarios() {
  var seg = $('#scn-seg');
  ['пессимист', 'база', 'оптимист'].forEach(function (name) {
    if (!E.presets[name]) return;
    var b = H('button', { type: 'button', 'data-s': name, 'aria-pressed': 'false' }, seg, name);
    b.addEventListener('click', function () {
      var keep = { p_in: st.p_in, p_out: st.p_out, usd: st.usd };
      st = Object.assign(baseState(), E.presets[name], keep); fixShares(st); schedule();
    });
  });
  $('#reset-btn').addEventListener('click', function () { st = baseState(); schedule(); });
}
function scenarioOf() {
  var names = ['пессимист', 'база', 'оптимист'];
  for (var i = 0; i < names.length; i++) {
    var s = Object.assign(baseState(), E.presets[names[i]] || {});
    var ok = E.params.every(function (p) { return ['p_in', 'p_out', 'usd'].indexOf(p.key) >= 0 || Math.abs(s[p.key] - st[p.key]) < 1e-9; });
    if (ok) return names[i];
  }
  return null;
}
var TILES = [
  { k: 'payers', l: 'Платящих в месяц', f: function (v) { return num(v, 1); } },
  { k: 'rev_payer', l: 'Выручка на платящего', f: rub },
  { k: 'contrib', l: 'Вклад на платящего после привлечения', f: rub },
  { k: 'ltv_cac', l: 'LTV / CAC', f: function (v) { return num(v, 2) + '×'; }, s: '≥ 3 — здоровая экономика' },
  { k: 'profit_month', l: 'Прибыль в месяц', f: rub, s: 'до разработки' },
  { k: 'be_month', l: 'Безубыточность, платящих в месяц', f: function (v) { return isFinite(v) ? num(v) : 'не окупается'; } },
  { k: 'payback', l: 'Окупаемость разработки', f: function (v) { return isFinite(v) ? num(v, 1) + ' мес' : 'не окупается'; } }
];
var tileEls = {};
function buildTiles() {
  var host = $('#otiles');
  TILES.forEach(function (t) {
    var d = H('div', { class: 'tile' }, host);
    H('div', { class: 'lab' }, d, t.l); tileEls[t.k] = { v: H('div', { class: 'val' }, d), s: H('div', { class: 'sub' }, d) }; tileEls[t.k].t = t;
  });
}
var pending = false;
function schedule() { if (pending) return; pending = true; requestAnimationFrame(function () { pending = false; update(); }); }
var OUT = null;
function update() {
  OUT = compute(st);
  Object.keys(inputs).forEach(function (k) {
    var x = inputs[k], p = x.p;
    if (x.r) {
      if (+x.r.value !== st[k]) x.r.value = st[k];
      x.out.replaceChildren(); x.out.append(fmtP(p, st[k]));
      if (Math.abs(st[k] - p.base) > 1e-12) { var sm = H('small', null, x.out, 'база ' + fmtP(p, p.base)); }
      x.r.setAttribute('aria-valuetext', fmtP(p, st[k]));
    }
    if (x.num && document.activeElement !== x.num) x.num.value = +st[k].toFixed(4);
  });
  var dm = $('#d-shmonth'); if (dm) dm.textContent = 'Доля покупок «месяц» (остаток): ' + pct(OUT.sh_month) + ' · средний чек ' + rub(OUT.check) + ' · дней подписки на покупку ' + num(OUT.sub_days, 1);
  modelBtns.forEach(function (x) {
    var on = x.m ? (Math.abs(x.m.p_in - st.p_in) < 1e-9 && Math.abs(x.m.p_out - st.p_out) < 1e-9) : !modelBtns.some(function (y) { return y.m && Math.abs(y.m.p_in - st.p_in) < 1e-9 && Math.abs(y.m.p_out - st.p_out) < 1e-9; });
    x.b.setAttribute('aria-pressed', String(on));
  });
  var sc = scenarioOf();
  $('#scn-seg').querySelectorAll('button').forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.s === sc)); });
  TILES.forEach(function (t) {
    var e = tileEls[t.k]; e.v.textContent = t.f(OUT[t.k]); e.v.classList.toggle('txt', !isFinite(OUT[t.k]));
    e.s.textContent = (t.s ? t.s + ' · ' : '') + 'в базе: ' + t.f(BASE_OUT[t.k]);
  });
  var ss = $('#sticky-sum'); ss.replaceChildren();
  [['Прибыль', rub(OUT.profit_month) + '/мес'], ['безубыточность', isFinite(OUT.be_month) ? num(OUT.be_month) : 'нет'],
    ['окупаемость', isFinite(OUT.payback) ? num(OUT.payback, 1) + ' мес' : 'не окупается']].forEach(function (x) {
    var sp = H('span', null, ss, x[0] + ': '); H('b', null, sp, x[1]);
  });
  renderBeBadge(); renderBdTable(); TOR = tornadoData(); renderNeed(); renderFormulaTable();
}
// плашка окупаемости: зелёная, когда прибыль в месяц больше нуля, иначе красная
function renderBeBadge() {
  var host = $('#be-badge'); host.replaceChildren();
  var good = OUT.profit_month > 0;
  host.className = 'be-badge ' + (good ? 'good' : 'bad');
  var p1 = H('span', null, host, 'Прибыль в месяц: '); H('b', null, p1, rub(OUT.profit_month));
  var p2 = H('span', null, host, good ? 'окупает разработку за: ' : 'статус: ');
  H('b', null, p2, good ? (isFinite(OUT.payback) ? num(OUT.payback, 1) + ' мес' : 'не окупает разработку') : 'не окупается');
  var p3 = H('span', null, host);
  if (isFinite(OUT.be_month)) {
    var gap = OUT.be_month - OUT.payers;
    p3.append('запас: '); H('b', null, p3, num(OUT.payers, 1) + ' из ' + num(OUT.be_month) + ' платящих для безубыточности');
    if (gap > 0.05) p3.append(' (не хватает ' + num(gap, 1) + ')');
  } else { p3.append('запас: '); H('b', null, p3, 'безубыточность недостижима при текущем вкладе на платящего'); }
}
// иерархическая разбивка доходов и расходов в месяц — вместо кривой, которая «прыгала» при каждом ползунке
function renderBdTable() {
  var host = $('#bd-table'); host.replaceChildren();
  var modelDialog = OUT.model_paid * OUT.payers, modelFree = OUT.free_month, modelParse = OUT.extract;
  var modelTotal = modelDialog + modelFree + modelParse;
  var supportTotal = OUT.support_h * OUT.rate, fixedTotal = OUT.hosting + supportTotal;
  var acqAmt = OUT.revenue_month * OUT.acq, taxAmt = OUT.revenue_month * OUT.tax, adminAmt = OUT.revenue_month * OUT.admin_share;
  var shareTotal = acqAmt + taxAmt + adminAmt, cacTotal = OUT.payers * OUT.cac;
  var tb = H('tbody', null, host);
  function grp(label) { H('td', { colspan: 2 }, H('tr', { class: 'grp' }, tb), label); }
  function row(label, val, cls) { var tr = H('tr', { class: cls || 'sub' }, tb); H('td', null, tr, label); H('td', { class: 'n' }, tr, val); }
  grp('Доходы');
  row('Платящих в месяц', num(OUT.payers, 1) + ' чел.');
  row('Выручка на платящего', rub(OUT.rev_payer));
  row('Выручка в месяц', rub(OUT.revenue_month), 'total');
  grp('Расходы — модель (ИИ)');
  row('Диалог с платящими', rub(modelDialog));
  row('Бесплатные сообщения', rub(modelFree));
  row('Разбор объявлений (фон)', rub(modelParse));
  row('Итого модель', rub(modelTotal) + ' (' + pct(OUT.revenue_month ? modelTotal / OUT.revenue_month : 0, 1) + ' от выручки)', 'total');
  grp('Постоянные расходы');
  row('Сервер', rub(OUT.hosting));
  row('Поддержка (' + num(OUT.support_h, 1) + ' ч × ' + rub(OUT.rate) + ')', rub(supportTotal));
  row('Итого постоянные', rub(fixedTotal), 'total');
  grp('Доли от выручки');
  row('Эквайринг (' + pct(OUT.acq, 1) + ')', rub(acqAmt));
  row('Налог (' + pct(OUT.tax, 1) + ')', rub(taxAmt));
  row('Админ партнёрского канала (' + pct(OUT.admin_share, 1) + ')', rub(adminAmt));
  row('Итого доли', rub(shareTotal), 'total');
  grp('Привлечение');
  row('CAC × платящих в месяц', rub(cacTotal), 'total');
  grp('Итого');
  row('Прибыль в месяц', rub(OUT.profit_month), 'total');
  var modelShare = OUT.revenue_month ? modelTotal / OUT.revenue_month : 0;
  var base = OUT.contrib > 0
    ? 'Каждый платящий приносит ' + rub(OUT.contrib) + ' после привлечения; весь список статей — ниже, от выручки до прибыли.'
    : 'Вклад платящего ' + rub(OUT.contrib) + ' — не больше нуля: с ростом числа платящих убыток только растёт.';
  $('#curve-sub').textContent = base + ' Модель — ' + pct(modelShare, 1) + ' от выручки: курс доллара и цены модели почти ' +
    'не влияют на прибыль, даже бесплатная модель не перекроет постоянные расходы и привлечение — для этого нужны охват, ' +
    'конверсия или цена подписки (раздел «хватит одного из» ниже).';
}
// торнадо больше не рисуется отдельной главой, но его данные (±20% по каждому параметру) всё ещё
// питают «хватит одного из» ниже (renderNeed) — отсюда и название остаётся tornadoData
var TOR = [];
function tornadoData() {
  var p0 = OUT.profit_month;
  return SLIDERS.map(function (p) {
    var lo = clampP(p, st[p.key] * 0.8), hi = clampP(p, st[p.key] * 1.2);
    var yl = evalWith(st, p.key, lo).profit_month, yh = evalWith(st, p.key, hi).profit_month;
    return { p: p, lo: lo, hi: hi, dl: yl - p0, dh: yh - p0, swing: Math.abs(yh - yl) };
  }).sort(function (a, b) { return b.swing - a.swing; });
}
// подбор: какое значение одного рычага даёт окупаемость разработки ≤ 12 мес (прибыль в месяц ≥ разработка / 12)
function solve(key) {
  var p = PARAM[key];
  function ok(x) { var v = evalWith(st, key, x); return v.profit_month > 0 && v.profit_month - v.dev / 12 >= 0; }
  var cur = st[key], c0 = ok(cur), N = 240, best = null, top = { y: -Infinity, x: cur };
  for (var i = 0; i <= N; i++) {
    var x = p.min + (p.max - p.min) * i / N;
    if (ok(x) !== c0 && (best === null || Math.abs(x - cur) < Math.abs(best - cur))) best = x;
    var y = evalWith(st, key, x).profit_month; if (y > top.y) top = { y: y, x: x };
  }
  if (best === null) return { reach: c0, none: true, top: top };
  var a = cur, b = best;
  var gs = (p.max - p.min) / N, near = best > cur ? Math.max(cur, best - gs) : Math.min(cur, best + gs);
  if (ok(near) === c0) a = near;
  for (var j = 0; j < 60; j++) { var mid = (a + b) / 2; if (ok(mid) === c0) a = mid; else b = mid; }
  return { reach: c0, x: c0 ? a : b, up: best > cur };
}
// тот же порог за пределами ползунка: бисекция по расширенному диапазону (доли — до 1, остальное — до ×100 от текущего)
function solveWide(key) {
  var p = PARAM[key]; if (!p) return null;
  var cur = st[key], hi = p.unit === 'доля' ? 1 : cur * 100;
  function ok(x) { var v = evalWith(st, key, x); return v.profit_month > 0 && v.profit_month - v.dev / 12 >= 0; }
  if (ok(cur) || !ok(hi)) return null;
  var a = cur, b = hi;
  for (var j = 0; j < 100; j++) { var mid = (a + b) / 2; if (ok(mid)) b = mid; else a = mid; }
  return b;
}
var WIDE = [['reach', 'охват', function (x) { return num(x / 1000) + ' тыс. ищущих в месяц'; }],
  ['c_pay', 'конверсия в оплату', function (x) { return pct(x, 1); }],
  ['apc', 'APC', function (x) { return num(x, 1) + ' покупки на платящего'; }]];
function renderNeed() {
  var host = $('#need'); host.replaceChildren();
  var already = isFinite(OUT.payback) && OUT.payback <= 12;
  var levers = TOR.filter(function (r) { return r.swing > 1e-6; }).slice(0, 6);
  var res = levers.map(function (r) { return solve(r.p.key); });
  var noneAll = !already && res.every(function (x) { return x.none; });
  H('h3', null, host, already ? 'Уже окупается за ' + num(OUT.payback, 1) + ' мес. Запас: окупаемость останется ≤ 12 мес, пока ОДИН рычаг не дойдёт до:'
    : 'Чтобы окупиться за 12 месяцев, хватит ОДНОГО из:');
  if (noneAll) H('p', { class: 'derived', style: 'margin:0 0 8px' }, host, 'Сейчас — ни одного: в одиночку ни один рычаг не вытягивает даже на всём своём диапазоне. ' +
    'Нужно двигать несколько сразу — например, сценарий «оптимист». Ниже — лучшее, чего достигает каждый рычаг.');
  var ol = H('ol', null, host);
  levers.forEach(function (r, i) {
    var x = res[i], li = H('li', null, ol);
    H('b', null, li, short(r.p.label));
    if (x.none && already) { li.append(': '); H('span', { class: 'na' }, li, 'держится во всём диапазоне'); }
    else if (x.none) {
      li.append(': '); H('span', { class: 'na' }, li, 'недостижимо; лучшее — ' + rub(x.top.y) + '/мес при ' + fmtP(r.p, x.top.x));
    } else {
      li.append(': ' + fmtP(r.p, st[r.p.key]) + ' → ' + fmtP(r.p, x.x, Math.abs(x.x) < 100 ? 1 : 0));
    }
  });
  if (noneAll) {
    var wide = WIDE.map(function (w) {
      var x = solveWide(w[0]); if (x === null) return null;
      var k = st[w[0]] === PARAM[w[0]].base ? ' к базе' : ' к текущему';
      return w[1] + ' ≥ ' + w[2](x) + ' (×' + num(x / st[w[0]], 1) + k + ')';
    }).filter(Boolean);
    if (wide.length) {
      var pw = H('p', { style: 'margin:10px 0 0;font-size:14px' }, host);
      H('b', null, pw, 'За пределами ползунков: ');
      pw.append(wide.join(', или ') + ' — каждое по отдельности нереально.');
    }
  }
  H('div', { class: 'derived' }, host, 'Рычаги — шесть параметров с самым сильным влиянием на прибыль при сдвиге на ±20%; подбор — бисекцией по формулам модели, остальные параметры как сейчас. Разработка ' + rub(OUT.dev) + ', нужно ≥ ' + rub(OUT.dev / 12) + ' прибыли в месяц.');
}
var formulaBuilt = false, fCells = {};
function unitFmt(v, unit) {
  if (!isFinite(v)) return 'не окупается';
  if (unit === 'доля') return pct(v, 1);
  if (unit === '$') return '$' + num(v, 6);
  if (unit === '₽' && Math.abs(v) < 10) return num(v, 3) + ' ₽';
  return num(v, Math.abs(v) < 100 ? 2 : 0) + ' ' + unit;
}
function renderFormulaTable() {
  if (!formulaBuilt) {
    var t = table($('#formula-table'), ['Показатель', 'Словами', 'Формула', 'Сейчас'], [], [3]);
    var tb = t.querySelector('tbody');
    E.formulas.forEach(function (f) {
      var tr = H('tr', null, tb); H('td', null, tr, f.label); H('td', null, tr, f.words);
      var c = H('td', null, tr); H('code', null, c, f.expr); fCells[f.key] = H('td', { class: 'n' }, tr);
    });
    var fixed = E.params.filter(function (p) { return !p.slider; });
    table($('#fixed-table'), ['Допущение', 'Значение', 'Источник'], fixed.map(function (p) { return [p.label, fmtP(p, p.base), p.source]; }), [1]);
    formulaBuilt = true;
  }
  E.formulas.forEach(function (f) { fCells[f.key].textContent = unitFmt(OUT[f.key], f.unit); });
}

// ---------- тема ----------
var THEMES = ['auto', 'light', 'dark'], THEME_RU = { auto: 'авто', light: 'светлая', dark: 'тёмная' };
function setTheme(t) {
  if (t === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', t);
  $('#theme-btn').textContent = 'Тема: ' + THEME_RU[t];
  try { localStorage.setItem('rent-theme', t); } catch (e) { /* хранилище недоступно */ }
}
function initTheme() {
  var q = (location.hash + '&' + location.search).match(/theme=(light|dark|auto)/), t = 'auto';
  if (q) t = q[1]; else { try { t = localStorage.getItem('rent-theme') || 'auto'; } catch (e) { t = 'auto'; } }
  if (THEMES.indexOf(t) < 0) t = 'auto';
  setTheme(t);
  $('#theme-btn').addEventListener('click', function () {
    var cur = document.documentElement.getAttribute('data-theme') || 'auto';
    setTheme(THEMES[(THEMES.indexOf(cur) + 1) % 3]);
  });
}

// ---------- запуск ----------
function renderCharts() { renderScatter(); renderDrivers(); renderFlow(); if (OUT) { renderBdTable(); } }
initTheme();
renderKpis(); scatterStatic(); bindIce(); drvStatic(); flowStatic(); renderExtract();
buildScenarios(); buildModelBox($('#model-bar')); buildTiles(); buildParams();
renderScatter(); renderDrivers(); renderFlow();
update();
var lastW = innerWidth, rt = null;
window.addEventListener('resize', function () {
  if (innerWidth === lastW) return; lastW = innerWidth;
  clearTimeout(rt); rt = setTimeout(renderCharts, 150);
});
var fd = $('#foot-dates');
var cianKey = Object.keys(D.flow.summary).find(function (k) { return /всего собрано/.test(k); });
var cianDates = cianKey && (cianKey.match(/\(([^)]+)\)/) || [])[1];
fd.textContent = 'Данные: Telegram — ' + monLabel(D.flow.months[0], true) + ' – ' + monLabel(D.flow.months[D.flow.months.length - 1], true) +
  (cianDates ? '; Циан — ' + cianDates : '') + '; курс доллара на ' + E.usd_rate_date + '; страница собрана ' + document.body.dataset.built + '.';
if (/debug/.test(location.hash)) { // проверка вёрстки: что вылезает за ширину окна
  setTimeout(function () {
    var over = [];
    document.querySelectorAll('body *').forEach(function (el) {
      var r = el.getBoundingClientRect();
      if (r.width && (r.right > innerWidth + 0.5 || r.left < -0.5) && !el.closest('#tip') && !el.closest('details:not([open])')) over.push((el.id || el.tagName) + ':' + Math.round(r.right));
    });
    var pre = H('pre', { id: 'dbg' }, document.body, JSON.stringify({ sw: document.documentElement.scrollWidth, iw: innerWidth, over: over.slice(0, 20) }));
  }, 300);
}
})();
"""

TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ассистент аренды СПб</title>
<meta name="description" content="Полная цена аренды в Петербурге: сюрприз доплаты, драйверы интереса, надёжность данных, разбор объявлений и калькулятор юнит-экономики">
<style>
__THEME__
__CSS__
</style>
</head>
<body data-built="__BUILT__">
__BODY__
<script type="application/json" id="dash-data">__DASH__</script>
<script type="application/json" id="econ-data">__ECON__</script>
<script type="application/json" id="eval-data">__EVAL__</script>
<script type="application/json" id="kpi-data">__KPI__</script>
<script id="calc-core">__CORE__</script>
<script>__UI__</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dash", default=DEFAULTS["dash"], help="dashboard_data.json")
    ap.add_argument("--econ", default=DEFAULTS["econ"], help="econ_model.json")
    ap.add_argument("--eval", default=DEFAULTS["eval"], help="eval_results.json (необязательно)")
    ap.add_argument("--out", default=DEFAULTS["out"], help="куда записать index.html")
    a = ap.parse_args()

    dash = load(a.dash)
    econ = load(a.econ)
    ev = load(a.eval, required=False)
    for key in ("params", "formulas", "models", "presets", "check_cases"):
        if key not in econ:
            sys.exit(f"в econ_model.json нет раздела {key}")
    k = kpi(dash)
    built = dash.get("generated") or econ.get("usd_rate_date") or ""
    html = TEMPLATE
    for ph, val in (("__THEME__", css_theme_blocks()), ("__CSS__", CSS), ("__BODY__", BODY),
                    ("__CORE__", JS_CORE), ("__UI__", JS_UI), ("__AUTHOR__", html_lib.escape(AUTHOR)),
                    ("__REPO_URL__", html_lib.escape(REPO_URL)),
                    ("__BUILT__", built), ("__DASH__", emb(dash)), ("__ECON__", emb(econ)),
                    ("__EVAL__", emb(slim_eval(ev))), ("__KPI__", emb(k))):
        html = html.replace(ph, val)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8", newline="\n")
    print(f"-> {out.name}: {out.stat().st_size / 1024:.0f} КБ; медианная доплата {k['median_extra']:.0f} ₽, "
          f"> 10%: {k['share_over10']:.0%}, без суммы КУ: {k['no_sum_all']:.0%}, топ-10: {k['top10_dropped']} из 10; "
          f"разбор: {'есть' if ev else 'нет файла'}")


if __name__ == "__main__":
    main()
