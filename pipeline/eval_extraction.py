"""Оценка извлечения карточек из объявлений: регулярки (baseline) против языковой модели на размеченной выборке.

    python rent_spb/pipeline/eval_extraction.py --dry-run          # сколько вызовов и сколько это стоит; API не вызывается
    python rent_spb/pipeline/eval_extraction.py --no-llm           # только регулярки
    python rent_spb/pipeline/eval_extraction.py --mode hybrid --dry-run   # гибрид: доля постов в LLM и стоимость
    python rent_spb/pipeline/eval_extraction.py                    # регулярки + модель (ключ LLM_API_KEY / OPENROUTER_API_KEY в .env)
    python rent_spb/pipeline/eval_extraction.py --selftest         # примеры из system_prompt.md проходят схему
    python rent_spb/pipeline/eval_extraction.py --sync-workflow    # промпт, схема, модель, регулярки, маскировка -> n8n_workflow.json

Эталон — data/gold/gold_test.jsonl (по строке на пост: текст + карточка по card_schema.json). Промпт пишется по dev-выборке,
test открывается только здесь. Перед моделью текст маскируется функцией gold_sample.mask() — той же, что в эталоне
и (через --sync-workflow) в n8n. Гибрид (--mode hybrid) — по ПОЛЯМ: коммуналка, комиссия, залог и роль автора (author_role)
берутся из регулярок, а модель подключается к посту, только если регулярка по группе ничего не нашла (needs_llm_fields());
остальные поля (deal, цена, дата заезда…) — из модели, если пост не по шаблону (нет «🛋»), карточка регулярок не проходит
схему или цена не найдена: needs_llm(), в n8n — узел «Валидация регулярок» (по-полевая маршрутизация в n8n пока не сделана).
Ответы модели кэшируются в data/gold/llm_cache.jsonl: повторный запуск денег не тратит. Итог — таблица в консоль
и data/gold/eval_results.json. Схему проверяет встроенный валидатор подмножества JSON Schema — тот же, что в узле
«Валидация карточки» workflow (правила в одном файле card_schema.json, исполнителя два: Python и JavaScript).
"""
import argparse, collections, hashlib, inspect, json, os, re, sys, time
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
GOLD = ROOT / "data" / "gold"
SCHEMA = json.loads((HERE / "card_schema.json").read_text(encoding="utf-8"))
PROMPT_MD = HERE / "system_prompt.md"
WORKFLOW = HERE / "n8n_workflow.json"
sys.path[:0] = [str(HERE), str(ROOT)]
import tg_merge  # noqa: E402  — разбор шаблонного поста Telegram: цена, формат, площадь, метро, срок
from gold_sample import MASK_RULES, mask  # noqa: E402  — маскировка контактов: та же, что в эталоне и в n8n

URL = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY, FALLBACK = "deepseek/deepseek-v4.1-flash", "moonshotai/kimi-k2.6"
PRICES = {PRIMARY: (0.045, 0.50), FALLBACK: (0.65, 3.41)}  # $ за 1 млн токенов вход/выход, каталог OpenRouter 30.10.2026
OR_FEE, RUB_PER_USD = 0.055, 85         # комиссия OpenRouter при пополнении; курс на 31.10.2026
MAX_TOKENS, TEMPERATURE, TIMEOUT = 1200, 0, 60  # 400 не хватало: DeepSeek V4.1 тратит часть потолка на скрытые
# reasoning-токены (usage.completion_tokens_details.reasoning_tokens), JSON обрезался на середине — найдено 07.11
CHARS_PER_TOKEN = 2.7                   # грубая оценка для русского текста, только для --dry-run
USER_TEMPLATE = "Источник: {source}\nДата публикации: {posted}\n«««\n{text}\n»»»"
POOL_WEIGHTS = {"шаблон": 0.902, "свободный": 0.098}   # доли слоёв в пуле постов; в эталоне свободный текст завышен
POSTS_PER_MONTH = 54   # постов в пуле в месяц: 1189 за 01.2025–10.2026 (gold_sample.py); для --dry-run

ENUMS = ["is_offer", "deal", "kind", "rooms", "utilities_status", "commission_status", "deposit_status", "author_role"]
NUMBERS = ["price_rub", "area_m2", "floor", "floors", "utilities_rub", "commission_rub", "commission_pct",
           "deposit_rub", "metro_walk_min"]
STRINGS = ["metro", "street", "move_from"]
STATUSES = {"utilities_status": "не указано", "commission_status": "не указано", "deposit_status": "не указано",
            "deal": "unknown", "author_role": "неизвестно"}
FLAGS = SCHEMA["properties"]["red_flags"]["items"]["enum"]


# ---------- валидатор подмножества JSON Schema (зеркало JS в узле «Валидация карточки») ----------
def _type_ok(v, t):
    num = isinstance(v, (int, float)) and not isinstance(v, bool)
    return {"null": v is None, "boolean": isinstance(v, bool), "number": num,
            "integer": num and float(v).is_integer(), "string": isinstance(v, str),
            "array": isinstance(v, list), "object": isinstance(v, dict)}[t]


def _eq(a, b):
    return a == b and isinstance(a, bool) == isinstance(b, bool) and (a is None) == (b is None)


def validate(v, s, path="$"):
    """Список ошибок вида «$.поле: ключевое_слово»; пустой список — карточка проходит схему."""
    errs = []
    if "type" in s:
        if not any(_type_ok(v, t) for t in (s["type"] if isinstance(s["type"], list) else [s["type"]])):
            return [f"{path}: type"]
    if "enum" in s and not any(_eq(v, e) for e in s["enum"]):
        errs.append(f"{path}: enum")
    if "const" in s and not _eq(v, s["const"]):
        errs.append(f"{path}: const")
    if _type_ok(v, "number"):
        if "minimum" in s and v < s["minimum"]: errs.append(f"{path}: minimum")
        if "maximum" in s and v > s["maximum"]: errs.append(f"{path}: maximum")
        if "exclusiveMinimum" in s and v <= s["exclusiveMinimum"]: errs.append(f"{path}: exclusiveMinimum")
    if isinstance(v, str):
        if "minLength" in s and len(v) < s["minLength"]: errs.append(f"{path}: minLength")
        if "maxLength" in s and len(v) > s["maxLength"]: errs.append(f"{path}: maxLength")
        if "pattern" in s and not re.search(s["pattern"], v): errs.append(f"{path}: pattern")
    if isinstance(v, list):
        for i, x in enumerate(v):
            if "items" in s: errs += validate(x, s["items"], f"{path}[{i}]")
        if s.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in v}) < len(v):
            errs.append(f"{path}: uniqueItems")
        if "contains" in s and not any(not validate(x, s["contains"]) for x in v):
            errs.append(f"{path}: contains")
    if isinstance(v, dict):
        errs += [f"{path}.{k}: required" for k in s.get("required", []) if k not in v]
        for k, sub in s.get("properties", {}).items():
            if k in v: errs += validate(v[k], sub, f"{path}.{k}")
        if s.get("additionalProperties") is False:
            errs += [f"{path}.{k}: additionalProperties" for k in v if k not in s.get("properties", {})]
    for sub in s.get("allOf", []):
        errs += validate(v, sub, path)
    if "not" in s and not validate(v, s["not"], path):
        errs.append(f"{path}: not")
    if "if" in s:
        branch = s.get("then") if not validate(v, s["if"], path) else s.get("else")
        if branch: errs += validate(v, branch, path)
    return errs


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[«»\"“”]", "", str(s).lower().replace("ё", "е"))).strip()


def check_card(card, text):
    """Схема + правило вне схемы: каждая цитата evidence дословно есть в (замаскированном) тексте."""
    errs = validate(card, SCHEMA)
    if isinstance(card, dict) and isinstance(card.get("evidence"), dict):
        t = _norm(text)
        errs += [f"$.evidence.{k}: not_in_text" for k, q in card["evidence"].items()
                 if isinstance(q, str) and _norm(q) not in t]
    return errs


# ---------- baseline: регулярки ----------
# Коммуналка — копия регулярок из генератора тетради данных итогового проекта (build_data_nb.py, util_from_text):
# там они живут внутри строки с кодом ячейки, импортировать нельзя. Менять — в обоих местах.
KU = (r"(?<![а-яёa-z])(?:ку|к/у|к\.у\.?|жку|жкх|коммунальные(?:\s+(?:услуги|платежи))?|"
      r"коммунальных(?:\s+(?:услуг|платежей))?|коммуналка|коммуналку|коммуналки|комуналка|комуналку)(?![а-яёa-z])")
NUM = r"(\d{1,3}(?:[ .,'’]\d{3})+|\d+(?:[.,]\d)?)"
UNIT = r"[ \t]*(к|тыс\w*|т\.?[ \t]?р\.?|т(?![а-яё])|₽|руб\w*|р(?![а-яё]))?(?![ \t]*%)"
RANGE = NUM + UNIT + r"(?:[ \t]*(?:[-–—]|до)[ \t]*" + NUM + UNIT + ")?"
FILL = r"(?:[ \t:=~≈(+\-—–]|примерно|около|ориентировочно|в[ \t]+среднем|зимой|летом|от|до|по|порядка|в[ \t]+районе)*"
KU_INCL = re.compile(
    r"(?:" + KU + r")(?:(?!не\s|без\s)[^.\n!;]){0,25}?(?:включ|входят|входит|в\s+стоимост|в\s+цен)|"
    r"(?<!не\s)(?:включая|включен\w*|входят|входит|вместе\s+с|с\s+уч[её]том|в\s+т\.?\s?ч\.?)[\s:(\-—–]+(?:все\s+)?"
    r"(?:оплата\s+)?(?:" + KU + r")|(?:цена|стоимость|оплата|аренда)\s+с\s+(?:" + KU + r")", re.I)
KU_AFTER = re.compile("(?:" + KU + r")" + FILL + RANGE, re.I)
KU_BEFORE = re.compile(r"(?:\+|плюс)[ \t~]*(?:от[ \t]*)?" + RANGE + r"[ \t]*(?:в[ \t]+)?\(?[ \t]*(?:" + KU + r")", re.I)
METERS = re.compile(r"по\s+сч[её]тчик|сч[её]тчик\w*\s+(?:отдельно|оплачива|сверху|дополнительно|не\s+включ)|"
                    r"\+\s*сч[её]тчик|(?:свет|электричество|вода|электроэнергия)\s+(?:по\s+сч|отдельно)", re.I)
# «счётчики сверху» = платится сверху, сумма не названа: «+КУ», «+ к/у», «+ счётчики», «КУ оплачиваются отдельно»,
# «по счётчикам», «включены, кроме счётчиков»; «не указано» — только когда о коммуналке ни слова (разметка эталона 01.11)
KU_ON_TOP = re.compile(r"(?:\+|плюс)[ \t]*(?:" + KU + r")|(?:" + KU + r")[ \t]+(?:оплачива\w*[ \t]+)?"
                       r"(?:отдельно|сверху|дополнительно|по[ \t]+квитанц)", re.I)
EXCEPT_METERS = re.compile(r"кроме[ \t]+сч[её]тчик", re.I)


def to_rub(v, unit):
    v = float(re.sub(r"[ .,'’]", "", v)) if re.search(r"[ .,'’]\d{3}$", v) else float(v.replace(",", "."))
    return v * 1000 if v < 100 else v


def util_from_text(t):
    if not isinstance(t, str) or not t:
        return "не указано", None
    if KU_INCL.search(t):
        return ("счётчики сверху" if METERS.search(t) or EXCEPT_METERS.search(t) else "включена"), None
    for rx in (KU_AFTER, KU_BEFORE):
        for m in rx.finditer(t):
            g = m.groups()
            vals = [to_rub(g[0], g[1])] + ([to_rub(g[2], g[3] or g[1])] if g[2] else [])
            if all(300 <= v <= 30000 for v in vals):
                return "сумма", round(sum(vals) / len(vals))
    return ("счётчики сверху" if METERS.search(t) or KU_ON_TOP.search(t) else "не указано"), None


util_status = util_from_text   # прежнее имя: уточнение 01.11 теперь внутри util_from_text, как в build_data_nb.py


MONEY = NUM + r"[ \t]*(к|тыс\w*|т\.?[ \t]?р\.?|т(?![а-яё])|₽|руб\w*|р(?![а-яё]))?"
ROOMS = {"студия": "студия", "1-к": "1", "2-к": "2"}
# Регулярки разбора — здесь; узел n8n «Регулярки» получает их же через узел «Настройки» (regex_json, --sync-workflow).
# Ищут по тексту в нижнем регистре, флаги объявлений (flag_*) — по исходному тексту без учёта регистра.
RX = {k: re.compile(p, f) for k, p, f in (
    ("daily", r"посуточн|/\s*сутки|в\s+сутки", 0),
    ("price_ctx", r"(?:цена|стоимость|аренда|оплата)[^\n\d]{0,20}([^\n]{0,30})", 0),
    ("floor", r"этаж\s*(\d+)\s*(?:из|/)\s*(\d+)|(\d+)\s*/\s*(\d+)\s*эт|(\d+)\s*этаж|этаж\s*(\d+)", 0),
    ("walk", r"(\d+)\s*мин\w*\.?\s*(?:пешком\s*)?до\s*метро|до\s*метро[^\n\d]{0,25}?(\d+)\s*мин", 0),
    ("com_none", r"без\s*комисс|комисси[яи]\s+нет|комиссия\s*:?\s*0\s*%?(?!\d)|от\s+собственник|без\s+посредник", 0),
    ("com_pct", r"(?:комисс\w*|агентск\w*)[^\n\d%]{0,15}(\d{1,3})\s*%", 0),
    ("com_sum", r"(?:комисс\w*|агентск\w*)[^\n\d]{0,15}" + MONEY, 0),
    ("dep_none", r"без\s+(?:залога|депозита)", 0),
    ("dep_sum", r"(?:залог|депозит)\w*[^\n\d]{0,15}" + MONEY, 0),
    ("flag_prepay", r"(?:предоплат|перевед|переведит|бронь)[^\n]{0,40}(?:до|без)\s+просмотр"
                    r"(?!\w*\s+просят\s+только\s+мошенник)", re.I),   # шапка канала — не флаг
    ("flag_messenger", r"whats\s?app|ватсап|вотсап|viber|вайбер|\[MESSENGER_URL\]", re.I),
    ("flag_injection", r"игнорир\w*\s+(?:все\s+)?(?:предыдущ|правил|инструкц)|ignore\s+(?:all\s+)?previous|"
                       r"(?:модель|нейросеть|ии|бот)[,:]?\s+(?:отметь|пометь|напиши|ответь)", re.I),
    # кто автор: риэлтор — агент/агентство/АН/риэлтор (кроме «без агентов», «агентам не звонить»); собственник — явные слова
    ("ar_realtor", r"(?<![а-яё])(?:агент(?:ом|а|у|ы|ов|ам)?|агентств\w*|риэлтор\w*|риелтор\w*)(?![а-яё])", 0),
    ("ar_an", r"(?<![А-Яа-яЁёA-Za-z])АН(?![А-Яа-яЁёA-Za-z])", 0),   # аббревиатура: ищется по исходному тексту
    ("ar_neg", r"(?:без|не|нет)\s+(?:агент\w*|риэлтор\w*|риелтор\w*)|(?:агент\w*|риэлтор\w*|риелтор\w*)[,\s]+"
               r"(?:просьба\s+|прошу\s+|пожалуйста\s+)?(?:не|без)\s", 0),
    ("ar_owner", r"на\s+связи\s+собственник|от\s+собственник|от\s+хозя[ие]н|без\s+посредник|сда[её]т\s+собственник", 0),
    ("seeker", r"^\s*(?:#\S+\s+)*(?:ищу|ищем|сниму|снимем)", 0),
    ("street_tail", r"[,\s]+(?:д\.?|дом)?\s*\d+.*$", 0),
    ("months", r"от\s+(\d+)\s*(?:-?х\s*)?мес", 0),
    ("long_words", r"надолго|длительн|долгосрочн", 0),
    ("apart", r"апартамент|апарт-отел", 0),
    ("room_word", r"(?<![а-яё])комнат[ауы]?\b", 0),
    ("rooms_adj", r"\d\s*-?\s*комнатн", 0),
    ("closed", r"\bсдан[аы]?\b", 0),
)}
FLAG_RX = (("предоплата_без_просмотра", "flag_prepay"), ("уход_в_другой_мессенджер", "flag_messenger"),
           ("инструкция_модели_в_тексте", "flag_injection"))
# Цитата для статуса коммуналки: какими выражениями её искать (по исходному тексту, re.I)
UTIL_QUOTE = {"включена": ("KU_INCL",), "сумма": ("KU_AFTER", "KU_BEFORE"),
              "счётчики сверху": ("EXCEPT_METERS", "METERS", "KU_ON_TOP", "KU_INCL")}
# Копии регулярок tg_merge.py — только для узла n8n (Python берёт их из tg_merge; совпадение проверяет --selftest)
TG_RX = {"tg_field": (r"️? ?([^\n]+)", 0), "tg_price": (r"(\d{1,3}(?:[ .,]\d{3})+|\d+)\s*(к\b|тыс|т\.р)?", 0),
         "tg_price_sep": (r"[ .,]", 0), "tg_rooms": (r"(\d)\s*-?\s*(?:комн|\+|$)", 0),
         "tg_area": (r"(\d+[.,]?\d*)\s*(?:кв\.?\s*м|м²|м2)", re.I),
         "tg_metro": (r"метро\s+([А-Яа-я\- ]+?)(?:[,.]|\d| и )", 0), "tg_start": (r"c (\d\d)\.(\d\d)\.(\d{4})", 0),
         "tg_phone": (r"(\+?\d[\d\s\-()]{8,}\d)", 0),
         "rub_thousands": (r"[ .,'’]\d{3}$", 0), "rub_sep": (r"[ .,'’]", 0)}   # две последние — из to_rub()


def _quote(text, low, m):
    """Цитата для evidence: найденный фрагмент как в тексте (поиск шёл по low той же длины), не длиннее 200 знаков."""
    if not m:
        return None
    return (text[m.start():m.end()] if len(text) == len(low) else m[0]).strip()[:200] or None


def baseline_card(text, posted=None):
    """Карточка по регулярным выражениям: логика разбора поста из tg_merge.parse_post + коммуналка + комиссия/залог.
    Цитаты evidence — найденные фрагменты: без них карточка не проходит схему (R9–R13) и в гибриде ушла бы в LLM."""
    p = tg_merge.parse_post({"id": 0, "date": posted or "1970-01-01", "text": "#спб\n" + text}) or {}
    low = text.lower()
    m_daily = RX["daily"].search(low)
    daily = bool(m_daily)
    price = ev_price = None
    if not daily and p.get("price"):
        price, ev_price = p["price"], tg_merge.field(text, "💰")
    elif not daily:
        src = tg_merge.field(text, "💸")
        mc = None if src else RX["price_ctx"].search(low)
        price, ev_price = tg_merge.parse_price(src or (mc[1] if mc else None)), src or _quote(text, low, mc)
    fl = RX["floor"].search(low)
    floor = int(fl[1] or fl[3] or fl[5] or fl[6]) if fl else None
    floors = int(fl[2] or fl[4]) if fl and (fl[2] or fl[4]) else None
    walk = RX["walk"].search(low)
    util, util_rub = util_status(text)
    ev_util = next((_quote(text, text, m) for k in UTIL_QUOTE.get(util, ()) if (m := globals()[k].search(text))), None)
    com_st, com_rub, com_pct = "не указано", None, None
    if m_com := RX["com_none"].search(low):
        com_st = "нет"
    elif m_com := RX["com_pct"].search(low):
        com_st, com_pct = "процент", float(m_com[1])
    elif m_com := RX["com_sum"].search(low):
        com_st, com_rub = "сумма", int(to_rub(m_com[1], m_com[2]))
    dep_st, dep_rub = "не указано", None
    if m_dep := RX["dep_none"].search(low):
        dep_st = "нет"
    elif m_dep := RX["dep_sum"].search(low):
        dep_st, dep_rub = "сумма", int(to_rub(m_dep[1], m_dep[2]))
    fmt = p.get("fmt")
    rooms = ROOMS.get(fmt) or ("3+" if fmt and fmt[0].isdigit() and int(fmt[0]) >= 3 else None)
    flags = [f for f, k in FLAG_RX if RX[k].search(text)]
    is_offer = not RX["seeker"].search(low)
    if is_offer and not price and not daily:
        flags.append("нет_цены")
    street = p.get("address") and RX["street_tail"].sub("", p["address"]).strip() or None
    months = RX["months"].search(low)
    m_long = None if months else RX["long_words"].search(low)
    i_until = text.find("📆 до")
    if daily:
        deal, ev_deal = "daily", _quote(text, low, m_daily)
    elif i_until >= 0:
        deal, ev_deal = "short", text[i_until:].split("\n")[0].strip()[:200]
    elif p.get("lease") in ("длительно", "коротко"):
        deal, ev_deal = {"длительно": "long", "коротко": "short"}[p["lease"]], tg_merge.field(text, "⏱")
    elif months:
        deal, ev_deal = ("long" if int(months[1]) >= 11 else "short"), _quote(text, low, months)
    else:
        deal, ev_deal = ("long", _quote(text, low, m_long)) if m_long else ("unknown", None)
    if RX["apart"].search(low):
        kind = "апартаменты"
    elif "🛋" in text:
        kind = {"квартира": "квартира", "комната": "комната", "дом": "другое"}.get(p.get("object"), "unknown")
    else:   # свободный текст: есть площадь или этаж и нет признаков комнаты — квартира
        kind = ("комната" if RX["room_word"].search(low) and not RX["rooms_adj"].search(low)
                else "квартира" if (p.get("area") or floor) else "unknown")
    q_com, q_dep = _quote(text, low, m_com), _quote(text, low, m_dep)
    # роль автора: признаки риэлтора сильнее (комиссия платится посреднику); «без комиссии» роль не доказывает
    low_ar = RX["ar_neg"].sub(lambda m: " " * len(m[0]), low)
    role, q_role = "неизвестно", None
    if is_offer:
        if m_ar := RX["ar_realtor"].search(low_ar) or RX["ar_an"].search(text):
            role, q_role = "риэлтор", _quote(text, low, m_ar)
        elif com_st in ("сумма", "процент"):
            role, q_role = "риэлтор", q_com
        elif m_ar := RX["ar_owner"].search(low):
            role, q_role = "собственник", _quote(text, low, m_ar)
    evidence = {"price_rub": price and ev_price, "deal": ev_deal, "utilities_status": ev_util,
                "utilities_rub": util_rub and ev_util, "commission_status": q_com, "commission_pct": com_pct and q_com,
                "commission_rub": com_rub and q_com, "deposit_status": q_dep, "deposit_rub": dep_rub and q_dep,
                "author_role": q_role}
    return {"is_offer": is_offer, "is_closed": bool(RX["closed"].search(low)), "deal": deal, "kind": kind,
            "rooms": rooms, "area_m2": p.get("area"), "floor": floor, "floors": floors,
            "price_rub": int(price) if price else None, "utilities_status": util, "utilities_rub": util_rub,
            "commission_status": com_st, "commission_rub": com_rub, "commission_pct": com_pct,
            "deposit_status": dep_st, "deposit_rub": dep_rub, "metro": p.get("metro_tg"),
            "metro_walk_min": int(walk[1] or walk[2]) if walk else None,
            "street": street, "move_from": p.get("move_from"), "red_flags": flags,
            "author_role": role, "evidence": {k: v for k, v in evidence.items() if isinstance(v, str) and v}}


def needs_llm(text, card):
    """Гибрид: причины отдать пост языковой модели; пусто — карточка регулярок идёт в запись. Зеркало узла n8n
    «Валидация регулярок»: (а) нет строки шаблона «🛋»; (б) карточка не проходит схему и межполевые правила;
    (в) регулярки не нашли цену."""
    return [r for r, bad in (("не_шаблон", "🛋" not in text), ("не_прошла_проверку", bool(check_card(card, text))),
                             ("нет_цены", card.get("price_rub") is None)) if bad]


# Гибрид по ПОЛЯМ: деньги и роль автора берутся из регулярок; LLM-значение группы — только если регулярка ничего не нашла
FIELD_GROUPS = {"utilities_status": ("utilities_status", "utilities_rub"),
                "commission_status": ("commission_status", "commission_rub", "commission_pct"),
                "deposit_status": ("deposit_status", "deposit_rub"), "author_role": ("author_role",)}
REGEX_FIELDS = {f for g in FIELD_GROUPS.values() for f in g}   # поля, не отдаваемые модели при «нашли регулярки»


def needs_llm_fields(card):
    """Группы полей, по которым регулярки ничего не нашли (статус «не указано», роль «неизвестно») -> причины «поле:…»."""
    return ["поле:" + k for k in FIELD_GROUPS if card.get(k) == STATUSES[k]]


def merge_hybrid(base, llm, post_reasons, field_reasons):
    """Карточка гибрида. Без причин или без ответа модели — карточка регулярок. Остальные поля (deal, цена, дата заезда, тип…)
    — из модели, если пост не по шаблону / не прошёл проверку / нет цены. Группы денег и роль автора — из регулярок,
    а из модели только те, что названы в field_reasons."""
    if not llm or not (post_reasons or field_reasons):
        return base
    take = {f for k in FIELD_GROUPS if "поле:" + k in field_reasons for f in FIELD_GROUPS[k]}
    if post_reasons:
        take |= set(SCHEMA["properties"]) - REGEX_FIELDS - {"evidence"}
    out = {k: (llm[k] if k in take and k in llm else v) for k, v in base.items() if k != "evidence"}
    ev_l, ev_b = llm.get("evidence") or {}, base.get("evidence") or {}
    out["evidence"] = {k: v for k in set(ev_l) | set(ev_b) if isinstance(v := (ev_l if k in take else ev_b).get(k), str)}
    return out


# ---------- языковая модель через OpenRouter ----------
def system_prompt():
    return PROMPT_MD.read_text(encoding="utf-8").strip()


def build_request(model, text, posted, source=None):
    """Запрос как в узле «Маскировка контактов»: источник и дата — отдельными строками до текста, не внутри него."""
    user = USER_TEMPLATE.format(source=source or "неизвестен", posted=posted or "неизвестна", text=text)
    return {"model": model, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"}, "usage": {"include": True},
            "messages": [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]}


def cache_key(req):
    return hashlib.sha256(json.dumps(req, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_cache(path):
    if not path.exists():
        return {}
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["key"]: r for r in recs}


def call_llm(text, row, key, cache, cache_path):
    """Основная модель; при ошибке или таймауте — запасная (как в workflow). Ответы — в кэш."""
    import requests
    for model in (PRIMARY, FALLBACK):
        req = build_request(model, text, row["posted"], row["source"])
        k = cache_key(req)
        if k in cache:
            return cache[k]
        t0 = time.time()
        try:
            r = requests.post(URL, json=req, timeout=TIMEOUT,
                              headers={"Authorization": f"Bearer {key}", "X-Title": "rent-spb-eval"})
            r.raise_for_status()
            body = r.json()
            rec = {"key": k, "model": body.get("model", model), "content": body["choices"][0]["message"]["content"],
                   "usage": body.get("usage", {}), "latency_ms": round((time.time() - t0) * 1000)}
        except Exception as e:  # noqa: BLE001 — любая ошибка сети/API ведёт на запасную модель
            print(f"  {model}: {type(e).__name__}: {str(e)[:120]}")
            continue
        cache[k] = rec
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
    return None


def parse_card(content):
    try:
        card = json.loads(re.sub(r"^```(?:json)?|```$", "", (content or "").strip()).strip())
        return card if isinstance(card, dict) else None
    except json.JSONDecodeError:
        return None


# ---------- эталон и метрики ----------
def load_gold(path):
    """Строка эталона: текст (text / text_masked / post) и карточка (card / gold / label или поля на верхнем уровне)."""
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        card = next((r[k] for k in ("card", "gold", "label", "labels") if isinstance(r.get(k), dict)), r)
        text = next(r[k] for k in ("text", "text_masked", "post") if k in r)
        rows.append({"id": r.get("id", r.get("post_id", i)), "posted": r.get("posted", r.get("date")),
                     "source": r.get("source", r.get("channel")), "text": text,
                     "layer": "шаблон" if "🛋" in text else "свободный",   # слой выборки — по строке шаблона «🛋»
                     "card": {k: card[k] for k in SCHEMA["properties"] if k in card}})   # is_closed не размечен
    return rows


def _num_eq(a, b, tol=0.0):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol * abs(float(b))


def _str_norm(s):
    return re.sub(r"^(?:м\.|метро|ст\.\s*м\.)\s*", "", _norm(s)) if s is not None else None


def full_price_known(c):
    return bool(c and c.get("price_rub") is not None and (c.get("utilities_status") == "включена" or (
        c.get("utilities_status") == "сумма" and c.get("utilities_rub") is not None)))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p and r else None, "tp": tp, "fp": fp, "fn": fn}


def metrics(gold, pred):
    """Поля — на постах, где эталон is_offer = true; is_offer — на всех."""
    allp = list(zip(gold, pred))
    offers = [(g, p or {}) for g, p in allp if g["is_offer"]]
    out = {"is_offer": {"accuracy": sum((p or {}).get("is_offer") == g["is_offer"] for g, p in allp) / len(allp)}}
    for f in ENUMS[1:]:
        out[f] = {"accuracy": sum(p.get(f) == g[f] for g, p in offers) / len(offers)}
    for f in NUMBERS:
        known = [(g, p) for g, p in offers if g[f] is not None]
        out[f] = {"n_known": len(known),
                  "exact_all": sum((p.get(f) is None and g[f] is None) or _num_eq(p.get(f), g[f]) for g, p in offers)
                  / len(offers),
                  "exact_known": sum(_num_eq(p.get(f), g[f]) for g, p in known) / len(known) if known else None,
                  "pm5_known": sum(_num_eq(p.get(f), g[f], 0.05) for g, p in known) / len(known) if known else None,
                  "invented_rate": sum(g[f] is None and p.get(f) is not None for g, p in offers) / len(offers)}
    for f in STRINGS:
        out[f] = {"accuracy": sum(_str_norm(p.get(f)) == _str_norm(g[f]) for g, p in offers) / len(offers)}
    for f, unk in STATUSES.items():
        gk = [g[f] not in (unk, None) for g, _ in offers]
        pk = [p.get(f) not in (unk, None) for _, p in offers]
        out[f + " (известно)"] = prf(sum(a and b for a, b in zip(gk, pk)), sum(b and not a for a, b in zip(gk, pk)),
                                     sum(a and not b for a, b in zip(gk, pk)))
    for fl in FLAGS:
        gk = [fl in (g["red_flags"] or []) for g, _ in allp]
        pk = [fl in ((p or {}).get("red_flags") or []) for _, p in allp]
        out["флаг " + fl] = prf(sum(a and b for a, b in zip(gk, pk)), sum(b and not a for a, b in zip(gk, pk)),
                                sum(a and not b for a, b in zip(gk, pk)))
    out["полная цена известна (доля)"] = {"share": sum(full_price_known(p) for _, p in offers) / len(offers)}
    return out


def by_layer(rows, preds):
    """Метрики по слоям (шаблон / свободный) и взвешенно по пулу: свободный текст в эталоне взят с повышенной долей."""
    out = {}
    for layer in POOL_WEIGHTS:
        idx = [i for i, r in enumerate(rows) if r["layer"] == layer]
        if any(rows[i]["card"]["is_offer"] for i in idx):
            out[layer] = metrics([rows[i]["card"] for i in idx], [preds[i] for i in idx])
    if len(out) == len(POOL_WEIGHTS):
        a, b = (out[k] for k in POOL_WEIGHTS)
        wa, wb = POOL_WEIGHTS.values()
        out["взвешенно"] = {f: {m: (None if a[f][m] is None or b[f][m] is None else wa * a[f][m] + wb * b[f][m])
                                for m in a[f] if m not in ("tp", "fp", "fn", "n_known")} for f in a}
    return out


# подписи столбцов: г-рег — гибрид на постах, оставшихся у регулярок; b→llm — регулярки на постах, ушедших в LLM
METHOD_LABELS = {"baseline": "base", "llm": "llm", "hybrid": "гибр", "hybrid_regex": "г-рег", "baseline_to_llm": "b→llm"}


def print_table(res):
    cols = [(c, layer) for c in METHOD_LABELS if c in res["methods"] for layer in res["methods"][c]]
    first = res["methods"][cols[0][0]][cols[0][1]]
    head = {"шаблон": "шабл", "свободный": "своб", "взвешенно": "взв"}
    print(f"\n{'поле':<40}{'метрика':<14}" + "".join(f"{METHOD_LABELS[c] + ':' + head[lay]:>11}" for c, lay in cols))
    for field in first:
        for metric in first[field]:
            if metric in ("tp", "fp", "fn", "n_known"):
                continue
            vals = [res["methods"][c][lay][field].get(metric) for c, lay in cols]
            print(f"{field:<40}{metric:<14}" + "".join(f"{'—' if v is None else f'{v:.1%}':>11}" for v in vals))
    print("\nДоля карточек с известной полной ценой (аренда + КУ известна или включена):")
    for lay, share in res["gold_full_price_share"].items():
        print(f"  {lay:<10} эталон {share:.1%} / " + " / ".join(
            f"{c} {res['methods'][c][lay]['полная цена известна (доля)']['share']:.1%}"
            for c in res["methods"] if lay in res["methods"][c]))


def estimate_cost(rows, n=None):
    sp = len(system_prompt())
    texts = [r["text"] for r in rows] or ["x" * 600]
    tin = (sp + sum(len(USER_TEMPLATE) + len(t) for t in texts) / len(texts)) / CHARS_PER_TOKEN
    n = n or len(rows)
    out = {}
    for label, tout in (("типично", 250), ("потолок", MAX_TOKENS)):
        usd = n * (tin * PRICES[PRIMARY][0] + tout * PRICES[PRIMARY][1]) / 1e6 * (1 + OR_FEE)
        out[label] = {"calls": n, "tokens_in_per_call": round(tin), "tokens_out_per_call": tout,
                      "usd": round(usd, 4), "rub": round(usd * RUB_PER_USD, 2)}
    return out


# ---------- синхронизация с n8n ----------
_W = r"\p{L}\p{N}_"   # \w Python (Unicode) в JavaScript
_SP = r"\t\n\x0b\x0c\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000"   # \s Python
_JS_OUT = {"w": f"[{_W}]", "W": f"[^{_W}]", "d": r"\p{Nd}", "D": r"\P{Nd}", "s": f"[{_SP}]", "S": f"[^{_SP}]",
           "b": f"(?:(?<=[{_W}])(?![{_W}])|(?<![{_W}])(?=[{_W}]))",
           "B": f"(?:(?<=[{_W}])(?=[{_W}])|(?<![{_W}])(?![{_W}]))"}
_JS_IN = {"w": _W, "d": r"\p{Nd}", "s": _SP}
UTIL_RX = ("KU_INCL", "KU_AFTER", "KU_BEFORE", "METERS", "KU_ON_TOP", "EXCEPT_METERS")
VALIDATOR_BLOCK = re.compile(r"// >>> валидатор.*?// <<< валидатор[^\n]*", re.S)


def js_regex(pattern):
    """Шаблон Python re -> источник RegExp для JavaScript (флаг u) с той же семантикой: \\w \\d \\s \\b — по Unicode,
    «.» — любой символ, кроме \\n, «$» — конец строки или перед последним \\n. Незнакомая конструкция — ошибка."""
    out, i, cls = [], 0, False
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            e = pattern[i + 1]
            if e in "ux":
                n = 6 if e == "u" else 4
                out.append(pattern[i:i + n])
                i += n
                continue
            if cls and e in _JS_IN:
                out.append(_JS_IN[e])
            elif not cls and e in _JS_OUT:
                out.append(_JS_OUT[e])
            elif e == "-":
                out.append("\\-" if cls else "-")
            elif e in "tnrfv.()[]{}*+?|^$\\/" or (e.isdigit() and not cls):
                out.append(c + e)
            else:
                raise ValueError(f"js_regex: \\{e} в {pattern!r}")
            i += 2
            continue
        if cls:
            cls = c != "]"
            out.append(c)
        elif c == "[":
            if pattern[i + 1:i + 2] == "]" or pattern[i + 1:i + 3] == "^]":
                raise ValueError(f"js_regex: «]» в начале класса в {pattern!r}")
            cls = True
            out.append(c)
        else:
            out.append({".": r"[^\n]", "$": r"(?=\n?$)"}.get(c, c))
        i += 1
    return "".join(out)


def _js_flags(flags):
    return "i" if flags & re.I else ""


def workflow_settings():
    rx = {k: [js_regex(v.pattern), _js_flags(v.flags)] for k, v in RX.items()}
    rx.update({k: [js_regex(globals()[k].pattern), _js_flags(globals()[k].flags)] for k in UTIL_RX})
    rx.update({k: [js_regex(p), _js_flags(f)] for k, (p, f) in TG_RX.items()})
    return {"system_prompt": system_prompt(), "schema_json": json.dumps(SCHEMA, ensure_ascii=False),
            "user_template": USER_TEMPLATE,
            "models_json": json.dumps({"primary": PRIMARY, "fallback": FALLBACK, "max_tokens": MAX_TOKENS,
                                       "temperature": TEMPERATURE, "prices_per_1m_usd": PRICES, "or_fee": OR_FEE},
                                      ensure_ascii=False),
            "regex_json": json.dumps(rx, ensure_ascii=False),
            "mask_json": json.dumps([[js_regex(p), f, label, keep] for p, f, label, keep in MASK_RULES],
                                    ensure_ascii=False)}


def sync_workflow(write=False):
    """Сверяет (или переписывает при write=True) workflow с файлами: узел «Настройки» — промпт, схема, модель,
    регулярки и маскировка; общий блок валидатора — из узла «Валидация карточки» в «Валидация регулярок»."""
    if not WORKFLOW.exists():
        return ["нет n8n_workflow.json"]
    wf = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {n["name"]: n["parameters"] for n in wf["nodes"]}
    assigns = nodes["Настройки"]["assignments"]["assignments"]
    have, diff = {a["name"]: a for a in assigns}, []
    for k, v in workflow_settings().items():
        if k not in have:
            assigns.append({"id": f"c3333333-0000-4000-8000-{len(assigns) + 1:012d}", "name": k, "value": v,
                            "type": "string"})
        elif have[k]["value"] != v:
            have[k]["value"] = v
        else:
            continue
        diff.append(k)
    src, dst = (nodes.get(n, {}).get("jsCode", "") for n in ("Валидация карточки", "Валидация регулярок"))
    blk, old = VALIDATOR_BLOCK.search(src), VALIDATOR_BLOCK.search(dst)
    if blk and old and blk[0] != old[0]:
        nodes["Валидация регулярок"]["jsCode"] = dst.replace(old[0], blk[0])
        diff.append("валидатор в «Валидация регулярок»")
    if write and diff:
        WORKFLOW.write_text(json.dumps(wf, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return diff


def selftest():
    """Примеры из промпта проходят схему и проверку цитат; маскировка снимает контакты; копии регулярок tg_merge
    совпадают с исходником; все регулярки переводятся для JavaScript."""
    md = system_prompt()
    ok = True
    for m in re.finditer(r"«««\n(.*?)\n»»»\n(\{.*?\})(?:\n|$)", md, re.S):
        errs = check_card(json.loads(m[2]), m[1])
        ok &= not errs
        print("пример:", "ок" if not errs else errs)
    for raw, want in (("звоните +7 (921) 123-45-67", "звоните [PHONE]"), ("пишите @some_user", "пишите [USER]"),
                      ("цена 40000-45000", "цена 40000-45000"), ("https://wa.me/79210000000", "[MESSENGER_URL]")):
        ok &= mask(raw) == want
        print("маска:", repr(mask(raw)), "ок" if mask(raw) == want else f"ждали {want!r}")
    src = Path(tg_merge.__file__).read_text(encoding="utf-8") + inspect.getsource(to_rub)
    stale = [k for k, (p, _) in TG_RX.items() if f'r"{p}"' not in src]
    ok &= not stale
    print("копии регулярок tg_merge:", "ок" if not stale else f"разошлись {stale}")
    workflow_settings()   # js_regex падает на незнакомой конструкции
    return ok


def routing_stats(rows, routes):
    """Гибрид: доля постов, ушедших в LLM, по слоям и взвешенно по пулу; причины (а–в) — сколько раз каждая."""
    out = {}
    for lay in POOL_WEIGHTS:
        idx = [i for i, r in enumerate(rows) if r["layer"] == lay]
        k = sum(bool(routes[i]) for i in idx)
        out[lay] = {"n": len(idx), "llm": k, "share": k / len(idx) if idx else None,
                    "reasons": dict(collections.Counter(x for i in idx for x in routes[i]))}
    if all(out[lay]["share"] is not None for lay in POOL_WEIGHTS):
        out["взвешенно"] = {"share": sum(w * out[lay]["share"] for lay, w in POOL_WEIGHTS.items())}
    return out


def print_costs(rows, llm_rows, w_share, monthly):
    """Стоимость гибрида против «всё в LLM»: на эталоне, на 200 постах потока (доли пула) и на месячном потоке."""
    print(f"Стоимость, {PRIMARY} с комиссией OpenRouter {OR_FEE:.1%}, ₽ (типично / потолок по выходу):")
    def cost(sample, n):   # длина текста — средняя по своей выборке: в гибриде в LLM уходит в основном свободный текст
        return " / ".join(f"{e['rub']:.2f}" for e in estimate_cost(sample, n).values()) if n else "0.00 / 0.00"
    for label, n_all, n_hyb in ((f"эталон, {len(rows)} постов", len(rows), len(llm_rows)),
                                ("поток, 200 постов", 200, 200 * w_share),
                                (f"месяц, {monthly} постов", monthly, monthly * w_share)):
        print(f"  {label:<22} всё в LLM: {n_all:>5.0f} выз. {cost(rows, n_all):>13} | "
              f"гибрид: {n_hyb:>5.1f} выз. {cost(llm_rows, n_hyb):>13}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gold", type=Path, default=GOLD / "gold_test.jsonl")
    ap.add_argument("--mode", choices=("baseline", "llm", "hybrid"), default="llm",
                    help="baseline — только регулярки; llm — регулярки и модель на всех постах; hybrid — модель "
                         "только для постов по правилам needs_llm()")
    ap.add_argument("--dry-run", action="store_true", help="не вызывать API: число вызовов и оценка стоимости")
    ap.add_argument("--no-llm", action="store_true", help="только регулярки (то же, что --mode baseline)")
    ap.add_argument("--monthly", type=int, default=POSTS_PER_MONTH, help="постов в месяц для оценки стоимости")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sync-workflow", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    diff = sync_workflow(write=a.sync_workflow)
    if diff:
        print(("Обновлено в n8n_workflow.json: " if a.sync_workflow else
               "ВНИМАНИЕ: workflow расходится с файлами (запустите --sync-workflow): ") + ", ".join(diff))
    if a.sync_workflow:
        return
    if a.no_llm:
        a.mode = "baseline"
    rows = load_gold(a.gold)[: a.limit] if a.gold.exists() else []
    if not rows:
        print(f"Нет эталона {a.gold}: метрики не считаются.")
    key = None
    if a.mode != "baseline" and not a.dry_run:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
        if not key:
            print("Ключа нет (OPENROUTER_API_KEY / LLM_API_KEY) — режим --dry-run.")
            a.dry_run = True
    cache_path = GOLD / "llm_cache.jsonl"
    cache = load_cache(cache_path)
    base = [baseline_card(r["text"], r["posted"]) for r in rows]
    post_rs = [needs_llm(r["text"], c) if a.mode == "hybrid" else [] for r, c in zip(rows, base)]
    field_rs = [needs_llm_fields(c) if a.mode == "hybrid" else [] for c in base]
    routes = [p + f if a.mode == "hybrid" else ["все_в_llm"] for p, f in zip(post_rs, field_rs)]
    to_llm = [i for i, rs in enumerate(routes) if rs]
    if a.dry_run and a.mode != "baseline":
        todo = [i for i in to_llm
                if cache_key(build_request(PRIMARY, mask(rows[i]["text"]), rows[i]["posted"], rows[i]["source"]))
                not in cache]
        print(f"Вызовов модели нужно: {len(todo)} из {len(to_llm)} отправляемых в LLM (остальные в кэше); постов "
              f"{len(rows)}.")
        if a.mode == "llm":
            for n in sorted({len(todo), 200} - {0}):
                for label, e in estimate_cost(rows or [], n).items():
                    print(f"  {n} вызовов, {label}: ~{e['tokens_in_per_call']} ток. вход + {e['tokens_out_per_call']} "
                          f"выход на вызов → ${e['usd']} ≈ {e['rub']} ₽ ({PRIMARY}, с комиссией OpenRouter {OR_FEE:.1%})")
    if not rows:
        return
    gold = [r["card"] for r in rows]
    bad_gold = sum(bool(validate(g, SCHEMA)) for g in gold)
    layers = {lay: sum(r["layer"] == lay for r in rows) for lay in POOL_WEIGHTS}
    share = {lay: sum(full_price_known(r["card"]) for r in rows if r["layer"] == lay and r["card"]["is_offer"])
             / max(1, sum(r["layer"] == lay and r["card"]["is_offer"] for r in rows)) for lay in POOL_WEIGHTS}
    if all(layers.values()):
        share["взвешенно"] = sum(w * share[lay] for lay, w in POOL_WEIGHTS.items())
    res = {"date": date.today().isoformat(), "gold_file": a.gold.name, "mode": a.mode, "n": len(rows),
           "layers": layers, "pool_weights": POOL_WEIGHTS, "n_offers": sum(bool(g["is_offer"]) for g in gold),
           "gold_failing_schema": bad_gold, "gold_full_price_share": share,
           "prompt_sha256": hashlib.sha256(system_prompt().encode()).hexdigest()[:12], "methods": {}}
    res["methods"]["baseline"] = by_layer(rows, base)
    llm = {}
    if a.mode != "baseline" and not a.dry_run:
        usage, invalid, fallback = [0, 0], 0, 0
        for i in to_llm:
            masked = mask(rows[i]["text"])
            rec = call_llm(masked, rows[i], key, cache, cache_path)
            llm[i] = parse_card(rec and rec["content"])
            invalid += llm[i] is None or bool(check_card(llm[i], masked))
            fallback += bool(rec and FALLBACK in rec["model"])
            usage[0] += (rec or {}).get("usage", {}).get("prompt_tokens", 0)
            usage[1] += (rec or {}).get("usage", {}).get("completion_tokens", 0)
        res["llm_run"] = {"model": PRIMARY, "calls": len(to_llm), "fallback_used": fallback, "invalid_cards": invalid,
                          "tokens_in": usage[0], "tokens_out": usage[1]}
    if a.mode == "llm" and llm:
        res["methods"]["llm"] = by_layer(rows, [llm[i] for i in range(len(rows))])
    if a.mode == "hybrid":
        res["routing"] = routing_stats(rows, routes)
        print("Гибрид — в LLM: " + ", ".join(
            f"{lay} {v['llm']}/{v['n']} = {v['share']:.1%} {v['reasons']}" for lay, v in res["routing"].items()
            if lay in POOL_WEIGHTS) + (f"; взвешенно по пулу {res['routing']['взвешенно']['share']:.1%}"
                                        if "взвешенно" in res["routing"] else ""))
        if a.dry_run:
            print_costs(rows, [rows[i] for i in to_llm], res["routing"].get("взвешенно", {}).get("share", 1), a.monthly)
        keep = [i for i in range(len(rows)) if not routes[i]]
        if llm:
            res["methods"]["hybrid"] = by_layer(rows, [merge_hybrid(base[i], llm.get(i), post_rs[i], field_rs[i])
                                                       for i in range(len(rows))])
        else:
            res["hybrid_llm_part"] = {"status": "ждёт прогона", "posts": len(to_llm)}
            print(f"Метрики гибрида на {len(to_llm)} постах, ушедших в LLM: ждёт прогона (нет ключа). Ниже «г-рег» — "
                  f"итог гибрида на {len(keep)} постах, оставшихся у регулярок; «b→llm» — регулярки на LLM-части.")
        for name, idx in (("hybrid_regex", keep), ("baseline_to_llm", to_llm)):   # подвыборки: без взвешивания по пулу
            res["methods"][name] = by_layer([rows[i] for i in idx], [base[i] for i in idx])
            res["methods"][name].pop("взвешенно", None)
    print(f"Эталон: {res['n']} постов ({', '.join(f'{k} {v}' for k, v in layers.items())}), объявлений "
          f"{res['n_offers']}; не проходят схему: {bad_gold}. Поля — на объявлениях; «взв» — по долям пула {POOL_WEIGHTS}")
    print_table(res)
    out = GOLD / "eval_results.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
