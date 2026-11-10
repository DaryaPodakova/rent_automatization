"""Воспроизводимая выборка эталона из экспорта Telegram: пул -> дедуп -> квоты по месяцам -> dev/test -> маскировка.

    python rent_spb/pipeline/gold_sample.py --tg path/to/result.json              # -> data/gold/sample_{dev,test}.jsonl
    python rent_spb/pipeline/gold_sample.py --tg ... --out <папка> --force        # перезаписать уже существующие файлы
    python rent_spb/pipeline/gold_sample.py --tg ... --out <папка> --compare data/gold   # сверить с gold_{dev,test}.jsonl

Пул — сообщения с 01.01.2025 двух слоёв: «шаблон» (хэштег города в начале поста и строка «🛋») и «свободный» (хэштег
города без «🛋» и рубрика «Нестыдные | Петербург»). Дубли — по первому: тот же адрес, цена и площадь; тот же код рубрики
«#Аnnn». В слое квоты по месяцам пропорциональны пулу (наибольший остаток), внутри месяца — случайно; dev — равномерно
по времени: шаблон 20 + 150, свободный 10 + 50. Тексты маскируются функцией mask(): её же импортирует
eval_extraction.py, а узел n8n «Маскировка контактов» получает те же правила через eval_extraction.py --sync-workflow.
Скрипт пишет только в --out и не перезаписывает существующие файлы без --force.
"""
import argparse, collections, json, random, re, sys
from pathlib import Path

HERE = Path(__file__).parent
GOLD = HERE.parent / "data" / "gold"
SEED, SINCE = 20261101, "2025-01-01"
PLAN = {"шаблон": (20, 150), "свободный": (10, 50)}   # слой: (dev, test)
CITY = re.compile(r"#\s?(спб|санкт|петербург|питер)")
RUBRIC = re.compile(r"Нестыдные\s*\|\s*Петербург")

# ---------- маскировка контактов: одна функция для эталона, eval_extraction.py и узла n8n ----------
# Правила — строки в синтаксисе Python re; для JavaScript их переводит eval_extraction.js_regex() (\w, \s, \d, \b, «.»
# и «$» — с той же семантикой Unicode, что в Python). Порядок важен: карты и мессенджеры раньше общих ссылок,
# почта раньше ника. (шаблон, флаги, метка, сохранить группу 1 перед меткой)
SEP = r"[\s\-‐-―\u00a0().]*"
MASK_RULES = [
    (r"(?:https?://)?(?:www\.)?(?:yandex\.[a-z]+/maps|maps\.yandex\.[a-z]+|goo\.gl/maps|maps\.app\.goo\.gl|"
     r"google\.[a-z.]+/maps|2gis\.[a-z]+)\S*", "i", "[MAP]", False),
    (r"(?:https?://)?(?:wa\.me|api\.whatsapp\.com|chat\.whatsapp\.com|invite\.viber\.com|viber\.click)/?\S*", "i",
     "[MESSENGER_URL]", False),
    (r"[\w.\-]+@[\w\-]+\.[\w.]+", "", "[EMAIL]", False),
    (r"(https?://\S+|www\.\S+|t\.me/\S+|\b[\w\-]+\.(?:ru|com|me|org|net|рф|su|io|link)\b(?:/\S*)?)", "i", "[URL]", False),
    (r"(?<!\d)(?:\d{4}[ \-]?){3}\d{4}(?!\d)", "", "[CARD]", False),
    (r"(?<![\d.])(?:\+\s?7|8|7)?" + SEP + r"[3489]\d{2}" + SEP + r"\d{3}" + SEP + r"\d{2}" + SEP + r"\d{2}(?!\d)", "",
     "[PHONE]", False),
    (r"@[A-Za-z0-9_]{2,}", "", "[USER]", False),
    (r"(👋️?)[ \t]*[^\n]+", "", "[NAME]", True),     # подпись после 👋 — имя
    (r"((?:[Нн]а связи|[Дд]изайнер\w*|[Фф]отограф\w*|[Аа]втор\w* (?:фото|интерьера|проекта))(?:\s+(?:собственни\w*|"
     r"хозя\w*|агент\w*|риелтор\w*|риэлтор\w*|арендодател\w*|владел\w*|менеджер\w*|представител\w*))?\s*[:—\-]?)\s*"
     r"([А-ЯЁA-Z][а-яёa-z]+)(?:\s+[А-ЯЁA-Z][а-яёa-z]+)?", "", "[NAME]", True),
]
_RULES = [(re.compile(p, re.I if "i" in f else 0), label, keep) for p, f, label, keep in MASK_RULES]


def mask(text, names=()):
    """Контакты -> [MAP] [MESSENGER_URL] [EMAIL] [URL] [CARD] [PHONE] [USER] [NAME]; names — доп. имена дословно."""
    for rx, label, keep in _RULES:
        text = rx.sub(lambda m: (m[1] + " " if keep else "") + label, text)
    for n in names:
        text = text.replace(n, "[NAME]")
    return text


# ---------- пул и выборка ----------
def post_text(m):
    t = m["text"]
    return t if isinstance(t, str) else "".join(x if isinstance(x, str) else x.get("text", "") for x in t)


def _line(t, emoji):
    return (re.search(emoji + r"[^\n]*", t) or [""])[0]


def build_pool(messages, since=SINCE):
    """Посты слоёв «шаблон» и «свободный» с ключом дедупа; порядок — как в экспорте."""
    pool = []
    for m in messages:
        if m.get("type") != "message" or m["date"] < since:
            continue
        t = post_text(m)
        if not t:
            continue
        if CITY.search(t[:200].lower()):
            layer = "шаблон" if "🛋" in t else "свободный"
            addr = re.sub(r"\W", "", _line(t, "📍").lower())
            key = ("T", addr, _line(t, "💰"), _line(t, "📐")) if addr else ("T", m["id"])
        elif RUBRIC.search(t):
            code = re.search(r"#А(\d{3})", t[:80])
            layer, key = "свободный", ("N", code[1] if code else m["id"])
        else:
            continue
        pool.append({"id": m["id"], "month": m["date"][:7], "layer": layer, "key": key, "text": t})
    return pool


def dedup(pool):
    seen, out = set(), []
    for p in pool:
        if p["key"] not in seen:
            seen.add(p["key"])
            out.append(p)
    return out


def allocate(items, n, rng):
    """n постов с квотами по месяцам пропорционально пулу (наибольший остаток); внутри месяца — rng.shuffle."""
    by = collections.defaultdict(list)
    for p in items:
        by[p["month"]].append(p)
    quota = {k: n * len(v) / len(items) for k, v in by.items()}
    base = {k: int(v) for k, v in quota.items()}
    for k in sorted(quota, key=lambda k: -(quota[k] - base[k]))[: n - sum(base.values())]:
        base[k] += 1
    out = []
    for k in sorted(by):
        v = by[k][:]
        rng.shuffle(v)
        out += v[: base[k]]
    return out


def sample(pool, seed=SEED, plan=PLAN):
    """Выборка по слоям в порядке plan (общий генератор — порядок слоёв влияет на результат); dev — равномерно по времени."""
    rng, sel = random.Random(seed), []
    for layer, (n_dev, n_test) in plan.items():
        s = allocate([p for p in pool if p["layer"] == layer], n_dev + n_test, rng)
        s.sort(key=lambda p: (p["month"], p["id"]))
        dev = {round(i * (len(s) - 1) / (n_dev - 1)) for i in range(n_dev)}
        sel += [dict(p, split="dev" if i in dev else "test") for i, p in enumerate(s)]
    return sel


def compare(out_rows, gold_dir):
    """Совпадение множеств post_id с gold_{dev,test}.jsonl и число постов, чей замаскированный текст отличается."""
    ok = True
    for split in ("dev", "test"):
        path = gold_dir / f"gold_{split}.jsonl"
        gold = {json.loads(s)["post_id"]: json.loads(s)["text_masked"]
                for s in path.read_text(encoding="utf-8").splitlines() if s.strip()}
        mine = {r["post_id"]: r["text_masked"] for r in out_rows if r["split"] == split}
        same = set(gold) == set(mine)
        ok &= same
        print(f"{split}: эталон {len(gold)}, выборка {len(mine)}, post_id совпадают: {'да' if same else 'НЕТ'}"
              + ("" if same else f" (только в эталоне {len(set(gold) - set(mine))}, только в выборке "
                                 f"{len(set(mine) - set(gold))})")
              + f"; текст отличается у {sum(gold[k] != mine[k] for k in set(gold) & set(mine))}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tg", type=Path, required=True, help="экспорт Telegram (result.json)")
    ap.add_argument("--out", type=Path, default=GOLD, help="папка для sample_dev.jsonl / sample_test.jsonl")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--since", default=SINCE)
    ap.add_argument("--names", type=Path, help="файл с именами для маскировки дословно, по одному в строке (не в репозитории)")
    ap.add_argument("--force", action="store_true", help="перезаписать существующие файлы")
    ap.add_argument("--compare", type=Path, help="папка с gold_dev.jsonl / gold_test.jsonl для сверки post_id")
    a = ap.parse_args()
    files = {s: a.out / f"sample_{s}.jsonl" for s in ("dev", "test")}
    exist = [str(f) for f in files.values() if f.exists()]
    if exist and not a.force:
        sys.exit("Файлы уже есть (нужен --force): " + ", ".join(exist))
    names = [s.strip() for s in a.names.read_text(encoding="utf-8").splitlines() if s.strip()] if a.names else []
    raw = build_pool(json.loads(a.tg.read_text(encoding="utf-8"))["messages"], a.since)
    pool = dedup(raw)
    months = sorted({p["month"] for p in raw})
    print(f"Пул с {a.since}: {len(raw)} постов, после дедупа {len(pool)} за {len(months)} мес. "
          f"(в среднем {len(raw) / len(months):.0f} / {len(pool) / len(months):.0f} в месяц)")
    print("Доли слоёв после дедупа: " + ", ".join(
        f"{lay} {sum(p['layer'] == lay for p in pool) / len(pool):.3f}" for lay in PLAN))
    rows = [{"post_id": p["id"], "split": p["split"], "layer": p["layer"], "month": p["month"],
             "text_masked": mask(p["text"], names)} for p in sample(pool, a.seed)]
    rows.sort(key=lambda r: (r["split"] != "dev", r["post_id"]))
    a.out.mkdir(parents=True, exist_ok=True)
    for split, path in files.items():
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"-> {path} ({sum(r['split'] == split for r in rows)})")
    if a.compare:
        sys.exit(0 if compare(rows, a.compare) else 1)


if __name__ == "__main__":
    main()
