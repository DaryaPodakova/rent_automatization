"""Объявления из Telegram-выгрузки канала + Циан (rent_scored.csv) -> одна таблица data/offers_all.csv.

    python rent_spb/tg_merge.py --tg topic_analysis/data/appartaments/result.json
    python rent_spb/tg_merge.py --tg ... --since 2025-01-01 --geocode   # + координаты по адресу по локальной выгрузке OSM data/osm_addr_spb.tsv

Из постов берутся только поля объявления; имена, @username, телефоны и комментарии не читаются и не сохраняются.
У Telegram-объявлений нет координат: без --geocode metro_min / club_min для них пустые.
"""
import argparse, json, re
from pathlib import Path
import pandas as pd
from rent_analysis import CLUBS, METRO, nearest

DATA = Path(__file__).parent / "data"
MONTHS = dict(zip("янв фев мар апр мая июн июл авг сен окт ноя дек".split(), range(1, 13)))
SCRAPE_MONTH = 10  # Циан снят 16–17.10.2026
CHANNEL = "leavestay"  # t.me/leavestay — «Уезжаю / Остаюсь»
PHONE = re.compile(r"(\+?\d[\d\s\-()]{8,}\d)")


def text(m):
    t = m["text"]
    return t if isinstance(t, str) else "".join(x if isinstance(x, str) else x.get("text", "") for x in t)


def field(t, emoji):
    m = re.search(emoji + r"️? ?([^\n]+)", t)
    return m.group(1).strip() if m else None


def parse_price(s):
    if not s:
        return None
    s = s.lower().replace("\xa0", " ")
    m = re.search(r"(\d{1,3}(?:[ .,]\d{3})+|\d+)\s*(к\b|тыс|т\.р)?", s)
    if not m:
        return None
    v = float(re.sub(r"[ .,]", "", m.group(1)))
    v = v * 1000 if m.group(2) and v < 1000 else v
    return v if 5000 <= v <= 500000 else None


def parse_rooms(s):
    s = (s or "").lower()
    if "студ" in s:
        return "студия"
    m = re.search(r"(\d)\s*-?\s*(?:комн|\+|$)", s)
    return f"{m.group(1)}-к" if m else None


def parse_post(m):
    t = text(m)
    head = t[:120].lower()
    city = re.search(r"#\s?(спб|санкт|петербург|питер)", head) and "спб"
    if not city:
        return None
    area = field(t, "📐") or t
    a = re.search(r"(\d+[.,]?\d*)\s*(?:кв\.?\s*м|м²|м2)", area, re.I)
    kind = (field(t, "🛋") or "").lower()
    free = not kind  # старый свободный формат
    obj = "комната" if "комнат" in kind and "комнатн" not in kind else "дом" if "дом" in kind else "квартира"
    fmt = parse_rooms(field(t, "👉") or (t[:200] if free else ""))
    metro = field(t, "🚇") or (re.search(r"метро\s+([А-Яа-я\- ]+?)(?:[,.]|\d| и )", t) or [None, None])[1]
    addr = field(t, "📍")
    start = re.search(r"c (\d\d)\.(\d\d)\.(\d{4})", field(t, "📆") or "")
    lease = (field(t, "⏱") or "").lower()
    return {
        "source": "telegram", "tg_id": m["id"], "posted": m["date"][:10], "object": obj, "fmt": fmt,
        "area": float(a.group(1).replace(",", ".")) if a else None,
        "price": parse_price(field(t, "💰")),
        "metro_tg": metro.lstrip("#").replace("_", " ").strip() if metro else None,
        "address": PHONE.sub("", addr).strip() if addr else None,
        "lease": ("длительно" if "длит" in lease else "коротко" if lease else None),
        "move_from": f"{start[3]}-{start[2]}-{start[1]}" if start else None,
        "template": not free,
    }


STREET_TYPES = set("улица ул проспект пр пр-т переулок пер набережная наб шоссе ш бульвар б-р площадь пл проезд аллея линия дорога тупик мост реки речки канала канал д дом".split())


def street_key(s):
    w = re.findall(r"[а-яa-z0-9]+", str(s).lower().replace("ё", "е"))
    return " ".join(sorted(x for x in w if x not in STREET_TYPES))


def house_key(h):
    h = str(h).lower().replace("ё", "е").replace("корпус", "к").replace("корп", "к")
    m = re.match(r"\s*(\d+)\s*(?:[/\-]|к\.?)?\s*(\d*)\s*([а-я]?)", h)
    return (m[1] + ("к" + m[2] if m[2] else "") + m[3]) if m else None


def station_key(n):
    return re.sub(r"[^а-яa-z0-9]+", " ", str(n).lower().replace("ё", "е")).strip()


def geocode(addresses, osm_path):
    """Адрес «Улица Х, 14к1» -> (lat, lng) по локальной выгрузке OSM; нет дома — центр улицы."""
    osm = pd.read_csv(osm_path, sep="	", names=["lat", "lng", "street", "house"], dtype=str).dropna()
    osm["lat"], osm["lng"] = osm.lat.astype(float), osm.lng.astype(float)
    osm["sk"], osm["hk"] = osm.street.map(street_key), osm.house.map(house_key)
    houses = osm.dropna(subset=["hk"]).groupby(["sk", "hk"])[["lat", "lng"]].mean()
    streets = osm.groupby("sk")[["lat", "lng"]].median()
    out = {}
    for a in addresses:
        m = re.match(r"(.+?)[,\s]+(\d[\d\s/\-кК.а-яА-Я]*)$", a.strip())
        if not m:
            continue
        sk, hk = street_key(m[1]), house_key(m[2])
        if (sk, hk) in houses.index:
            out[a] = [*houses.loc[(sk, hk)], "дом"]
        elif sk in streets.index:
            out[a] = [*streets.loc[sk], "улица"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tg", required=True, help="result.json экспорта канала")
    ap.add_argument("--cian", default=str(DATA / "rent_scored.csv"))
    ap.add_argument("--cian-raw", default=str(DATA / "cian_raw.json"))
    ap.add_argument("--since", default="2025-01-01")
    ap.add_argument("--geocode", action="store_true", help="привязать адреса по локальной выгрузке OSM")
    ap.add_argument("--osm", default=str(DATA / "osm_addr_spb.tsv"))
    a = ap.parse_args()

    raw = json.loads(Path(a.tg).read_text(encoding="utf-8"))["messages"]
    rows = [r for m in raw if m.get("type") == "message" and "💰" in text(m) and (r := parse_post(m))]
    tg = pd.DataFrame(rows)
    tg = tg[(tg.posted >= a.since) & tg.price.notna()].copy()
    tg["is_dup"] = tg.duplicated(["address", "price", "area"]) & tg.address.notna()

    tg["lat"] = tg["lng"] = None
    if a.geocode:
        cache = geocode(tg.address.dropna().unique(), a.osm)
        tg["geo"] = tg.address.map(lambda x: (cache.get(x) or [None] * 3)[2])
        for i, ad in tg.address.items():
            if ad in cache:
                tg.loc[i, ["lat", "lng"]] = cache[ad][:2]
        pts = tg.dropna(subset=["lat"])
        for col, places in (("target", METRO), ("club", CLUBS)):
            res = [nearest((r.lat, r.lng), places) for r in pts.itertuples()]
            tg.loc[pts.index, f"{col}_min"] = [round(x) for x, _ in res]
            tg.loc[pts.index, col] = [n for _, n in res]
    tg["total_month"] = tg.price  # коммуналка в постах текстом, не разбираем

    cian = pd.read_csv(a.cian, encoding="utf-8-sig")
    cian = cian.rename(columns={"metro": "target", "metro_min": "target_min"})
    cian["source"], cian["object"] = "циан", "квартира"
    # «added» у Циана — «26 сен, 18:17» / «вчера» / «сегодня»; снято в октябре 2026
    mon = cian.added.str.extract(r"\d+ (\w{3}),")[0].map(MONTHS).fillna(SCRAPE_MONTH).astype(int)
    cian["year"], cian["month"] = 2026 - (mon > SCRAPE_MONTH), mon
    # настоящее метро объявления: у Циана — ближайшая станция из карточки, у Telegram — из поста
    near = {x["url"]: min(x["metros"], key=lambda m: m[1])[0] for x in json.loads(Path(a.cian_raw).read_text(encoding="utf-8")) if x["metros"]}
    cian["metro"] = cian.url.map(near)
    names = {station_key(n): n for n in near.values()}
    tg["metro"] = tg.metro_tg.map(lambda n: names.get(station_key(n), str(n).capitalize()) if pd.notna(n) else None)
    tg["url"] = f"https://t.me/{CHANNEL}/" + tg.tg_id.astype(str)
    tg = tg.drop(columns=["metro_tg"])
    tg["source"] = "тг"
    tg["year"], tg["month"] = tg.posted.str[:4].astype(int), tg.posted.str[5:7].astype(int)
    all_ = pd.concat([cian, tg], ignore_index=True)
    first = ["source", "year", "month", "url", "metro"]
    all_ = all_[first + [c for c in all_.columns if c not in first]]
    all_.to_csv(DATA / "offers_all.csv", index=False, encoding="utf-8-sig")

    print(f"Циан: {len(cian)}, Telegram с {a.since}: {len(tg)} ({int(tg.template.sum())} по шаблону, "
          f"{int((~tg.is_dup).sum())} без повторов), с адресом {int(tg.address.notna().sum())}, "
          f"с координатами {int(tg.lat.notna().sum())}")
    print(tg.groupby("object").size().to_string())
    print(f"-> {DATA / 'offers_all.csv'}")


if __name__ == "__main__":
    main()
