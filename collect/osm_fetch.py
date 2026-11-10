"""Объекты OpenStreetMap по Петербургу (Overpass) и минуты пешком от объявлений до ближайшего объекта категории.

Готовый шаблон: категории («драйверы интереса») заданы пресетами, запросы руками писать не нужно.

    python rent_spb/osm_fetch.py fetch                      # все категории -> data/osm_poi_spb.tsv (кэш)
    python rent_spb/osm_fetch.py fetch metro fitness ice    # только эти; что уже в кэше — не качается
    python rent_spb/osm_fetch.py fetch park --force         # перекачать
    python rent_spb/osm_fetch.py list                       # пресеты и фильтры

Из кода:  df = osm_fetch.add_minutes(df, ["metro", "fitness"])  ->  колонки metro_min, fitness_min.
Минуты — rent_analysis.walk_min (прямая × 1,3 / 80 м в минуту). Данные © участники OpenStreetMap, ODbL.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from rent_analysis import walk_min  # noqa: E402

CACHE = Path(__file__).parent / "data" / "osm_poi_spb.tsv"
BBOX = "59.63,29.42,60.25,30.76"  # юг,запад,север,восток: город с Кронштадтом, Зеленогорском, Колпино, Пушкином
ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
UA = "rent_spb-osm_fetch/1.0 (study project)"  # Overpass просит осмысленный User-Agent, только ASCII
RETRIES, PAUSE_SEC, TIMEOUT_SEC = 5, 20, 300

# ключ -> фильтры Overpass; nwr = точка, линия или отношение (для линий и отношений берётся центр)
PRESETS = {
    "metro": ['node["railway"="station"]["station"="subway"]'],
    "fitness": ['nwr["leisure"="fitness_centre"]'],
    "kindergarten": ['nwr["amenity"="kindergarten"]'],
    "school": ['nwr["amenity"="school"]'],
    "clinic": ['nwr["amenity"~"^(clinic|hospital)$"]'],
    "ice": ['nwr["leisure"="ice_rink"]', 'nwr["leisure"~"^(stadium|sports_centre)$"]["sport"~"ice_hockey"]'],
    "sport": ['nwr["leisure"~"^(stadium|sports_centre)$"]'],
    "park": ['nwr["leisure"="park"]'],
}


def query(cat):
    body = "".join(f"  {f}({BBOX});\n" for f in PRESETS[cat])
    return f"[out:json][timeout:{TIMEOUT_SEC}];\n(\n{body});\nout center;"


def overpass(q):
    """POST в Overpass через curl; при 429/504/обрыве — пауза и следующий сервер."""
    for i in range(RETRIES):
        url = ENDPOINTS[i % len(ENDPOINTS)]
        r = subprocess.run(["curl", "-s", "-m", str(TIMEOUT_SEC + 60), "-A", UA, "-w", "\n%{http_code}",
                            "--data-urlencode", "data@-", url], input=q.encode("utf-8"), capture_output=True)
        out = r.stdout.decode("utf-8", "ignore")
        body, _, code = out.rpartition("\n")
        if code == "200":
            try:
                return json.loads(body)["elements"]
            except (ValueError, KeyError):
                code = "битый JSON"
        wait = PAUSE_SEC * (i + 1)
        print(f"  {url}: {code or 'нет ответа'}, жду {wait} с", flush=True)
        time.sleep(wait)
    sys.exit("Overpass не ответил, попробуйте позже")


def fetch_category(cat):
    rows = []
    for e in overpass(query(cat)):
        c = e if "lat" in e else e.get("center", {})
        if "lat" in c:
            name = " ".join(str(e.get("tags", {}).get("name", "")).split())  # без табов и переносов
            rows.append((cat, name, c["lat"], c["lon"]))
    return pd.DataFrame(rows, columns=["category", "name", "lat", "lng"])


def load(categories=None, force=False, path=CACHE):
    """Объекты нужных категорий из кэша; чего нет в кэше (или всё при force) — качается и дописывается."""
    categories = list(categories or PRESETS)
    unknown = set(categories) - set(PRESETS)
    if unknown:
        raise ValueError(f"нет пресета: {sorted(unknown)}; есть {list(PRESETS)}")
    path = Path(path)
    poi = pd.read_csv(path, sep="\t", keep_default_na=False) if path.exists() else \
        pd.DataFrame(columns=["category", "name", "lat", "lng"])
    todo = categories if force else [c for c in categories if c not in set(poi["category"])]
    for i, cat in enumerate(todo):
        if i:
            time.sleep(PAUSE_SEC / 4)
        new = fetch_category(cat)
        print(f"{cat}: {len(new)} объектов", flush=True)
        rest = poi[poi["category"] != cat]
        poi = pd.concat([rest, new], ignore_index=True) if len(rest) else new
        path.parent.mkdir(parents=True, exist_ok=True)
        poi.to_csv(path, sep="\t", index=False)  # после каждой категории: обрыв не теряет скачанное
    return poi[poi["category"].isin(categories)].reset_index(drop=True)


def nearest_min(lat, lng, poi_lat, poi_lng, k=3):
    """Минуты до ближайшего объекта. Кандидаты — k ближайших по плоской метрике (numpy, сразу матрицей),
    точное время — walk_min из rent_analysis только для них."""
    lat, lng = np.asarray(lat, float), np.asarray(lng, float)
    plat, plng = np.asarray(poi_lat, float), np.asarray(poi_lng, float)
    out = np.full(len(lat), np.nan)
    ok = ~(np.isnan(lat) | np.isnan(lng))
    if not ok.any() or not len(plat):
        return out
    cos = np.cos(np.radians(np.nanmean(lat[ok])))
    d2 = ((lat[ok, None] - plat) ** 2 + ((lng[ok, None] - plng) * cos) ** 2)
    k = min(k, len(plat))
    cand = np.argpartition(d2, k - 1, axis=1)[:, :k]
    out[ok] = [min(walk_min((a, b), (plat[j], plng[j])) for j in js)
               for a, b, js in zip(lat[ok], lng[ok], cand)]
    return out


def add_minutes(df, categories, lat="lat", lng="lng", poi=None):
    """Колонки <cat>_min: округлённые минуты пешком до ближайшего объекта категории (Int64, пусто без координат)."""
    poi = load(categories) if poi is None else poi
    df = df.copy()
    for cat in categories:
        p = poi[poi["category"] == cat]
        df[f"{cat}_min"] = pd.array(np.round(nearest_min(df[lat], df[lng], p["lat"], p["lng"])), dtype="Int64")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="скачать категории в кэш")
    f.add_argument("categories", nargs="*", default=list(PRESETS), help=f"из {list(PRESETS)}")
    f.add_argument("--force", action="store_true", help="перекачать, даже если есть в кэше")
    sub.add_parser("list", help="показать пресеты")
    a = ap.parse_args()
    if a.cmd == "list":
        for k, v in PRESETS.items():
            print(f"{k:13} {' | '.join(v)}")
        return
    poi = load(a.categories, force=a.force)
    counts = poi["category"].value_counts().reindex(a.categories, fill_value=0)
    print(counts.to_string(), f"\nитого {len(poi)} -> {CACHE}", sep="\n")


if __name__ == "__main__":
    main()
