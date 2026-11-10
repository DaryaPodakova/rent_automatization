"""Парсер аренды Циан (Санкт-Петербург) по прямоугольникам вокруг станций метро.

Берёт JSON объявлений со страниц поиска (координаты, цена, условия, апартаменты или нет).
Телефоны не собирает.

    python rent_spb/cian_parser.py                         # зоны по умолчанию
    python rent_spb/cian_parser.py --zones zone_A --rooms 1 studio
    python rent_spb/cian_parser.py --zone "Моя=59.91,30.31,59.93,30.36"
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
SEARCH = "https://spb.cian.ru/cat.php?deal_type=rent&engine_version=2&offer_type=flat&region=2&type=4{rooms}&bbox={bbox}&p={p}"
ROOMS = {"studio": "room9", "1": "room1", "2": "room2", "3": "room3"}
PAGE_SIZE, MAX_PAGES, PAUSE_SEC = 28, 54, 4

# bbox = lat_min,lng_min,lat_max,lng_max; ~1,3 км от вестибюля (≈20 мин пешком по улицам)
ZONES = {
    "zone_A": "59.000,30.000,59.020,30.040",   # заполните своими координатами: lat_min,lng_min,lat_max,lng_max
}


def fetch(url):
    return subprocess.run(["curl", "-s", "-A", UA, url], capture_output=True).stdout.decode("utf-8", "ignore")


def offers_from(html):
    for m in re.finditer(r'"offers":\[\{', html):
        arr, _ = json.JSONDecoder().raw_decode(html, m.start() + len('"offers":'))
        if arr and "cianId" in arr[0]:
            return arr
    return []


def to_row(o, zone):
    g, bt, b = o.get("geo") or {}, o.get("bargainTerms") or {}, o.get("building") or {}
    return {
        "id": o["cianId"], "search_zone": zone, "url": f"https://spb.cian.ru/rent/flat/{o['cianId']}/",
        "lat": (g.get("coordinates") or {}).get("lat"), "lng": (g.get("coordinates") or {}).get("lng"),
        "address": g.get("userInput"),
        "metros": [(u.get("name"), u.get("time"), u.get("transportType")) for u in g.get("undergrounds") or []],
        "is_apartments": o.get("isApartments"), "rooms": o.get("roomsCount"), "flat_type": o.get("flatType"),
        "area": o.get("totalArea"), "kitchen": o.get("kitchenArea"), "floor": o.get("floorNumber"),
        "floors": b.get("floorsCount"), "build_year": b.get("buildYear"),
        "price": bt.get("priceRur") or bt.get("price"), "utilities": bt.get("utilitiesTerms"),
        "deposit": bt.get("deposit"), "client_fee": bt.get("clientFee"), "prepay_months": bt.get("prepayMonths"),
        "lease_term": bt.get("leaseTermType"), "has_furniture": o.get("hasFurniture"),
        "by_homeowner": o.get("isByHomeowner"), "added": o.get("added"),
        "photo": ((o.get("photos") or [{}])[0]).get("thumbnailUrl"), "photos": len(o.get("photos") or []),
        "description": o.get("description"),
    }


def collect(zones, rooms):
    room_qs = "".join(f"&{ROOMS[r]}=1" for r in rooms)
    rows = {}
    for zone, bbox in zones.items():
        for p in range(1, MAX_PAGES + 1):
            html = fetch(SEARCH.format(rooms=room_qs, bbox=bbox, p=p))
            if p == 1:
                total = re.search(r'"totalOffers":(\d+)', html)
                print(f"{zone}: на Циане {total.group(1) if total else '?'} объявлений", flush=True)
            offs = offers_from(html)
            new = [o for o in offs if o["cianId"] not in rows]
            rows.update({o["cianId"]: to_row(o, zone) for o in new})
            print(f"  стр. {p}: {len(offs)} объявлений, новых {len(new)}", flush=True)
            if len(offs) < PAGE_SIZE or not new:
                break
            time.sleep(PAUSE_SEC)
    return list(rows.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zones", nargs="+", default=list(ZONES), help=f"из {list(ZONES)}")
    ap.add_argument("--zone", action="append", default=[], help='своя зона "Имя=lat_min,lng_min,lat_max,lng_max"')
    ap.add_argument("--rooms", nargs="+", default=["studio", "1"], choices=list(ROOMS))
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "cian_raw.json"))
    a = ap.parse_args()

    zones = {z: ZONES[z] for z in a.zones if z in ZONES}
    zones.update(dict(z.split("=", 1) for z in a.zone))
    if not zones:
        sys.exit("нет зон для сбора")
    rows = collect(zones, a.rooms)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"итого {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
