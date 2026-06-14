"""オッズページのHTML構造を確認"""
import logging
logging.basicConfig(level=logging.WARNING)

from jra_predictor.scraper.base import get_with_browser

RACE_ID = "202609030411"
URL = f"https://race.netkeiba.com/odds/index.html?race_id={RACE_ID}&type=b1"

print(f"Fetching: {URL}")
soup = get_with_browser(URL, timeout_ms=25000)

if not soup:
    print("ERROR: got None")
    exit()

print(f"title: {soup.title.string if soup.title else 'none'}")

# テーブル構造
tables = soup.find_all("table")
print(f"\n=== tables: {len(tables)} ===")
for i, t in enumerate(tables[:5]):
    trs = t.find_all("tr")
    cls = t.get("class")
    print(f"[{i}] class={cls} rows={len(trs)}")
    for j, tr in enumerate(trs[:3]):
        tds = tr.find_all(["td","th"])
        print(f"  tr[{j}] class={tr.get('class')} cells={len(tds)}: {[c.get_text(strip=True)[:12] for c in tds[:6]]}")

# HorseList行
print(f"\n=== tr.HorseList ===")
rows = soup.select("tr.HorseList")
print(f"count: {len(rows)}")
for tr in rows[:3]:
    tds = tr.find_all("td")
    print(f"  cells={len(tds)}: {[td.get_text(strip=True)[:12] for td in tds[:8]]}")

# Odds系クラス
print(f"\n=== tr/td with odds-related classes ===")
for cls_pat in ["Odds","odds","Win","win","Num"]:
    els = soup.select(f"[class*='{cls_pat}']")
    if els:
        print(f"class*={cls_pat}: {len(els)} elements")
        for el in els[:2]:
            print(f"  <{el.name} class={el.get('class')}> {el.get_text(strip=True)[:30]}")

# 全trクラス（ユニーク）
print(f"\n=== unique tr classes ===")
tr_classes = set()
for tr in soup.find_all("tr"):
    c = tr.get("class")
    if c:
        tr_classes.add(str(c))
for c in sorted(tr_classes)[:15]:
    print(f"  {c}")

# divやtbodyのID/class
print(f"\n=== divs with id or class containing 'odds'/'horse' ===")
for el in soup.find_all(["div","section","tbody"]):
    id_ = el.get("id","")
    cls_ = " ".join(el.get("class") or [])
    if any(k in (id_+cls_).lower() for k in ["odds","horse","list"]):
        print(f"  <{el.name} id='{id_}' class='{cls_[:40]}'> children={len(el.find_all(recursive=False))}")
