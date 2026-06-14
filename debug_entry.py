"""shutuba_past.html の実際のHTML構造をPlaywrightで確認"""
import logging
logging.basicConfig(level=logging.WARNING)

from jra_predictor.scraper.base import get_with_browser

URL = "https://race.netkeiba.com/race/shutuba_past.html?race_id=202609030411&rf=shutuba_submenu"

print(f"Fetching: {URL}")
soup = get_with_browser(URL, timeout_ms=20000)

if not soup:
    print("ERROR: got None")
    exit()

print(f"title: {soup.title.string if soup.title else 'none'}")

# 全テーブル
tables = soup.find_all("table")
print(f"\n=== tables: {len(tables)} ===")
for i, t in enumerate(tables[:8]):
    trs = t.find_all("tr")
    tds_first = trs[1].find_all(["td","th"]) if len(trs) > 1 else []
    print(f"[{i}] class={t.get('class')} rows={len(trs)} first_row_cells={len(tds_first)}")
    if tds_first:
        print(f"     cells: {[c.get_text(strip=True)[:15] for c in tds_first[:6]]}")

# 馬リンク
print(f"\n=== horse links ===")
links = soup.select("a[href*='/horse/']")
print(f"count: {len(links)}")
for a in links[:5]:
    print(f"  {a.get('href')} -> {a.get_text(strip=True)}")

# HorseListクラス
print(f"\n=== tr classes (unique) ===")
tr_classes = set()
for tr in soup.find_all("tr"):
    c = tr.get("class")
    if c:
        tr_classes.add(str(c))
for c in sorted(tr_classes)[:20]:
    print(f"  {c}")
