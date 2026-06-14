"""Playwrightでnetkeiba両ページのテスト"""
import logging
logging.basicConfig(level=logging.INFO)

from jra_predictor.scraper.base import get_with_browser

RACE_ID = '202606140901'

# 1. db.netkeiba 結果ページ
print("=== db.netkeiba result page (Playwright) ===")
url1 = f"https://db.netkeiba.com/race/{RACE_ID}/"
soup1 = get_with_browser(url1, wait_selector="table.race_table_01", timeout_ms=20000)
if soup1:
    print(f"title: {soup1.title.string if soup1.title else 'none'}")
    table = soup1.select_one("table.race_table_01")
    print(f"race_table_01: {'found' if table else 'not found'}")
    if table:
        rows = table.select("tr")
        print(f"rows: {len(rows)}")
    # 全テーブル確認
    all_tables = soup1.find_all("table")
    print(f"all tables: {len(all_tables)}")
    for i, t in enumerate(all_tables[:5]):
        print(f"  [{i}] class={t.get('class')} rows={len(t.find_all('tr'))}")
    horse_links = soup1.select("a[href*='/horse/']")
    print(f"horse links: {len(horse_links)}")
    if horse_links:
        print(f"  first: {horse_links[0].get('href')} -> {horse_links[0].get_text(strip=True)}")
else:
    print("NG: got None")

print()

# 2. shutuba ページ（レース終了後だと空かも）
print("=== shutuba page (Playwright, longer wait) ===")
url2 = f"https://race.netkeiba.com/race/shutuba.html?race_id={RACE_ID}"
soup2 = get_with_browser(url2, timeout_ms=20000)  # wait_selectorなし、5秒待機
if soup2:
    print(f"title: {soup2.title.string if soup2.title else 'none'}")
    rows = soup2.select("tr.HorseList")
    print(f"tr.HorseList: {len(rows)}")
    horse_links = soup2.select("a[href*='/horse/']")
    print(f"horse links: {len(horse_links)}")
else:
    print("NG: got None")
