"""shutuba_past.html を Playwright で取得テスト"""
import logging
logging.basicConfig(level=logging.INFO)

from jra_predictor.scraper.base import get_with_browser

RACE_ID = '202606140901'  # 今日の完了済みレース

print("=== shutuba_past.html (Playwright) ===")
url = f"https://race.netkeiba.com/race/shutuba_past.html?race_id={RACE_ID}"
soup = get_with_browser(url, timeout_ms=20000)

if soup:
    print(f"title: {soup.title.string if soup.title else 'none'}")
    rows = soup.select("tr.HorseList")
    print(f"tr.HorseList rows: {len(rows)}")

    all_tables = soup.find_all("table")
    print(f"all tables: {len(all_tables)}")
    for i, t in enumerate(all_tables[:5]):
        trs = t.find_all("tr")
        print(f"  [{i}] class={t.get('class')} rows={len(trs)}")
        if len(trs) > 1:
            tds = trs[1].find_all("td")
            if tds:
                print(f"       first data row cells={len(tds)}: {[c.get_text(strip=True)[:12] for c in tds[:6]]}")

    horse_links = soup.select("a[href*='/horse/']")
    print(f"horse links: {len(horse_links)}")
    if horse_links:
        print(f"  first: {horse_links[0].get('href')} -> {horse_links[0].get_text(strip=True)}")
else:
    print("NG: got None")
