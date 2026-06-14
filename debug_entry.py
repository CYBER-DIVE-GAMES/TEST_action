"""Playwrightでnetkeibaのshutubaページを取得できるか確認"""
import logging
logging.basicConfig(level=logging.INFO)

from jra_predictor.scraper.base import get_with_browser
from config.settings import NETKEIBA_RACE

RACE_ID = '202606140901'
url = f"{NETKEIBA_RACE}/race/shutuba.html?race_id={RACE_ID}"

print(f"Playwright取得: {url}")
soup = get_with_browser(url, wait_selector="tr.HorseList", timeout_ms=20000)

if soup is None:
    print("ERROR: got None")
else:
    print(f"title: {soup.title.string if soup.title else 'none'}")
    rows = soup.select("tr.HorseList")
    print(f"tr.HorseList rows: {len(rows)}")
    if rows:
        tds = rows[0].select("td")
        print(f"  first row cols: {len(tds)}")
        for i, td in enumerate(tds[:8]):
            print(f"  td[{i}]: '{td.get_text(strip=True)[:20]}'")
        links = rows[0].select("a[href*='/horse/']")
        print(f"  horse links: {len(links)}")
        if links:
            print(f"  -> {links[0].get('href')} {links[0].get_text(strip=True)}")
    else:
        # 全テーブル確認
        tables = soup.find_all("table")
        print(f"tables found: {len(tables)}")
        for i, t in enumerate(tables[:5]):
            rows2 = t.find_all("tr")
            print(f"  [{i}] class={t.get('class')} rows={len(rows2)}")
        horse_links = soup.select("a[href*='/horse/']")
        print(f"horse links total: {len(horse_links)}")
        if horse_links:
            print(f"  first: {horse_links[0].get('href')} {horse_links[0].get_text(strip=True)}")
