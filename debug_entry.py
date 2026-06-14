import logging, warnings, time
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
from jra_predictor.scraper import RaceResultScraper
from config.settings import NETKEIBA_RACE

RACE_ID = '202606140901'

s = RaceResultScraper()

# オッズページ（type=b1）から馬情報を取得できるか確認
print('=== Win/Place Odds page (type=b1) ===')
soup = s.get(NETKEIBA_RACE + '/odds/index.html', params={'race_id': RACE_ID, 'type': 'b1'})
if soup:
    rows = soup.select('tr.HorseList')
    print(f'HorseList rows: {len(rows)}')
    if rows:
        first = rows[0]
        tds = first.select('td')
        print(f'  cols in first row: {len(tds)}')
        for i, td in enumerate(tds):
            print(f'  td[{i}]: "{td.get_text(strip=True)[:30]}" class={td.get("class")}')
        # 馬リンク
        links = first.select('a[href*="/horse/"]')
        print(f'  horse links in row: {len(links)}')
        if links:
            print(f'  -> {links[0].get("href")} {links[0].get_text(strip=True)}')
        # 騎手リンク
        jlinks = first.select('a[href*="/jockey/"]')
        print(f'  jockey links in row: {len(jlinks)}')
        if jlinks:
            print(f'  -> {jlinks[0].get("href")} {jlinks[0].get_text(strip=True)}')
else:
    print('got None')
