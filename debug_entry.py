import logging, warnings, time
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
from jra_predictor.scraper import RaceResultScraper
from config.settings import NETKEIBA_RACE

RACE_ID = '202606140901'

s = RaceResultScraper()
s._init_session()

# 1. 内部APIを試す
url = NETKEIBA_RACE + '/api/api_get_race_info.html'
r = s.session.get(url, params={'race_id': RACE_ID, 'rf': 'shutuba_past'}, timeout=15)
print('=== API ===')
print('status:', r.status_code)
print('response (first 400):', r.text[:400])

time.sleep(2)

# 2. shutuba_past.htmlを試す
print('\n=== shutuba_past.html ===')
soup = s.get(NETKEIBA_RACE + '/race/shutuba_past.html', params={'race_id': RACE_ID})
if soup:
    tables = soup.find_all('table')
    print('tables found:', len(tables))
    for i, t in enumerate(tables[:3]):
        rows = t.find_all('tr')
        print(f'  [{i}] class={t.get("class")} rows={len(rows)}')
    links = soup.select('a[href*="/horse/"]')
    print('horse links:', len(links))
    if links:
        print('first horse link:', links[0].get('href'), '->', links[0].get_text(strip=True))
else:
    print('got None')
