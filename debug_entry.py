import logging, warnings, time
import requests
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

RACE_ID = '202606140901'
# race_id format: YYYYMMDDCCRR
# 2026/06/14, course=09(Hanshin), race=01
date_str = RACE_ID[:8]   # 20260614
course   = RACE_ID[8:10] # 09
race_no  = RACE_ID[10:]  # 01

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
    'Accept-Language': 'ja,en-US;q=0.9',
}

sess = requests.Session()
sess.headers.update(HEADERS)

# JRA公式の出馬表ページを試す
# 例: https://www.jra.go.jp/race/result/2024/12/01/hanshin/01/index.html
# course code mapping
course_names = {
    '01':'sapporo','02':'hakodate','03':'fukushima','04':'niigata',
    '05':'tokyo','06':'nakayama','07':'chukyo','08':'kyoto',
    '09':'hanshin','10':'kokura'
}
course_name = course_names.get(course, '')
year = date_str[:4]
month = date_str[4:6]
day = date_str[6:8]

url = f'https://www.jra.go.jp/race/result/{year}/{month}/{day}/{course_name}/{race_no}/index.html'
print(f'=== JRA official: {url} ===')
try:
    r = sess.get(url, timeout=10)
    print(f'status: {r.status_code}')
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, 'lxml')
        print(f'title: {soup.title.string if soup.title else "none"}')
        tables = soup.find_all('table')
        print(f'tables: {len(tables)}')
        links = soup.select('a[href*="horse"]')
        print(f'horse links: {len(links)}')
        if links:
            print(f'first: {links[0].get("href")} -> {links[0].get_text(strip=True)}')
except Exception as e:
    print(f'error: {e}')

time.sleep(2)

# JRA公式の出馬表（shutuba）ページ
url2 = f'https://www.jra.go.jp/race/shutuba/{year}/{month}/{day}/{course_name}/{race_no}/index.html'
print(f'\n=== JRA shutuba: {url2} ===')
try:
    r2 = sess.get(url2, timeout=10)
    print(f'status: {r2.status_code}')
    if r2.status_code == 200:
        soup2 = BeautifulSoup(r2.text, 'lxml')
        print(f'title: {soup2.title.string if soup2.title else "none"}')
        tables2 = soup2.find_all('table')
        print(f'tables: {len(tables2)}')
except Exception as e:
    print(f'error: {e}')
