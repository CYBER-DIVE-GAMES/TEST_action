from jra_predictor.scraper.base import BaseScaper
s = BaseScaper()
soup = s.get('https://db.netkeiba.com/horse/2019104988/')
if soup is None:
    print('ページ取得失敗')
else:
    t1 = soup.select_one('table.db_h_race_results')
    print('table.db_h_race_results:', t1 is not None)
    tables = soup.select('table')
    print('テーブル数:', len(tables))
    for i, t in enumerate(tables[:5]):
        print('  table[' + str(i) + '] class=' + str(t.get('class')))
    if t1 is not None:
        rows = t1.select('tr')
        print('tr数:', len(rows))
        for tr in rows[:3]:
            tds = tr.select('td')
            print('  td数:', len(tds), '/ 最初のtd:', tds[0].get_text(strip=True) if tds else '(なし)')
