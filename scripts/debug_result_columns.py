"""
race_table_01の実際のカラム構造を確認する。
last_3fとtrack_conditionのインデックスがずれていないか診断。
"""
import sys
sys.path.insert(0, '.')
from jra_predictor.scraper.race_result import RaceResultScraper

# 2022年の既知レースで確認
race_id = "202201010101"
s = RaceResultScraper()

print(f"レースページ取得: {race_id}")
soup = s._fetch_race_page(race_id)
if soup is None:
    print("ページ取得失敗")
    sys.exit(1)

# race_info情報確認
info = s._parse_race_info_from_soup(soup, race_id)
print("\n【race_info】")
for k, v in info.items():
    print(f"  {k}: {v}")

# テーブル構造確認
table = (
    soup.select_one("table.race_table_01")
    or soup.select_one("table.result_table_02")
    or next((t for t in soup.find_all("table") if t.select("a[href*='/horse/']")), None)
)
if table is None:
    print("\nテーブルが見つかりません")
    sys.exit(1)

rows = table.select("tr")
print(f"\n【テーブル構造】tr数: {len(rows)}")

# ヘッダー行
print("\nヘッダー:")
if rows:
    ths = rows[0].select("th")
    for i, th in enumerate(ths):
        print(f"  [{i}] {th.get_text(strip=True)}")

# 最初のデータ行
print("\n最初のデータ行のtd値:")
for tr in rows[1:]:
    tds = tr.select("td")
    if len(tds) >= 6 and tr.select_one("a[href*='/horse/']"):
        for i, td in enumerate(tds):
            print(f"  [{i}] {td.get_text(strip=True)}")
        break
