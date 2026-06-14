"""結果ページとPlaywrightの動作確認"""
import logging
logging.basicConfig(level=logging.INFO)

from jra_predictor.scraper import RaceResultScraper

s = RaceResultScraper()

# 1. 完了済みレース → db.netkeiba.com 結果ページ（requests）
RACE_ID = '202606140901'
print(f"=== fetch_race_result({RACE_ID}) ===")
df = s.fetch_race_result(RACE_ID)
if df is not None and not df.empty:
    print(f"OK: {len(df)} horses")
    print(df[['horse_number','horse_name','horse_id','finish_order']].head())
else:
    print("NG: no data")

# 2. Playwrightが実際に動くかシンプルに確認
print("\n=== Playwright basic test ===")
from jra_predictor.scraper.base import get_with_browser
soup = get_with_browser("https://www.google.com", timeout_ms=10000)
if soup:
    print(f"OK: got page, title={soup.title.string if soup.title else 'none'}")
else:
    print("NG: playwright failed")
