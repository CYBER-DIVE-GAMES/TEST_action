"""
Step 2: 2022-2024に出走した馬のhorse_historyのみを収集する。
レース結果・オッズは再取得しない。
horse_historyにはlast_3fも含まれるため、race_resultsのlast_3f問題も解決できる。
"""
import sys
import time
sys.path.insert(0, '.')

import sqlite3
from jra_predictor.scraper.horse_profile import HorseProfileScraper
from jra_predictor.data.database import Database

# 取得対象: 2022-2024に出走した馬（horse_historyに未登録のもの）
conn = sqlite3.connect('data/jra.db')
cur = conn.cursor()

horse_ids = [r[0] for r in cur.execute("""
    SELECT DISTINCT rr.horse_id
    FROM race_results rr
    WHERE substr(rr.race_id,1,4) >= '2022'
      AND rr.horse_id IS NOT NULL AND rr.horse_id != ''
      AND NOT EXISTS (
        SELECT 1 FROM horse_history hh WHERE hh.horse_id = rr.horse_id
      )
    ORDER BY rr.horse_id
""").fetchall()]

print(f"horse_history未取得馬数: {len(horse_ids):,}")
conn.close()

db = Database()
scraper = HorseProfileScraper()
ok = 0
ng = 0

for i, horse_id in enumerate(horse_ids):
    try:
        df = scraper.fetch_horse_history(horse_id)
        if df is not None and len(df) > 0:
            db.upsert_horse_history(df)
            ok += 1
        else:
            ng += 1

        if (i + 1) % 50 == 0:
            pct = (i + 1) / len(horse_ids) * 100
            print(f"  {i+1}/{len(horse_ids)} ({pct:.1f}%) 成功:{ok} 失敗:{ng}")

    except Exception as e:
        ng += 1
        if (i + 1) % 50 == 0:
            print(f"  エラー {horse_id}: {e}")

print(f"\n完了: 成功={ok} 失敗={ng}")
