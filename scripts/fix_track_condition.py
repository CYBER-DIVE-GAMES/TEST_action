"""
Step 1: 2022-2024のtrack_conditionをrace_infoページから再取得して補完する。
レース結果・オッズ・horse_historyは触らない。
"""
import sys
import time
sys.path.insert(0, '.')

import sqlite3
from jra_predictor.scraper.race_result import RaceResultScraper

conn = sqlite3.connect('data/jra.db')
cur = conn.cursor()

# track_conditionが空の2022-2024レースIDを取得
race_ids = [r[0] for r in cur.execute("""
    SELECT DISTINCT race_id FROM race_results
    WHERE substr(race_id,1,4) >= '2022'
      AND (track_condition IS NULL OR track_condition = '')
    ORDER BY race_id
""").fetchall()]

print(f"track_condition未取得レース数: {len(race_ids):,}")

scraper = RaceResultScraper()
ok = 0
ng = 0

for i, race_id in enumerate(race_ids):
    try:
        info = scraper.fetch_race_info(race_id)
        tc = info.get("track_condition", "") if info else ""
        if tc:
            cur.execute("""
                UPDATE race_results SET track_condition = ?
                WHERE race_id = ?
            """, (tc, race_id))
            conn.commit()
            ok += 1
        else:
            ng += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(race_ids)} 完了 (成功:{ok} 失敗:{ng})")

    except Exception as e:
        ng += 1
        print(f"  エラー {race_id}: {e}")

print(f"\n完了: 成功={ok} 失敗={ng}")
conn.close()
