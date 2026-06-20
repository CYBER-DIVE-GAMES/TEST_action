"""
odds_rawテーブルのtanshoデータをrace_results.win_oddsに書き戻す
2022年以降のスクレイピングデータに対して実行
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from jra_predictor.data import Database

db = Database()
with db.engine.connect() as conn:
    result = conn.execute(text("""
        UPDATE race_results
        SET win_odds = (
            SELECT CAST(o.odds AS REAL)
            FROM odds_raw o
            WHERE o.race_id = race_results.race_id
              AND o.bet_type = 'tansho'
              AND CAST(o.combination AS INTEGER) = race_results.horse_number
            LIMIT 1
        )
        WHERE win_odds IS NULL OR win_odds = 0
    """))
    conn.commit()
    print(f"更新件数: {result.rowcount}行")

# 確認
with db.engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT substr(race_id,1,4) as year,
               COUNT(*) as total,
               SUM(CASE WHEN win_odds > 0 THEN 1 ELSE 0 END) as has_odds
        FROM race_results
        WHERE substr(race_id,1,4) >= '2022'
        GROUP BY year ORDER BY year
    """)).fetchall()
    for r in rows:
        print(r)
