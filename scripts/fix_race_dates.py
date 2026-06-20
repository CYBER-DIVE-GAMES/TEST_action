"""
race_results.dateが空のレコードをrace_infoから補完する
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
        SET date = (
            SELECT date FROM race_info
            WHERE race_info.race_id = race_results.race_id
        )
        WHERE (date IS NULL OR date = '')
          AND substr(race_id,1,4) >= '2022'
    """))
    conn.commit()
    print(f"更新件数: {result.rowcount}行")

# 確認
with db.engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT substr(race_id,1,4) as year,
               COUNT(*) as total,
               SUM(CASE WHEN date IS NOT NULL AND date != '' THEN 1 ELSE 0 END) as has_date
        FROM race_results
        WHERE substr(race_id,1,4) >= '2022'
        GROUP BY year ORDER BY year
    """)).fetchall()
    for r in rows:
        print(r)
