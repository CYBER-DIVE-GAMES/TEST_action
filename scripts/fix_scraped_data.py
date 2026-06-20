"""
2022年以降のスクレイピングデータに対して、
race_infoテーブルのフィールドをrace_resultsに補完する。

Kaggleデータ（〜2021年）はすでにrace_resultsに全フィールドが入っているが、
Netkeibaスクレイピングデータはrace_infoに別保存されているため、
race_resultsのdate/course/distance等がNULLになっている。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from jra_predictor.data import Database

db = Database()

print("race_infoからrace_resultsへフィールドを補完中...")

# race_infoから補完できるフィールド
RACE_INFO_FIELDS = ["date", "course", "course_code", "race_number",
                    "race_name", "distance", "surface", "weather", "track_condition"]

with db.engine.connect() as conn:
    for field in RACE_INFO_FIELDS:
        result = conn.execute(text(f"""
            UPDATE race_results
            SET {field} = (
                SELECT {field} FROM race_info
                WHERE race_info.race_id = race_results.race_id
            )
            WHERE (race_results.{field} IS NULL OR race_results.{field} = '')
              AND substr(race_results.race_id, 1, 4) >= '2022'
        """))
        conn.commit()
        print(f"  {field}: {result.rowcount}行更新")

# race_infoにdistance/surfaceがない場合はhorse_historyから補完
print("\nhorse_historyからdistance/surfaceを補完中...")
with db.engine.connect() as conn:
    for field in ["distance", "surface"]:
        result = conn.execute(text(f"""
            UPDATE race_results
            SET {field} = (
                SELECT {field} FROM horse_history
                WHERE horse_history.race_id = race_results.race_id
                  AND horse_history.{field} IS NOT NULL
                LIMIT 1
            )
            WHERE (race_results.{field} IS NULL OR race_results.{field} = '')
              AND substr(race_results.race_id, 1, 4) >= '2022'
        """))
        conn.commit()
        print(f"  {field}: {result.rowcount}行更新")

# 確認
print("\n【補完後の確認】")
with db.engine.connect() as conn:
    for field in ["date", "course", "distance", "surface"]:
        rows = conn.execute(text(f"""
            SELECT substr(race_id,1,4) as year,
                   COUNT(*) as total,
                   SUM(CASE WHEN {field} IS NOT NULL AND {field} != '' THEN 1 ELSE 0 END) as has_val
            FROM race_results
            WHERE substr(race_id,1,4) >= '2022'
            GROUP BY year ORDER BY year
        """)).fetchall()
        print(f"\n  [{field}]")
        for r in rows:
            pct = r[2] / r[1] * 100 if r[1] > 0 else 0
            print(f"    {r[0]}年: {r[2]}/{r[1]} ({pct:.1f}%)")

print("\n完了！")
