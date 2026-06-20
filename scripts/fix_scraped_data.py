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

# まず実態診断
print("=== 診断 ===")
with db.engine.connect() as conn:
    # horse_historyに2022年のrace_idが存在するか？
    r = conn.execute(text("""
        SELECT COUNT(*) as cnt,
               COUNT(distance) as has_dist,
               COUNT(surface) as has_surf
        FROM horse_history
        WHERE substr(race_id,1,4) >= '2022'
    """)).fetchone()
    print(f"horse_history 2022+: 総行={r[0]}, distance有={r[1]}, surface有={r[2]}")

    # race_infoに2022年のdistanceが存在するか？
    r2 = conn.execute(text("""
        SELECT COUNT(*) as cnt, COUNT(distance) as has_dist
        FROM race_info WHERE substr(race_id,1,4) >= '2022'
    """)).fetchone()
    print(f"race_info 2022+: 総行={r2[0]}, distance有={r2[1]}")

    # race_resultsで実際にdistanceがNULLの行数
    r3 = conn.execute(text("""
        SELECT COUNT(*) FROM race_results
        WHERE distance IS NULL AND substr(race_id,1,4) >= '2022'
    """)).fetchone()
    print(f"race_results 2022+ distance=NULL: {r3[0]}行")

    # horse_historyとrace_resultsのrace_idが一致するか確認
    r4 = conn.execute(text("""
        SELECT COUNT(*) FROM race_results rr
        JOIN horse_history hh ON hh.race_id = rr.race_id
        WHERE rr.distance IS NULL AND substr(rr.race_id,1,4) >= '2022'
          AND hh.distance IS NOT NULL
    """)).fetchone()
    print(f"horse_historyでdistance補完できる行: {r4[0]}行")

    # horse_historyのrace_idサンプル
    samples = conn.execute(text("""
        SELECT DISTINCT race_id FROM horse_history
        WHERE substr(race_id,1,4) >= '2022' LIMIT 5
    """)).fetchall()
    print(f"horse_history race_idサンプル: {[r[0] for r in samples]}")

    # race_resultsのrace_idサンプル
    samples2 = conn.execute(text("""
        SELECT DISTINCT race_id FROM race_results
        WHERE substr(race_id,1,4) >= '2022' LIMIT 5
    """)).fetchall()
    print(f"race_results race_idサンプル: {[r[0] for r in samples2]}")


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
