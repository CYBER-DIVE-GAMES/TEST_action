"""
2022年以降のrace_infoのdistance/surfaceが欠損しているレースを再スクレイピングして補完する。
distance=NULLのrace_idをrace_resultsから取得し、netkeibaから再取得してrace_info/race_resultsを更新。
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from jra_predictor.data import Database
from jra_predictor.scraper import RaceResultScraper

db = Database()
scraper = RaceResultScraper()

# distance=NULLのrace_idを取得（重複除外）
with db.engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT DISTINCT race_id FROM race_results
        WHERE distance IS NULL AND substr(race_id,1,4) >= '2022'
        ORDER BY race_id
    """)).fetchall()

race_ids = [r[0] for r in rows]
print(f"再スクレイピング対象: {len(race_ids)}レース")

ok = 0
fail = 0
for i, race_id in enumerate(race_ids):
    info = scraper.fetch_race_info(race_id)
    if not info:
        fail += 1
        continue

    if not info.get("distance"):
        fail += 1
        continue

    with db.engine.connect() as conn:
        # race_infoを更新
        conn.execute(text("""
            INSERT OR REPLACE INTO race_info
            (race_id, race_name, date, course, course_code, race_number,
             surface, distance, weather, track_condition)
            VALUES (:race_id, :race_name, :date, :course, :course_code, :race_number,
                    :surface, :distance, :weather, :track_condition)
        """), {
            "race_id": race_id,
            "race_name": info.get("race_name", ""),
            "date": info.get("date", ""),
            "course": info.get("course", ""),
            "course_code": info.get("course_code", ""),
            "race_number": info.get("race_number", 0),
            "surface": info.get("surface", ""),
            "distance": info.get("distance"),
            "weather": info.get("weather", ""),
            "track_condition": info.get("track_condition", ""),
        })
        # race_resultsも更新
        for field in ["distance", "surface", "weather", "track_condition", "race_name", "date"]:
            if info.get(field):
                conn.execute(text(f"""
                    UPDATE race_results SET {field} = :{field}
                    WHERE race_id = :race_id AND ({field} IS NULL OR {field} = '')
                """), {field: info[field], "race_id": race_id})
        conn.commit()
    ok += 1

    if (i + 1) % 50 == 0:
        print(f"  進捗: {i+1}/{len(race_ids)} (成功:{ok} 失敗:{fail})")

    time.sleep(1.0)  # サーバー負荷軽減

print(f"\n完了: 成功={ok}, 失敗={fail}")

# 最終確認
with db.engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT substr(race_id,1,4) as year,
               COUNT(*) as total,
               SUM(CASE WHEN distance IS NOT NULL THEN 1 ELSE 0 END) as has_dist
        FROM race_results
        WHERE substr(race_id,1,4) >= '2022'
        GROUP BY year ORDER BY year
    """)).fetchall()
    print("\n【補完後のdistance確認】")
    for r in rows:
        pct = r[2] / r[1] * 100 if r[1] > 0 else 0
        print(f"  {r[0]}年: {r[2]}/{r[1]} ({pct:.1f}%)")
