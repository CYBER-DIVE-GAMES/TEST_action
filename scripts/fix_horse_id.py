"""
Kaggleデータ（~2021年）のhorse_idがNULLになっている問題を修正。
スクレイピングデータ（2022-2024）の馬名→horse_idのマッピングを使って
Kaggleデータ行にhorse_idを補完し、rolling statsが正しく計算されるようにする。
"""
import sqlite3

conn = sqlite3.connect('data/jra.db')
cur = conn.cursor()

print("馬名→horse_idのマッピングを構築中...")

# スクレイピングデータから馬名→horse_idのマッピングを作成
cur.execute("""
    CREATE TEMP TABLE horse_name_id AS
    SELECT horse_name, horse_id
    FROM race_results
    WHERE horse_id IS NOT NULL AND horse_id != ''
      AND substr(race_id,1,4) >= '2022'
    GROUP BY horse_name
    HAVING COUNT(DISTINCT horse_id) = 1
""")

mapping_count = cur.execute("SELECT COUNT(*) FROM horse_name_id").fetchone()[0]
print(f"  マッピング構築: {mapping_count}頭")

# Kaggleデータのhorse_idを補完
cur.execute("""
    UPDATE race_results
    SET horse_id = (
        SELECT horse_id FROM horse_name_id
        WHERE horse_name_id.horse_name = race_results.horse_name
    )
    WHERE (horse_id IS NULL OR horse_id = '')
      AND substr(race_id,1,4) <= '2021'
      AND EXISTS (
        SELECT 1 FROM horse_name_id
        WHERE horse_name_id.horse_name = race_results.horse_name
      )
""")
conn.commit()
updated = cur.rowcount
print(f"  Kaggleデータhorse_id補完: {updated}行")

# 確認
rows = cur.execute("""
    SELECT
        substr(race_id,1,4) as year,
        COUNT(*) as total,
        SUM(CASE WHEN horse_id IS NOT NULL AND horse_id != '' THEN 1 ELSE 0 END) as has_id
    FROM race_results
    WHERE substr(race_id,1,4) >= '2014'
    GROUP BY year ORDER BY year
""").fetchall()

print("\n【horse_id補完後の確認】")
for r in rows:
    pct = r[2] / r[1] * 100 if r[1] > 0 else 0
    print(f"  {r[0]}年: {r[2]}/{r[1]} ({pct:.1f}%)")

conn.close()
print("\n完了！次: python main.py backtest")
