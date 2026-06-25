"""
2022年以降のrace_resultsのpopularityがNULLになっている問題を修正。
odds_rawのtanshoオッズから人気順位（オッズ昇順のランク）を計算して補完する。
"""
import sqlite3

conn = sqlite3.connect('data/jra.db')
cur = conn.cursor()

print("odds_rawのtanshoから人気順位を計算してrace_resultsに補完中...")

cur.execute("""
    UPDATE race_results
    SET popularity = (
        SELECT ranked.rnk FROM (
            SELECT
                race_id,
                CAST(combination AS INTEGER) AS horse_number,
                RANK() OVER (PARTITION BY race_id ORDER BY odds ASC) AS rnk
            FROM odds_raw
            WHERE bet_type = 'tansho'
        ) ranked
        WHERE ranked.race_id = race_results.race_id
          AND ranked.horse_number = race_results.horse_number
    )
    WHERE (popularity IS NULL OR popularity = 0)
      AND substr(race_id, 1, 4) >= '2022'
""")
conn.commit()
print(f"更新件数: {cur.rowcount}行")

# 確認
r = cur.execute("""
    SELECT
        substr(race_id,1,4) as year,
        COUNT(*) as total,
        SUM(CASE WHEN popularity > 0 THEN 1 ELSE 0 END) as has_pop
    FROM race_results
    WHERE substr(race_id,1,4) >= '2022'
    GROUP BY year ORDER BY year
""").fetchall()

print("\n【補完後のpopularity確認】")
for row in r:
    pct = row[2] / row[1] * 100 if row[1] > 0 else 0
    print(f"  {row[0]}年: {row[2]}/{row[1]} ({pct:.1f}%)")

conn.close()
print("\n完了！")
