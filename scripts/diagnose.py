"""
ROIが低い原因を診断するスクリプト。
データの品質を2014-2021(Kaggle)と2022-2024(スクレイピング)で比較する。
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/jra.db')

print("=" * 60)
print("【診断1】race_resultsの基本カバレッジ")
print("=" * 60)

q = """
SELECT
    substr(race_id,1,4) as year,
    COUNT(*) as total,
    SUM(CASE WHEN finish_order IS NOT NULL AND finish_order > 0 THEN 1 ELSE 0 END) as has_finish,
    SUM(CASE WHEN win_odds IS NOT NULL AND win_odds > 0 THEN 1 ELSE 0 END) as has_odds,
    SUM(CASE WHEN popularity IS NOT NULL AND popularity > 0 THEN 1 ELSE 0 END) as has_pop,
    SUM(CASE WHEN last_3f IS NOT NULL AND last_3f > 0 THEN 1 ELSE 0 END) as has_last3f,
    SUM(CASE WHEN horse_id IS NOT NULL AND horse_id != '' THEN 1 ELSE 0 END) as has_horse_id,
    SUM(CASE WHEN distance IS NOT NULL AND distance > 0 THEN 1 ELSE 0 END) as has_distance,
    SUM(CASE WHEN surface IS NOT NULL AND surface != '' THEN 1 ELSE 0 END) as has_surface,
    SUM(CASE WHEN track_condition IS NOT NULL AND track_condition != '' THEN 1 ELSE 0 END) as has_track
FROM race_results
WHERE substr(race_id,1,4) >= '2014'
GROUP BY year ORDER BY year
"""
df = pd.read_sql(q, conn)
df["horse_id%"] = (df["has_horse_id"] / df["total"] * 100).round(1)
df["last3f%"] = (df["has_last3f"] / df["total"] * 100).round(1)
df["odds%"] = (df["has_odds"] / df["total"] * 100).round(1)
df["pop%"] = (df["has_pop"] / df["total"] * 100).round(1)
df["dist%"] = (df["has_distance"] / df["total"] * 100).round(1)
df["surf%"] = (df["has_surface"] / df["total"] * 100).round(1)
df["track%"] = (df["has_track"] / df["total"] * 100).round(1)

print(f"{'年':>4} {'総行':>6} {'horse_id':>10} {'last3f':>8} {'odds':>6} {'pop':>6} {'dist':>6} {'surf':>6} {'track':>6}")
for _, r in df.iterrows():
    print(f"{r['year']:>4} {int(r['total']):>6,} {r['horse_id%']:>9.1f}% {r['last3f%']:>7.1f}% {r['odds%']:>5.1f}% {r['pop%']:>5.1f}% {r['dist%']:>5.1f}% {r['surf%']:>5.1f}% {r['track%']:>5.1f}%")

print()
print("=" * 60)
print("【診断2】fukushoオッズ（推定vs実際）")
print("=" * 60)

q2 = """
SELECT
    substr(rr.race_id,1,4) as year,
    COUNT(*) as n,
    AVG(rr.win_odds) as avg_win_odds,
    AVG(POWER(rr.win_odds, 0.6) * 0.75) as avg_est_fuk,
    AVG(CAST(o.odds AS REAL)) as avg_real_fuk
FROM race_results rr
LEFT JOIN odds_raw o ON o.race_id = rr.race_id
    AND o.bet_type = 'fukusho'
    AND CAST(o.combination AS INTEGER) = rr.horse_number
WHERE substr(rr.race_id,1,4) >= '2020'
  AND rr.win_odds IS NOT NULL AND rr.win_odds > 0
GROUP BY year ORDER BY year
"""
df2 = pd.read_sql(q2, conn)
print(f"{'年':>4} {'n':>7} {'平均単勝odds':>12} {'推定fukusho':>12} {'実際fukusho':>12} {'差':>8}")
for _, r in df2.iterrows():
    diff = (r['avg_real_fuk'] - r['avg_est_fuk']) if r['avg_real_fuk'] else None
    diff_str = f"{diff:+.2f}" if diff else "N/A"
    real_str = f"{r['avg_real_fuk']:.2f}" if r['avg_real_fuk'] else "N/A"
    print(f"{r['year']:>4} {int(r['n']):>7,} {r['avg_win_odds']:>12.2f} {r['avg_est_fuk']:>12.2f} {real_str:>12} {diff_str:>8}")

print()
print("=" * 60)
print("【診断3】horse_historyの状況")
print("=" * 60)
r = conn.execute("SELECT COUNT(*), COUNT(DISTINCT horse_id) FROM horse_history").fetchone()
print(f"  総行数: {r[0]:,}  ユニーク馬数: {r[1]:,}")

print()
print("=" * 60)
print("【診断4】2022-2024の馬の前走成績がどれだけ取れているか")
print("=" * 60)

q4 = """
SELECT
    substr(race_id,1,4) as year,
    COUNT(*) as total,
    SUM(CASE WHEN horse_id IS NOT NULL AND horse_id != '' THEN 1 ELSE 0 END) as has_id
FROM race_results
WHERE substr(race_id,1,4) >= '2022'
GROUP BY year ORDER BY year
"""
df4 = pd.read_sql(q4, conn)
for _, r in df4.iterrows():
    pct = r['has_id'] / r['total'] * 100
    print(f"  {r['year']}年: horse_id={pct:.1f}% → 前走成績が計算できる馬の割合")

print()
print("""
【診断まとめ・解釈ガイド】
- horse_id%が低い年 → その年の馬の前走成績(win_rate等)がNaN → モデルに情報が入らない
- last3f%が低い → 上がり3Fの特徴量がNaN → 能力判定ができない
- 推定fukusho vs 実際fukusho の差 → バックテストの信頼性
- horse_history=0 → 今後の収集で改善予定
""")

conn.close()
