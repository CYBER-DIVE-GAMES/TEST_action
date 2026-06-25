import sqlite3
conn = sqlite3.connect('data/jra.db')
cur = conn.cursor()

r1 = cur.execute("SELECT COUNT(*) FROM odds_raw WHERE bet_type='fukusho' AND substr(race_id,1,4) >= '2022'").fetchone()
r2 = cur.execute("SELECT COUNT(*) FROM odds_raw WHERE bet_type='tansho' AND substr(race_id,1,4) >= '2022'").fetchone()
r3 = cur.execute("SELECT COUNT(DISTINCT race_id) FROM race_results WHERE substr(race_id,1,4) >= '2022'").fetchone()
r4 = cur.execute("SELECT COUNT(*) FROM race_results WHERE substr(race_id,1,4) >= '2022' AND popularity IS NOT NULL AND popularity > 0").fetchone()
r5 = cur.execute("SELECT COUNT(*) FROM race_results WHERE substr(race_id,1,4) >= '2022'").fetchone()
print('2022+レース数:', r3[0])
print('2022+ tanshoオッズ件数:', r2[0])
print('2022+ fukushoオッズ件数:', r1[0])
print('2022+ popularity有効行:', r4[0], '/', r5[0])

# 2021以前のfukusho確認
r6 = cur.execute("SELECT COUNT(*) FROM odds_raw WHERE bet_type='fukusho' AND substr(race_id,1,4) <= '2021'").fetchone()
print('2021以前のfukushoオッズ件数:', r6[0])

# horse_history確認
r7 = cur.execute("SELECT COUNT(*) FROM horse_history").fetchone()
r8 = cur.execute("SELECT COUNT(DISTINCT horse_id) FROM horse_history").fetchone()
print('horse_history総行数:', r7[0])
print('horse_history馬数:', r8[0])

# 2022-2024の馬がhorse_historyに存在するか
r9 = cur.execute("""
    SELECT COUNT(DISTINCT rr.horse_id)
    FROM race_results rr
    WHERE substr(rr.race_id,1,4) >= '2022'
      AND rr.horse_id IS NOT NULL AND rr.horse_id != ''
      AND EXISTS (SELECT 1 FROM horse_history hh WHERE hh.horse_id = rr.horse_id)
""").fetchone()
r10 = cur.execute("""
    SELECT COUNT(DISTINCT horse_id)
    FROM race_results
    WHERE substr(race_id,1,4) >= '2022'
      AND horse_id IS NOT NULL AND horse_id != ''
""").fetchone()
print('2022+の馬のうちhorse_historyあり:', r9[0], '/', r10[0])

conn.close()
