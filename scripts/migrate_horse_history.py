import sqlite3

conn = sqlite3.connect('data/jra.db')
cur = conn.cursor()

existing = [r[1] for r in cur.execute("PRAGMA table_info(horse_history)").fetchall()]
print("既存カラム:", existing)

to_add = {
    "race_number": "TEXT",
    "popularity": "INTEGER",
    "weight_carried": "REAL",
    "horse_weight_diff": "INTEGER",
    "prize": "REAL",
}

for col, typ in to_add.items():
    if col not in existing:
        cur.execute(f"ALTER TABLE horse_history ADD COLUMN {col} {typ}")
        print(f"追加: {col}")
    else:
        print(f"スキップ（既存）: {col}")

conn.commit()
conn.close()
print("完了")
