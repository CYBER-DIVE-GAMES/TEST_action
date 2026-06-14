"""
Kaggleデータのインポートとモデル学習を一括実行
"""
import sys
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from jra_predictor.data import Database
from jra_predictor.data.kaggle_importer import KaggleJraImporter
from jra_predictor.features import FeatureBuilder
from jra_predictor.models import RacePredictor

DATA_DIR = r"C:\JRAtool\data"
START_YEAR = 2010  # 2010年以降に絞る（約11年）

print("=" * 60)
print("Step 1: データインポート開始")
print("=" * 60)

db = Database()
imp = KaggleJraImporter(db)

import pandas as pd
csv_path = f"{DATA_DIR}/19860105-20210731_race_result.csv"

print(f"CSVを読み込み中（{START_YEAR}年以降のみ）...")
chunks = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, chunksize=10000)

total = 0
skipped = 0
for i, chunk in enumerate(chunks):
    # 年フィルタ
    if "レース日付" in chunk.columns:
        chunk = chunk[chunk["レース日付"].str[:4].astype(str, errors='ignore') >= str(START_YEAR)]
    if chunk.empty:
        skipped += 1
        continue
    df = imp._convert_result(chunk)
    imp._save_race_info(df)
    db.upsert_race_results(df)
    total += len(df)
    if i % 10 == 0:
        print(f"  {total:,} 行処理済み...")

print(f"\nインポート完了: {total:,} 行")

print("\n" + "=" * 60)
print("Step 2: 特徴量構築")
print("=" * 60)

builder = FeatureBuilder(db)
df_features = builder.build()
print(f"特徴量構築完了: {len(df_features):,} 行, {len(df_features.columns)} 特徴量")

# 過去データのみ（finish_orderがNULLでない = 完了したレース）
df_train = df_features[df_features["finish_order"].notna()].copy()
print(f"訓練データ: {len(df_train):,} 行 ({df_train['date'].min()} 〜 {df_train['date'].max()})")

print("\n" + "=" * 60)
print("Step 3: モデル学習（is_win）")
print("=" * 60)
win_model = RacePredictor("is_win")
win_model.train(df_train)
win_model.save()

print("\n" + "=" * 60)
print("Step 4: モデル学習（is_place / 複勝）")
print("=" * 60)
place_model = RacePredictor("is_place")
place_model.train(df_train)
place_model.save()

print("\n" + "=" * 60)
print("✅ 完了！ web/app.py を再起動してください")
print("=" * 60)
