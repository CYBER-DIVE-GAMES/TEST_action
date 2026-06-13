"""
JRA競馬中央競走データセット専用インポーター
Kaggleデータセット形式に対応
"""
import logging
import pandas as pd
from pathlib import Path
from .database import Database

logger = logging.getLogger(__name__)


class KaggleJraImporter:

    def __init__(self, db: Database):
        self.db = db

    def import_all(self, folder_path: str):
        """フォルダ内の全CSVを自動検出してインポート"""
        folder = Path(folder_path)
        files = list(folder.glob("*.csv"))
        logger.info(f"{len(files)} 個のCSVを検出: {[f.name for f in files]}")

        result_file = next((f for f in files if "race_result" in f.name), None)
        odds_file   = next((f for f in files if "odds" in f.name), None)

        if result_file:
            self.import_race_results(str(result_file))
        if odds_file:
            self.import_odds(str(odds_file))

        logger.info("全インポート完了")

    def import_race_results(self, csv_path: str):
        """race_result.csv をインポート"""
        logger.info(f"レース結果インポート開始: {csv_path}")

        # 大きいファイルなのでチャンク読み込み
        chunks = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            dtype=str,
            chunksize=10000,
        )

        total = 0
        for i, chunk in enumerate(chunks):
            df = self._convert_result(chunk)
            self._save_race_info(df)
            self.db.upsert_race_results(df)
            total += len(df)
            if i % 5 == 0:
                logger.info(f"  {total:,} 行処理済み...")

        logger.info(f"レース結果インポート完了: {total:,} 行")

    def _convert_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """カラム名を内部形式に変換"""
        rename = {
            "レースID":     "race_id",
            "レース日付":   "date",
            "競馬場コード": "course_code",
            "競馬場名":     "course",
            "レース番号":   "race_number",
            "レース名":     "race_name",
            "距離(m)":      "distance",
            "芝・ダート区分": "surface",
            "天候":         "weather",
            "馬場状態1":    "track_condition",
            "着順":         "finish_order",
            "枠番":         "frame_number",
            "馬番":         "horse_number",
            "馬名":         "horse_name",
            "性別":         "sex",
            "馬齢":         "age",
            "斤量":         "weight_carried",
            "騎手":         "jockey_name",
            "タイム":       "finish_time_raw",
            "着差":         "margin",
            "1コーナー":    "corner1",
            "2コーナー":    "corner2",
            "3コーナー":    "corner3",
            "4コーナー":    "corner4",
            "上り":         "last_3f",
            "単勝":         "win_odds",
            "人気":         "popularity",
            "馬体重":       "horse_weight",
            "場体重増減":   "horse_weight_diff",
            "調教師":       "trainer_name",
            "馬主":         "owner",
            "賞金(万円)":   "prize",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # 性別+馬齢を結合
        if "sex" in df.columns and "age" in df.columns:
            df["sex_age"] = df["sex"].fillna("") + df["age"].fillna("")

        # タイムを秒数に変換 (1:23.4 → 83.4)
        if "finish_time_raw" in df.columns:
            df["finish_time_sec"] = df["finish_time_raw"].apply(self._parse_time)

        # 通過順をまとめる
        corner_cols = [c for c in ["corner1","corner2","corner3","corner4"] if c in df.columns]
        if corner_cols:
            df["passing_order"] = df[corner_cols].apply(
                lambda r: "-".join(str(v) for v in r if pd.notna(v) and str(v).strip()), axis=1
            )

        # 数値変換
        for col in ["finish_order","frame_number","horse_number","horse_weight",
                    "horse_weight_diff","popularity","race_number","distance","age"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ["weight_carried","last_3f","win_odds","prize"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 着順が数値以外（中止・除外など）は除外
        if "finish_order" in df.columns:
            df = df[pd.to_numeric(df["finish_order"], errors="coerce").notna()]

        # 芝・ダート変換
        if "surface" in df.columns:
            df["surface"] = df["surface"].map({"芝": "芝", "ダ": "ダート", "ダート": "ダート"}).fillna(df["surface"])

        # 勝敗フラグ
        df["is_win"]   = (df["finish_order"] == 1).astype(int)
        df["is_place"] = (df["finish_order"] <= 3).astype(int)

        return df

    def _save_race_info(self, df: pd.DataFrame):
        """レース基本情報をrace_infoテーブルに保存"""
        info_cols = ["race_id","date","course","course_code","race_number",
                     "race_name","distance","surface","weather","track_condition"]
        cols = [c for c in info_cols if c in df.columns]
        df_info = df[cols].drop_duplicates("race_id")
        for _, row in df_info.iterrows():
            try:
                self.db.upsert_race_info(row.dropna().to_dict())
            except Exception:
                pass

    def import_odds(self, csv_path: str):
        """odds.csv をインポート"""
        logger.info(f"オッズインポート開始: {csv_path}")
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, nrows=5)
            logger.info(f"  オッズカラム: {list(df.columns)}")
            # カラム確認後に本実装
        except Exception as e:
            logger.warning(f"オッズインポートスキップ: {e}")

    @staticmethod
    def _parse_time(s: str):
        """1:23.4 → 83.4 秒に変換"""
        import re
        if not isinstance(s, str):
            return None
        m = re.match(r"(\d+):(\d+\.\d+)", s.strip())
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        try:
            return float(s)
        except (ValueError, TypeError):
            return None
