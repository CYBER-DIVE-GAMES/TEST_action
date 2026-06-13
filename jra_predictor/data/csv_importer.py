"""
CSVファイルからデータをインポートする
Kaggleや他のソースから取得したCSVをDBに取り込む
"""
import logging
import pandas as pd
from pathlib import Path
from .database import Database

logger = logging.getLogger(__name__)


class CsvImporter:
    """
    対応フォーマット:
    - netkeiba形式のCSV（race_results, race_info）
    - カスタムCSV（カラムマッピングで対応）
    """

    def __init__(self, db: Database):
        self.db = db

    def import_race_results(self, csv_path: str):
        """レース結果CSVをインポート"""
        path = Path(csv_path)
        if not path.exists():
            logger.error(f"ファイルが見つかりません: {csv_path}")
            return

        logger.info(f"インポート開始: {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        logger.info(f"  {len(df)} 行を読み込みました")
        logger.info(f"  カラム: {list(df.columns)}")

        # カラム名を正規化
        df = self._normalize_columns(df)

        if "race_id" not in df.columns:
            logger.error("race_id カラムが必要です")
            return

        # レース情報を抽出してDBに保存
        if "date" in df.columns:
            race_info_cols = ["race_id", "race_name", "date", "course",
                              "course_code", "race_number", "surface",
                              "distance", "weather", "track_condition"]
            info_cols = [c for c in race_info_cols if c in df.columns]
            df_info = df[["race_id"] + [c for c in info_cols if c != "race_id"]].drop_duplicates("race_id")
            for _, row in df_info.iterrows():
                self.db.upsert_race_info(row.to_dict())
            logger.info(f"  レース情報: {len(df_info)} 件")

        # レース結果をDBに保存
        self.db.upsert_race_results(df)
        logger.info(f"  レース結果: {len(df)} 行 インポート完了")

    def import_folder(self, folder_path: str):
        """フォルダ内の全CSVをインポート"""
        folder = Path(folder_path)
        csv_files = list(folder.glob("*.csv"))
        logger.info(f"{len(csv_files)} 個のCSVファイルを検出")

        for i, csv_file in enumerate(csv_files, 1):
            logger.info(f"[{i}/{len(csv_files)}] {csv_file.name}")
            self.import_race_results(str(csv_file))

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """様々なCSVのカラム名を統一する"""
        rename_map = {
            # 日本語カラム名
            "着順": "finish_order",
            "枠番": "frame_number",
            "馬番": "horse_number",
            "馬名": "horse_name",
            "性齢": "sex_age",
            "斤量": "weight_carried",
            "騎手": "jockey_name",
            "タイム": "finish_time_sec",
            "着差": "margin",
            "通過": "passing_order",
            "上り": "last_3f",
            "単勝": "win_odds",
            "人気": "popularity",
            "馬体重": "horse_weight",
            "調教師": "trainer_name",
            "賞金": "prize",
            "レースID": "race_id",
            "開催日": "date",
            "競馬場": "course",
            "レース名": "race_name",
            "距離": "distance",
            "馬場": "track_condition",
            "天候": "weather",
            # 英語別名
            "race_date": "date",
            "venue": "course",
            "odds": "win_odds",
        }
        return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
