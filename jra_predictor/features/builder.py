"""
特徴量エンジニアリング
高精度予測に向けた包括的な特徴量を生成する
"""
import logging
import numpy as np
import pandas as pd
from jra_predictor.data import Database

logger = logging.getLogger(__name__)


class FeatureBuilder:
    def __init__(self, db: Database):
        self.db = db

    def build(self, since_date: str = None) -> pd.DataFrame:
        """全特徴量を結合したDataFrameを構築

        since_date: "2019-01-01" 形式で指定するとその日以降のデータのみ使用（高速化）
        """
        logger.info("Building features...")

        df_result = self.db.read_table("race_results")
        df_info = self.db.read_table("race_info")
        df_horse = self.db.read_table("horse_profile")

        # 日付フィルター（predict-url等で高速化するため）
        if since_date and not df_result.empty and "date" in df_result.columns:
            df_result = df_result[df_result["date"].astype(str) >= since_date]
            logger.info(f"Filtered to {since_date} onwards: {len(df_result)} rows")

        if df_result.empty:
            logger.error("No data in DB. Run data import first.")
            return pd.DataFrame()

        # race_infoが空でも動くようにする
        if not df_info.empty:
            # 重複カラムを避けるため、race_infoから追加する列だけ選ぶ
            info_extra = [c for c in df_info.columns
                          if c not in df_result.columns or c == "race_id"]
            df = df_result.merge(df_info[info_extra], on="race_id", how="left")
        else:
            df = df_result.copy()

        # horse_profileが空でも動くようにする
        if not df_horse.empty and "horse_id" in df.columns:
            horse_cols = ["horse_id"] + [c for c in ["sire","dam_sire","birth_date"]
                                          if c in df_horse.columns]
            df = df.merge(df_horse[horse_cols], on="horse_id", how="left")

        # dateカラムの確保（race_resultsかrace_infoどちらかにある）
        if "date" not in df.columns:
            if "date_x" in df.columns:
                df["date"] = df["date_x"]
            elif "date_y" in df.columns:
                df["date"] = df["date_y"]
            else:
                logger.error("dateカラムが見つかりません")
                return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["race_id"] = df["race_id"].fillna("").astype(str)
        df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce").fillna(0)
        df = df.sort_values(["date", "race_id", "horse_number"], na_position="last").reset_index(drop=True)

        df = self._add_basic_features(df)
        df = self._add_horse_form_features(df)
        df = self._add_jockey_features(df)
        df = self._add_trainer_features(df)
        df = self._add_course_features(df)
        df = self._add_pace_features(df)
        df = self._add_pedigree_features(df)
        df = self._add_weight_features(df)
        df = self._add_odds_features(df)

        # object型の数値列を強制変換（LightGBMはobjectを受け付けない）
        # 文字列列（horse_name等）はto_numericでNaNになるが特徴量には使わないため問題なし
        NON_NUMERIC = {"race_id", "horse_name", "horse_id", "jockey_name", "jockey_id",
                       "trainer_name", "trainer_id", "race_name", "course", "course_code",
                       "surface", "track_condition", "sex_age", "sex", "margin",
                       "passing_order", "distance_cat", "sire", "dam_sire", "birth_date",
                       "owner", "date"}
        for col in df.columns:
            if df[col].dtype == object and col not in NON_NUMERIC:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"Features built: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 性別・年齢
        df["sex"] = df["sex_age"].str.extract(r"([牡牝騸セ])")
        df["age"] = pd.to_numeric(df["sex_age"].str.extract(r"(\d+)")[0], errors="coerce")

        # 出走頭数（同一レース内）
        df["field_count"] = df.groupby("race_id")["horse_number"].transform("count")

        # 馬番・枠番の相対位置
        df["horse_number_ratio"] = df["horse_number"] / df["field_count"]
        df["frame_number_norm"] = df["frame_number"] / 8.0

        # 着順（目的変数）
        df["is_win"] = (df["finish_order"] == 1).astype(int)
        df["is_place"] = (df["finish_order"] <= 3).astype(int)

        return df

    def _add_horse_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬の近走成績・能力指標"""
        # 時系列で過去データを参照（当該レース以前のみ）
        df = df.sort_values(["horse_id", "date", "race_id"], na_position="last")

        for window in [3, 5, 10]:
            col_win = f"win_rate_{window}"
            col_place = f"place_rate_{window}"
            col_avg_pop = f"avg_popularity_{window}"
            col_avg_odds = f"avg_odds_{window}"

            df[col_win] = df.groupby("horse_id")["is_win"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[col_place] = df.groupby("horse_id")["is_place"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[col_avg_pop] = df.groupby("horse_id")["popularity"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[col_avg_odds] = df.groupby("horse_id")["win_odds"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # 前走着順・前々走着順
        df["prev_finish"] = df.groupby("horse_id")["finish_order"].shift(1)
        df["prev2_finish"] = df.groupby("horse_id")["finish_order"].shift(2)

        # 前走オッズ vs 今回オッズの変化
        df["prev_odds"] = df.groupby("horse_id")["win_odds"].shift(1)
        df["odds_change"] = df["win_odds"] - df["prev_odds"]

        # 上がり3F平均（能力指標）
        df["avg_last3f_5"] = df.groupby("horse_id")["last_3f"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        # 前走からの休養日数
        df["prev_date"] = df.groupby("horse_id")["date"].shift(1)
        df["days_since_last"] = (df["date"] - df["prev_date"]).dt.days

        # キャリア（出走回数）
        df["career_runs"] = df.groupby("horse_id").cumcount()

        return df

    def _add_jockey_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """騎手の成績特徴量"""
        df = df.sort_values(["jockey_id", "date"], na_position="last")

        for window in [30, 100]:
            df[f"jockey_win_rate_{window}"] = df.groupby("jockey_id")["is_win"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=5).mean()
            )
            df[f"jockey_place_rate_{window}"] = df.groupby("jockey_id")["is_place"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=5).mean()
            )

        # 騎手×コース別勝率
        df["jockey_course_wins"] = df.groupby(["jockey_id", "course_code"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # 騎手×距離帯別勝率
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce").fillna(1600)
        df["distance_cat"] = pd.cut(df["distance"], bins=[0, 1400, 1800, 2200, 9999],
                                     labels=["sprint", "mile", "middle", "long"])
        df["jockey_dist_wins"] = df.groupby(["jockey_id", "distance_cat"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        return df

    def _add_trainer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """調教師の成績特徴量"""
        df = df.sort_values(["trainer_id", "date"], na_position="last")

        df["trainer_win_rate_50"] = df.groupby("trainer_id")["is_win"].transform(
            lambda x: x.shift(1).rolling(50, min_periods=5).mean()
        )
        df["trainer_place_rate_50"] = df.groupby("trainer_id")["is_place"].transform(
            lambda x: x.shift(1).rolling(50, min_periods=5).mean()
        )

        return df

    def _add_course_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """コース・距離適性"""
        # 馬のコース別成績
        df["horse_course_wins"] = df.groupby(["horse_id", "course_code"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df["horse_course_place"] = df.groupby(["horse_id", "course_code"])["is_place"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # 馬の距離適性
        df["horse_dist_wins"] = df.groupby(["horse_id", "distance_cat"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        ) if "distance_cat" in df.columns else 0.0

        # 芝・ダート別適性
        df["horse_surface_wins"] = df.groupby(["horse_id", "surface"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        # 馬場状態別（良・稍重・重・不良）
        df["horse_condition_wins"] = df.groupby(["horse_id", "track_condition"])["is_win"].transform(
            lambda x: x.shift(1).expanding().mean()
        )

        return df

    def _add_pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ペース・展開予測特徴量（レース前に判明している情報のみ使用）"""
        # 通過順位から先行・差し・追い込みタイプを推定（過去レースの結果）
        def estimate_running_style(passing):
            if not isinstance(passing, str):
                return np.nan
            positions = [int(x) for x in passing.split("-") if x.strip().isdigit()]
            if not positions:
                return np.nan
            avg_pos = np.mean(positions)
            if avg_pos <= 3:
                return 1  # 先行
            elif avg_pos <= 6:
                return 2  # 差し
            else:
                return 3  # 追い込み

        df["running_style"] = df["passing_order"].apply(estimate_running_style)

        # 馬の脚質傾向（過去履歴のみ参照: shift(1)）
        df["avg_running_style"] = df.groupby("horse_id")["running_style"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        # ※ last3f_rank と n_frontrunners は当レース結果に依存するため除外

        return df

    def _add_pedigree_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """血統特徴量"""
        if "sire" not in df.columns:
            return df

        # 父別勝率
        df["sire_win_rate"] = df.groupby("sire")["is_win"].transform(
            lambda x: x.shift(1).expanding(min_periods=10).mean()
        )
        df["sire_place_rate"] = df.groupby("sire")["is_place"].transform(
            lambda x: x.shift(1).expanding(min_periods=10).mean()
        )

        # 父×距離適性
        df["sire_dist_win_rate"] = df.groupby(["sire", "distance_cat"])["is_win"].transform(
            lambda x: x.shift(1).expanding(min_periods=5).mean()
        ) if "distance_cat" in df.columns else 0.0

        return df

    def _add_weight_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬体重・斤量特徴量"""
        # 馬体重トレンド（直近3走平均との差）
        df["avg_weight_3"] = df.groupby("horse_id")["horse_weight"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df["weight_vs_avg"] = df["horse_weight"] - df["avg_weight_3"]

        # 斤量（重いほど不利）
        df["weight_handicap"] = df.groupby("race_id")["weight_carried"].transform(
            lambda x: x - x.mean()
        )

        return df

    def _add_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """オッズ・市場評価特徴量"""
        # 人気の相対位置（同一レース内）
        df["popularity_norm"] = df.groupby("race_id")["popularity"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        )

        # 1番人気馬のオッズ（レース全体の質指標）
        df["fav_odds"] = df[df["popularity"] == 1].groupby("race_id")["win_odds"].transform("first")
        df["fav_odds"] = df.groupby("race_id")["fav_odds"].transform("first")

        # 自分のオッズ / 1番人気オッズ
        df["relative_odds"] = df["win_odds"] / (df["fav_odds"] + 1e-9)

        return df
