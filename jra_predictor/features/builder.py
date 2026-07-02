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
        df = self._add_class_features(df)
        df = self._add_change_features(df)

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
        """馬の近走成績・能力指標（horse_historyを優先使用、なければrace_resultsにフォールバック）"""
        df_hist = self.db.read_table("horse_history")

        if not df_hist.empty and "horse_id" in df_hist.columns and "finish_order" in df_hist.columns:
            logger.info(f"horse_historyから過去成績を計算: {len(df_hist)}行")
            df_hist = df_hist.copy()
            df_hist["date"] = pd.to_datetime(df_hist["date"], errors="coerce")
            df_hist = df_hist.dropna(subset=["horse_id", "date"])
            df_hist["finish_order"] = pd.to_numeric(df_hist["finish_order"], errors="coerce")
            df_hist["popularity"] = pd.to_numeric(df_hist["popularity"], errors="coerce")
            df_hist["odds"] = pd.to_numeric(df_hist["odds"], errors="coerce")
            df_hist["last_3f"] = pd.to_numeric(df_hist["last_3f"], errors="coerce")
            df_hist["is_win"] = (df_hist["finish_order"] == 1).astype(float)
            df_hist["is_place"] = (df_hist["finish_order"] <= 3).astype(float)
            df_hist = df_hist.sort_values(["horse_id", "date"]).reset_index(drop=True)

            stat_cols = {}
            for window in [3, 5, 10]:
                df_hist[f"win_rate_{window}"] = df_hist.groupby("horse_id")["is_win"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                df_hist[f"place_rate_{window}"] = df_hist.groupby("horse_id")["is_place"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                df_hist[f"avg_popularity_{window}"] = df_hist.groupby("horse_id")["popularity"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                df_hist[f"avg_odds_{window}"] = df_hist.groupby("horse_id")["odds"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                stat_cols.update({f"win_rate_{window}", f"place_rate_{window}",
                                   f"avg_popularity_{window}", f"avg_odds_{window}"})

            df_hist["prev_finish"] = df_hist.groupby("horse_id")["finish_order"].shift(1)
            df_hist["prev2_finish"] = df_hist.groupby("horse_id")["finish_order"].shift(2)
            df_hist["prev_odds"] = df_hist.groupby("horse_id")["odds"].shift(1)
            df_hist["avg_last3f_5"] = df_hist.groupby("horse_id")["last_3f"].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )
            df_hist["prev_date_h"] = df_hist.groupby("horse_id")["date"].shift(1)
            df_hist["days_since_last"] = (df_hist["date"] - df_hist["prev_date_h"]).dt.days
            df_hist["career_runs"] = df_hist.groupby("horse_id").cumcount()

            merge_cols = ["horse_id", "date"] + [
                "win_rate_3", "win_rate_5", "win_rate_10",
                "place_rate_3", "place_rate_5", "place_rate_10",
                "avg_popularity_3", "avg_popularity_5", "avg_popularity_10",
                "avg_odds_3", "avg_odds_5", "avg_odds_10",
                "prev_finish", "prev2_finish", "prev_odds",
                "avg_last3f_5", "days_since_last", "career_runs",
            ]
            df_hist_latest = df_hist[merge_cols].drop_duplicates(subset=["horse_id", "date"], keep="last")

            df = df.sort_values("date")
            df_hist_latest = df_hist_latest.sort_values("date")
            df = pd.merge_asof(
                df,
                df_hist_latest,
                on="date",
                by="horse_id",
                direction="backward",
                suffixes=("", "_h"),
            )
            # horse_historyにない馬はNaN（後続でLightGBMがNaN扱い）
            df["odds_change"] = df["win_odds"] - df.get("prev_odds", pd.Series(dtype=float))

        else:
            logger.info("horse_historyが空のためrace_resultsから過去成績を計算")
            df = df.sort_values(["horse_id", "date", "race_id"], na_position="last")

            for window in [3, 5, 10]:
                df[f"win_rate_{window}"] = df.groupby("horse_id")["is_win"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                df[f"place_rate_{window}"] = df.groupby("horse_id")["is_place"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                df[f"avg_popularity_{window}"] = df.groupby("horse_id")["popularity"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                df[f"avg_odds_{window}"] = df.groupby("horse_id")["win_odds"].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

            df["prev_finish"] = df.groupby("horse_id")["finish_order"].shift(1)
            df["prev2_finish"] = df.groupby("horse_id")["finish_order"].shift(2)
            df["prev_odds"] = df.groupby("horse_id")["win_odds"].shift(1)
            df["odds_change"] = df["win_odds"] - df["prev_odds"]
            df["avg_last3f_5"] = df.groupby("horse_id")["last_3f"].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )
            df["prev_date"] = df.groupby("horse_id")["date"].shift(1)
            df["days_since_last"] = (df["date"] - df["prev_date"]).dt.days
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

    def _add_class_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """レースクラス・頭数補正特徴量"""
        # レース名からクラスを数値化
        def race_class(name):
            if not isinstance(name, str):
                return 5
            if any(k in name for k in ["GI", "G1", "有馬", "天皇賞", "ジャパン", "宝塚", "安田"]):
                return 1
            if any(k in name for k in ["GII", "G2"]):
                return 2
            if any(k in name for k in ["GIII", "G3"]):
                return 3
            if any(k in name for k in ["(L)", "リステッド"]):
                return 4
            if "OP" in name or "オープン" in name:
                return 5
            if "3勝" in name or "1600万" in name:
                return 6
            if "2勝" in name or "1000万" in name:
                return 7
            if "1勝" in name or "500万" in name:
                return 8
            if "未勝利" in name:
                return 9
            if "新馬" in name or "メイクデビュー" in name:
                return 10
            return 5

        df["race_class"] = df["race_name"].apply(race_class)

        # 前走クラス
        df["prev_race_class"] = df.groupby("horse_id")["race_class"].shift(1)

        # クラス変化（正=昇級, 負=降級）
        df["class_change"] = df["prev_race_class"] - df["race_class"]

        # 前走クラス × 前走着順の複合指標（降級+好走 = 狙い目）
        df["class_finish_index"] = df.apply(
            lambda r: r["class_change"] * (6 - min(r["prev_finish"], 6))
            if pd.notna(r.get("class_change")) and pd.notna(r.get("prev_finish")) else float("nan"), axis=1
        )

        # 頭数補正：複勝基準確率（3/出走頭数）
        df["place_base_rate"] = 3.0 / df["field_count"].clip(lower=4)

        # 市場複勝確率（人気から逆算: 1/odds のスケール）
        # モデル確率との乖離を後で計算するための基準値
        df["is_shinsoba"] = df["race_name"].apply(
            lambda n: 1 if isinstance(n, str) and ("新馬" in n or "メイクデビュー" in n) else 0
        )

        return df

    def _add_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """前走からの変化特徴量"""
        df = df.sort_values(["horse_id", "date", "race_id"], na_position="last")

        # 前走距離
        df["prev_distance"] = df.groupby("horse_id")["distance"].shift(1)
        # 距離変化（正=距離延長, 負=距離短縮）
        df["distance_change"] = pd.to_numeric(df["distance"], errors="coerce") - pd.to_numeric(df["prev_distance"], errors="coerce")

        # 前走芝ダート
        df["prev_surface"] = df.groupby("horse_id")["surface"].shift(1)
        # 芝ダート変更フラグ（0=変更なし, 1=変更あり）
        df["surface_change"] = (df["surface"] != df["prev_surface"]).astype(float)
        df.loc[df["prev_surface"].isna(), "surface_change"] = float("nan")

        # 前走コース
        df["prev_course_code"] = df.groupby("horse_id")["course_code"].shift(1)
        # コース変更フラグ
        df["course_change"] = (df["course_code"] != df["prev_course_code"]).astype(float)
        df.loc[df["prev_course_code"].isna(), "course_change"] = float("nan")

        # ペース想定：同レース内の先行馬（running_style==1）の数
        if "running_style" in df.columns:
            df["n_frontrunners"] = df.groupby("race_id")["running_style"].transform(
                lambda x: (x == 1).sum()
            )
            # 自分が先行馬かつ前走者が多い = 厳しい展開
            df["pace_pressure"] = df.apply(
                lambda r: r["n_frontrunners"] - 1 if r.get("running_style") == 1 else 0, axis=1
            )
        else:
            df["n_frontrunners"] = float("nan")
            df["pace_pressure"] = float("nan")

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
