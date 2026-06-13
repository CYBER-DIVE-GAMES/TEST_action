"""
LightGBMによる着順予測モデル
勝率・複勝率を推定し、期待値計算の基盤を提供する
"""
import logging
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
from config.settings import RANDOM_SEED, CV_FOLDS, MODEL_DIR

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 使用する特徴量カラム
FEATURE_COLS = [
    # 基本情報
    "age", "horse_number_ratio", "frame_number_norm", "field_count",
    "weight_carried", "weight_handicap",
    # 近走成績
    "win_rate_3", "win_rate_5", "win_rate_10",
    "place_rate_3", "place_rate_5", "place_rate_10",
    "avg_popularity_3", "avg_popularity_5",
    "avg_odds_5", "odds_change",
    "prev_finish", "prev2_finish",
    "avg_last3f_5",
    "days_since_last", "career_runs",
    # 騎手
    "jockey_win_rate_30", "jockey_win_rate_100",
    "jockey_place_rate_30", "jockey_place_rate_100",
    "jockey_course_wins", "jockey_dist_wins",
    # 調教師
    "trainer_win_rate_50", "trainer_place_rate_50",
    # コース・距離適性
    "horse_course_wins", "horse_course_place",
    "horse_surface_wins", "horse_condition_wins",
    # ペース・脚質（avg_running_styleはshift(1)で過去履歴のみ参照）
    "avg_running_style",
    # 血統
    "sire_win_rate", "sire_place_rate", "sire_dist_win_rate",
    # 馬体重
    "weight_vs_avg", "horse_weight",
    # オッズ
    "popularity_norm", "relative_odds", "fav_odds",
    # レース情報
    "distance",
]


class RacePredictor:
    def __init__(self, target: str = "is_win"):
        """
        target: 'is_win' または 'is_place'
        """
        self.target = target
        self.model = None
        self.feature_importance_ = None
        self.model_path = MODEL_DIR / f"lgbm_{target}.pkl"

    def train(self, df: pd.DataFrame, tune_hyperparams: bool = True):
        """時系列交差検証でモデルを学習"""
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[feature_cols].fillna(-999)
        y = df[self.target]

        # 時系列でデータ分割（未来データでテスト）
        df_sorted = df.sort_values("date")
        X = X.loc[df_sorted.index]
        y = y.loc[df_sorted.index]

        if tune_hyperparams:
            params = self._tune(X, y)
        else:
            params = self._default_params()

        # 全データで最終モデルを学習
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=params.pop("n_estimators", 500),
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            valid_sets=[train_data],
        )

        self.feature_importance_ = pd.Series(
            self.model.feature_importance(importance_type="gain"),
            index=feature_cols,
        ).sort_values(ascending=False)

        logger.info(f"Model trained ({self.target}). "
                    f"Top features: {list(self.feature_importance_.head(5).index)}")

    def _tune(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Optunaによるハイパーパラメータ最適化"""
        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

        def objective(trial):
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "seed": RANDOM_SEED,
            }
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model = lgb.train(
                    params,
                    lgb.Dataset(X_tr, label=y_tr),
                    num_boost_round=300,
                    valid_sets=[lgb.Dataset(X_val, label=y_val)],
                    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
                )
                pred = model.predict(X_val)
                scores.append(log_loss(y_val, pred))
            return np.mean(scores)

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=50, show_progress_bar=False)

        best = study.best_params
        best.update({
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "seed": RANDOM_SEED,
            "n_estimators": 1000,
        })
        logger.info(f"Best params ({self.target}): {study.best_params}")
        return best

    def _default_params(self) -> dict:
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 63,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "seed": RANDOM_SEED,
            "n_estimators": 500,
        }

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """各馬の確率を予測"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[feature_cols].fillna(-999)
        return self.model.predict(X)

    def evaluate(self, df: pd.DataFrame) -> dict:
        """AUC・LogLoss を計算"""
        proba = self.predict_proba(df)
        y = df[self.target]
        return {
            "auc": roc_auc_score(y, proba),
            "logloss": log_loss(y, proba),
            "n_samples": len(y),
        }

    def save(self):
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load(self):
        self.model = joblib.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")
