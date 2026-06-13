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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from config.settings import RANDOM_SEED, CV_FOLDS, MODEL_DIR

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    "age", "horse_number_ratio", "frame_number_norm", "field_count",
    "weight_carried", "weight_handicap",
    "win_rate_3", "win_rate_5", "win_rate_10",
    "place_rate_3", "place_rate_5", "place_rate_10",
    "avg_popularity_3", "avg_popularity_5",
    "avg_odds_5", "odds_change",
    "prev_finish", "prev2_finish",
    "avg_last3f_5",
    "days_since_last", "career_runs",
    "jockey_win_rate_30", "jockey_win_rate_100",
    "jockey_place_rate_30", "jockey_place_rate_100",
    "jockey_course_wins", "jockey_dist_wins",
    "trainer_win_rate_50", "trainer_place_rate_50",
    "horse_course_wins", "horse_course_place",
    "horse_surface_wins", "horse_condition_wins",
    "avg_running_style",
    "sire_win_rate", "sire_place_rate", "sire_dist_win_rate",
    "weight_vs_avg", "horse_weight",
    "popularity_norm", "relative_odds", "fav_odds",
    "distance",
]


class RacePredictor:
    def __init__(self, target: str = "is_win"):
        self.target = target
        self.model = None
        self.calibrator = None
        self.feature_importance_ = None
        self.model_path = MODEL_DIR / f"lgbm_{target}.pkl"
        self.calibrator_path = MODEL_DIR / f"calibrator_{target}.pkl"

    def train(self, df: pd.DataFrame, tune_hyperparams: bool = False):
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]

        df_sorted = df.sort_values("date").reset_index(drop=True)
        X = df_sorted[feature_cols].fillna(-999)
        y = df_sorted[self.target]

        n = len(df_sorted)
        calib_start = int(n * 0.80)
        X_train, X_calib = X.iloc[:calib_start], X.iloc[calib_start:]
        y_train, y_calib = y.iloc[:calib_start], y.iloc[calib_start:]

        if tune_hyperparams:
            params = self._tune(X_train, y_train)
        else:
            params = self._default_params()

        n_estimators = params.pop("n_estimators", 500)
        train_data = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            valid_sets=[train_data],
        )

        raw_proba = self.model.predict(X_calib)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_proba, y_calib)

        brier_before = brier_score_loss(y_calib, raw_proba)
        calib_proba = self.calibrator.transform(raw_proba)
        brier_after = brier_score_loss(y_calib, calib_proba)
        logger.info(
            f"Calibration ({self.target}): "
            f"Brier {brier_before:.4f} -> {brier_after:.4f} "
            f"({'improved' if brier_after < brier_before else 'worsened'})"
        )

        self.feature_importance_ = pd.Series(
            self.model.feature_importance(importance_type="gain"),
            index=feature_cols,
        ).sort_values(ascending=False)

        logger.info(f"Model trained ({self.target}). "
                    f"Top features: {list(self.feature_importance_.head(5).index)}")

    def _tune(self, X: pd.DataFrame, y: pd.Series) -> dict:
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
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[feature_cols].fillna(-999)
        raw = self.model.predict(X)
        if self.calibrator is not None:
            return self.calibrator.transform(raw)
        return raw

    def evaluate(self, df: pd.DataFrame) -> dict:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[feature_cols].fillna(-999)
        y = df[self.target]
        raw = self.model.predict(X)
        calib = self.calibrator.transform(raw) if self.calibrator else raw
        return {
            "auc": roc_auc_score(y, calib),
            "logloss": log_loss(y, calib),
            "brier": brier_score_loss(y, calib),
            "n_samples": len(y),
        }

    def save(self):
        joblib.dump(self.model, self.model_path)
        if self.calibrator is not None:
            joblib.dump(self.calibrator, self.calibrator_path)
        logger.info(f"Model saved to {self.model_path}")

    def load(self):
        self.model = joblib.load(self.model_path)
        if self.calibrator_path.exists():
            self.calibrator = joblib.load(self.calibrator_path)
        logger.info(f"Model loaded from {self.model_path}")
