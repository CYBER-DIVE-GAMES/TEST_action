"""
バックテストエンジン
時系列を厳守した検証で、的中率・回収率を算出する
"""
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from jra_predictor.data import Database
from jra_predictor.features import FeatureBuilder
from jra_predictor.models import RacePredictor, ExpectedValueCalculator
from config.settings import TEST_YEARS

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(self, db: Database):
        self.db = db

    def run(
        self,
        test_years: int = TEST_YEARS,
        budget_per_race: float = 10000,
        tune_hyperparams: bool = False,
    ) -> dict:
        """
        過去N年をテスト期間として、前データで学習→テストデータで予測・集計

        Returns: 馬券種別の的中率・回収率レポート
        """
        builder = FeatureBuilder(self.db)
        df = builder.build()
        if df.empty:
            return {}

        df["date"] = pd.to_datetime(df["date"])
        cutoff = df["date"].max() - pd.DateOffset(years=test_years)

        df_train = df[df["date"] < cutoff].copy()
        df_test = df[df["date"] >= cutoff].copy()

        logger.info(f"Train: {len(df_train)} rows ({df_train['date'].min().date()} - {df_train['date'].max().date()})")
        logger.info(f"Test:  {len(df_test)} rows ({df_test['date'].min().date()} - {df_test['date'].max().date()})")

        # モデル学習
        win_model = RacePredictor("is_win")
        place_model = RacePredictor("is_place")
        win_model.train(df_train, tune_hyperparams=tune_hyperparams)
        place_model.train(df_train, tune_hyperparams=tune_hyperparams)
        win_model.save()
        place_model.save()

        # 評価
        win_eval = win_model.evaluate(df_test)
        place_eval = place_model.evaluate(df_test)
        logger.info(f"Win model  - AUC: {win_eval['auc']:.4f}, LogLoss: {win_eval['logloss']:.4f}")
        logger.info(f"Place model - AUC: {place_eval['auc']:.4f}, LogLoss: {place_eval['logloss']:.4f}")

        # バックテスト本体
        ev_calc = ExpectedValueCalculator(win_model, place_model)
        results = self._simulate(ev_calc, df_test, budget_per_race)

        report = self._calc_report(results)
        self._print_report(report)
        return report

    def _simulate(
        self,
        ev_calc: ExpectedValueCalculator,
        df_test: pd.DataFrame,
        budget: float,
    ) -> list[dict]:
        """テスト期間の全レースをシミュレート"""
        records = []
        race_ids = df_test["race_id"].unique()

        for race_id in tqdm(race_ids, desc="Backtest"):
            df_race = df_test[df_test["race_id"] == race_id]
            if len(df_race) < 3:
                continue

            # 実データ優先、なければdf推定値にフォールバック
            odds = self._get_odds_for_race(race_id)
            if not any(odds.values()):
                odds = self._build_odds_from_df(df_race)
            if not any(odds.values()):
                continue

            recs = ev_calc.recommend(df_race, odds, budget)
            if recs.empty:
                continue

            # 実際の結果を突き合わせ
            actual_results = df_race[df_race["finish_order"] <= 3][["horse_number", "finish_order"]]
            top3 = set(actual_results["horse_number"].tolist())
            winner = df_race[df_race["finish_order"] == 1]["horse_number"].values
            winner = winner[0] if len(winner) > 0 else None

            for _, row in recs.iterrows():
                hit = self._is_hit(row["bet_type"], row["combination"], top3, winner)
                payout = row["stake"] * row["odds"] if hit else 0
                records.append({
                    "race_id": race_id,
                    "bet_type": row["bet_type"],
                    "combination": row["combination"],
                    "stake": row["stake"],
                    "odds": row["odds"],
                    "expected_value": row["expected_value"],
                    "hit": int(hit),
                    "payout": payout,
                })

        return records

    @staticmethod
    def _build_odds_from_df(df_race: pd.DataFrame) -> dict:
        """race_resultsのwin_oddsから馬券オッズを構築（odds_rawがない場合のフォールバック）"""
        result = {"tan": {}, "fukusho": {}}
        if "win_odds" not in df_race.columns or "horse_number" not in df_race.columns:
            return result
        for _, row in df_race.iterrows():
            h = int(row["horse_number"])
            wo = float(row["win_odds"]) if pd.notna(row["win_odds"]) else np.nan
            if np.isnan(wo) or wo <= 0:
                continue
            result["tan"][h] = wo
            # 複勝オッズの簡易推定: 単勝オッズが低いほど複勝も低い
            result["fukusho"][h] = max(1.1, round(wo * 0.28 + 1.05, 1))
        return result

    def _get_odds_for_race(self, race_id: str) -> dict:
        """DBからオッズ取得・パース"""
        df_odds = self.db.read_table("odds_raw", f"race_id = '{race_id}'")
        result = {"fukusho": {}, "wide": {}, "umaren": {}, "sanrenpuku": {}}
        for _, row in df_odds.iterrows():
            bt = row["bet_type"]
            combo_str = row["combination"]
            odds = row["odds"]
            if bt not in result:
                continue
            try:
                if bt == "fukusho":
                    result[bt][int(combo_str)] = odds
                else:
                    combo = tuple(int(x) for x in combo_str.strip("()").split(", "))
                    result[bt][combo] = odds
            except Exception:
                pass
        return result

    @staticmethod
    def _is_hit(bet_type: str, combination: str, top3: set, winner) -> bool:
        parts = [int(x) for x in combination.split("-")]
        if bet_type == "単勝":
            return parts[0] == winner
        elif bet_type == "複勝":
            return parts[0] in top3
        elif bet_type == "ワイド":
            return parts[0] in top3 and parts[1] in top3
        elif bet_type == "馬連":
            return parts[0] in top3 and parts[1] in top3
        elif bet_type == "3連複":
            return all(p in top3 for p in parts)
        return False

    @staticmethod
    def _calc_report(records: list[dict]) -> dict:
        if not records:
            return {}
        df = pd.DataFrame(records)
        report = {}
        for bt in df["bet_type"].unique():
            sub = df[df["bet_type"] == bt]
            total_stake = sub["stake"].sum()
            total_payout = sub["payout"].sum()
            n_bets = len(sub)
            n_hits = sub["hit"].sum()
            report[bt] = {
                "n_bets": n_bets,
                "n_hits": int(n_hits),
                "hit_rate": round(n_hits / n_bets * 100, 1),
                "total_stake": int(total_stake),
                "total_payout": int(total_payout),
                "roi": round(total_payout / total_stake * 100, 1) if total_stake > 0 else 0,
                "profit": int(total_payout - total_stake),
                "avg_ev": round(sub["expected_value"].mean(), 3),
            }
        return report

    @staticmethod
    def _print_report(report: dict):
        print("\n" + "="*60)
        print("バックテスト結果")
        print("="*60)
        for bt, r in report.items():
            print(f"\n【{bt}】")
            print(f"  賭け回数:  {r['n_bets']:,}  / 的中: {r['n_hits']:,}  ({r['hit_rate']}%)")
            print(f"  投資合計:  ¥{r['total_stake']:,}")
            print(f"  回収合計:  ¥{r['total_payout']:,}")
            print(f"  回収率:    {r['roi']}%")
            print(f"  収支:      ¥{r['profit']:+,}")
            print(f"  平均期待値: {r['avg_ev']}")
        print("="*60)
