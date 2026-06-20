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
        ev_threshold_override: dict = None,
    ) -> dict:
        builder = FeatureBuilder(self.db)
        df = builder.build()
        if df.empty:
            return {}

        df["date"] = pd.to_datetime(df["date"])

        # 2022年以降はスクレイピングデータ（特徴量品質が異なる）
        # Kaggleデータ期間内（〜2021年末）のみでバックテスト
        KAGGLE_END = pd.Timestamp("2021-12-31")
        df_kaggle = df[df["date"] <= KAGGLE_END].copy()

        cutoff = df_kaggle["date"].max() - pd.DateOffset(years=test_years)
        df_test = df_kaggle[df_kaggle["date"] >= cutoff].copy()

        print(f"[INFO] Kaggleデータ最大日付: {df_kaggle['date'].max().date()}")
        print(f"[INFO] カットオフ: {cutoff.date()}")
        print(f"[INFO] テスト期間: {df_test['date'].min().date()} 〜 {df_test['date'].max().date()} ({len(df_test)}行)")
        print(f"[INFO] テストレース数: {df_test['race_id'].nunique()}")

        # テスト期間より前のKaggleデータで学習（時系列リーク防止）
        df_train = df_kaggle[df_kaggle["date"] < cutoff].copy()
        print(f"[INFO] 学習データ: {df_train['date'].min().date()} 〜 {df_train['date'].max().date()} ({len(df_train)}行)")

        win_model   = RacePredictor("is_win")
        place_model = RacePredictor("is_place")
        score_model = RacePredictor("is_place", no_odds=True)
        print("[INFO] カットオフ前データでモデルを学習中...")
        win_model.train(df_train)
        place_model.train(df_train)
        score_model.train(df_train)
        print("[INFO] 学習完了")

        report = self.run_strategy_comparison(
            df_test, win_model, place_model, score_model, budget_per_race, ev_threshold_override
        )
        return report

    def run_strategy_comparison(
        self,
        df_test: pd.DataFrame,
        win_model: "RacePredictor",
        place_model: "RacePredictor",
        score_model: "RacePredictor",
        budget: float,
        ev_threshold_override: dict = None,
    ) -> dict:
        strategies = {
            "D_top2_gap2.0_pop8": {"top_n": 2, "min_odds": 0, "score_gap": 2.0, "max_popularity": 8},
        }

        all_results = {}
        race_ids = df_test["race_id"].unique()

        for strat_name, cfg in strategies.items():
            records = []
            debug = {"total": 0, "no_odds": 0, "no_gap": 0, "no_bet": 0, "ok": 0}
            for race_id in tqdm(race_ids, desc=strat_name, leave=False):
                df_race = df_test[df_test["race_id"] == race_id].copy()
                if len(df_race) < 3:
                    continue
                debug["total"] += 1

                odds = self._get_odds_for_race(race_id)
                if not any(odds.values()):
                    odds = self._build_odds_from_df(df_race)
                if not odds.get("fukusho"):
                    debug["no_odds"] += 1
                    continue

                # AIスコア計算（no_oddsモデル → レース内正規化）
                ai_raw = score_model.predict_proba(df_race)
                total = ai_raw.sum()
                ai_win = ai_raw / total if total > 0 else ai_raw
                n = len(df_race)
                ai_pts = (ai_win * 100 * n).astype(int)

                df_race = df_race.copy()
                df_race["_ai_pts"] = ai_pts
                df_race_sorted = df_race.sort_values("_ai_pts", ascending=False).reset_index(drop=True)

                avg_pts = ai_pts.mean()
                top_pts = df_race_sorted["_ai_pts"].iloc[0]

                # スコアギャップ条件
                if cfg["score_gap"] > 0 and avg_pts > 0:
                    if top_pts / avg_pts < cfg["score_gap"]:
                        debug["no_gap"] += 1
                        continue

                top3_actual = set(df_race[df_race["finish_order"] <= 3]["horse_number"].tolist())
                winner = df_race[df_race["finish_order"] == 1]["horse_number"].values
                winner = winner[0] if len(winner) > 0 else None

                for rank in range(min(cfg["top_n"], len(df_race_sorted))):
                    row = df_race_sorted.iloc[rank]
                    hn = int(row["horse_number"])
                    fo = odds["fukusho"].get(hn, 0)
                    if fo <= 0:
                        continue
                    if fo < cfg["min_odds"]:
                        continue
                    pop_raw = row.get("popularity")
                    pop = int(pop_raw) if pop_raw and str(pop_raw) not in ("", "nan", "None") else 0
                    if pop > 0 and pop > cfg.get("max_popularity", 99):
                        continue

                    stake = 100
                    hit = hn in top3_actual
                    payout = stake * fo if hit else 0
                    debug["ok"] += 1
                    records.append({
                        "race_id": race_id,
                        "horse_number": hn,
                        "ai_pts": top_pts,
                        "fukusho_odds": fo,
                        "stake": stake,
                        "hit": int(hit),
                        "payout": payout,
                    })

            all_results[strat_name] = records
            print(f"\n[DEBUG {strat_name}] 総レース:{debug['total']} オッズなし:{debug['no_odds']} gap未満:{debug['no_gap']} ベット:{debug['ok']}")

        self._print_strategy_report(all_results, strategies)
        return all_results

    @staticmethod
    def _print_strategy_report(all_results: dict, strategies: dict):
        print("\n" + "="*75)
        print("バックテスト結果：D_top2_gap2.0_pop8")
        print("="*75)
        for name, records in all_results.items():
            if not records:
                print("データなし")
                continue
            df = pd.DataFrame(records)
            df["year"] = df["race_id"].str[:4]

            # 総合
            n = len(df)
            hits = df["hit"].sum()
            stake_total = df["stake"].sum()
            pay_total = df["payout"].sum()
            avg_odds = df["fukusho_odds"].mean()
            roi = pay_total / stake_total * 100 if stake_total > 0 else 0
            profit = pay_total - stake_total
            print(f"\n【総合】")
            print(f"  ベット数:  {n:,}回  /  的中: {int(hits):,}回  ({hits/n*100:.1f}%)")
            print(f"  平均オッズ: {avg_odds:.2f}倍")
            print(f"  投資合計:  ¥{int(stake_total):,}")
            print(f"  回収合計:  ¥{int(pay_total):,}")
            print(f"  回収率:    {roi:.1f}%")
            print(f"  収支:      ¥{int(profit):+,}")

            # 年別
            print(f"\n【年別内訳】")
            print(f"  {'年':>4} {'ベット':>7} {'的中率':>7} {'回収率':>7} {'収支':>12} {'平均オッズ':>10}")
            print(f"  " + "-"*55)
            for year, grp in df.groupby("year"):
                yn = len(grp)
                yh = grp["hit"].sum()
                ys = grp["stake"].sum()
                yp = grp["payout"].sum()
                yo = grp["fukusho_odds"].mean()
                yroi = yp / ys * 100 if ys > 0 else 0
                yprofit = yp - ys
                print(f"  {year:>4} {yn:>7,} {yh/yn*100:>6.1f}% {yroi:>6.1f}% ¥{int(yprofit):>+10,} {yo:>9.2f}倍")
        print("\n" + "="*75)

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
            # 複勝オッズ推定: べき乗式（実際の複勝オッズに近い近似）
            # 低オッズ馬は控えめに、高オッズ馬は減衰させる
            result["fukusho"][h] = max(1.1, round(wo ** 0.6 * 0.75, 1))
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
