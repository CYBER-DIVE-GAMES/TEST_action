"""
期待値計算・馬券推奨エンジン

核心ロジック:
  期待値 = モデル推定確率 × オッズ
  期待値 > 閾値 の馬券のみ推奨

複勝・ワイド・馬連・3連複に対応
"""
import logging
import itertools
import numpy as np
import pandas as pd
from config.settings import EV_THRESHOLD, KELLY_FRACTION

logger = logging.getLogger(__name__)


class ExpectedValueCalculator:

    def __init__(self, win_predictor, place_predictor):
        self.win_pred = win_predictor
        self.place_pred = place_predictor

    def recommend(
        self,
        df_race: pd.DataFrame,
        odds_dict: dict,
        budget: float = 10000,
    ) -> pd.DataFrame:
        """
        1レース分の推奨馬券を返す

        df_race: 1レースの特徴量DataFrame（複数行）
        odds_dict: {"fukusho": {馬番: odds}, "umaren": {(i,j): odds}, ...}
        budget: 予算（円）

        Returns: 推奨馬券DataFrame
        """
        win_prob = self.win_pred.predict_proba(df_race)
        place_prob = self.place_pred.predict_proba(df_race)

        horse_nums = df_race["horse_number"].values
        prob_map = {n: {"win": w, "place": p}
                    for n, w, p in zip(horse_nums, win_prob, place_prob)}

        recommendations = []

        # ---- 単勝 ----
        if "tan" in odds_dict and odds_dict["tan"]:
            for horse_num, horse_odds in odds_dict["tan"].items():
                if horse_num not in prob_map:
                    continue
                p = prob_map[horse_num]["win"]
                ev = p * horse_odds
                if ev >= EV_THRESHOLD.get("tan", 1.20):
                    stake = self._kelly_stake(p, horse_odds, budget)
                    recommendations.append({
                        "bet_type": "単勝",
                        "combination": str(horse_num),
                        "probability": round(p, 4),
                        "odds": horse_odds,
                        "expected_value": round(ev, 3),
                        "stake": int(stake),
                    })

        # ---- 複勝 ----
        if "fukusho" in odds_dict:
            for horse_num, horse_odds in odds_dict["fukusho"].items():
                if horse_num not in prob_map:
                    continue
                p = prob_map[horse_num]["place"]
                # 複勝は最小オッズで計算（保守的）
                min_odds = horse_odds if isinstance(horse_odds, float) else horse_odds.get("place_odds_min", 1.0)
                ev = p * min_odds
                if ev >= EV_THRESHOLD["fukusho"]:
                    stake = self._kelly_stake(p, min_odds, budget)
                    recommendations.append({
                        "bet_type": "複勝",
                        "combination": str(horse_num),
                        "probability": round(p, 4),
                        "odds": min_odds,
                        "expected_value": round(ev, 3),
                        "stake": int(stake),
                    })

        # ---- ワイド ----
        if "wide" in odds_dict:
            for (i, j), horse_odds in odds_dict["wide"].items():
                if i not in prob_map or j not in prob_map:
                    continue
                # ワイド的中確率: どちらも3着以内に入る確率（簡易計算）
                p_i = prob_map[i]["place"]
                p_j = prob_map[j]["place"]
                n = len(horse_nums)
                # 相関を考慮した近似
                p_wide = p_i * p_j * (n / (n - 1)) * 0.5
                ev = p_wide * horse_odds
                if ev >= EV_THRESHOLD["wide"]:
                    stake = self._kelly_stake(p_wide, horse_odds, budget)
                    recommendations.append({
                        "bet_type": "ワイド",
                        "combination": f"{i}-{j}",
                        "probability": round(p_wide, 4),
                        "odds": horse_odds,
                        "expected_value": round(ev, 3),
                        "stake": int(stake),
                    })

        # ---- 馬連 ----
        if "umaren" in odds_dict:
            for (i, j), horse_odds in odds_dict["umaren"].items():
                if i not in prob_map or j not in prob_map:
                    continue
                p_i_win = prob_map[i]["win"]
                p_j_win = prob_map[j]["win"]
                # 馬連確率 ≈ P(iが1着) * P(jが2着|iが1着) + P(jが1着) * P(iが2着|jが1着)
                n = len(horse_nums)
                place_others_i = [prob_map[k]["place"] for k in prob_map if k != i]
                place_others_j = [prob_map[k]["place"] for k in prob_map if k != j]
                p_j_2nd = prob_map[j]["place"] / max(sum(place_others_i), 1e-9)
                p_i_2nd = prob_map[i]["place"] / max(sum(place_others_j), 1e-9)
                p_umaren = p_i_win * p_j_2nd + p_j_win * p_i_2nd
                ev = p_umaren * horse_odds
                if ev >= EV_THRESHOLD["umaren"]:
                    stake = self._kelly_stake(p_umaren, horse_odds, budget)
                    recommendations.append({
                        "bet_type": "馬連",
                        "combination": f"{i}-{j}",
                        "probability": round(p_umaren, 4),
                        "odds": horse_odds,
                        "expected_value": round(ev, 3),
                        "stake": int(stake),
                    })

        # ---- 3連複 ----
        if "sanrenpuku" in odds_dict:
            for combo, horse_odds in odds_dict["sanrenpuku"].items():
                if len(combo) != 3:
                    continue
                i, j, k = combo
                if not all(x in prob_map for x in [i, j, k]):
                    continue
                p_i = prob_map[i]["place"]
                p_j = prob_map[j]["place"]
                p_k = prob_map[k]["place"]
                n = len(horse_nums)
                # 3頭全て3着以内に入る確率（近似）
                p_trio = p_i * p_j * p_k * (n * (n-1) * (n-2)) / (
                    max(sum(v["place"] for v in prob_map.values()), 1e-9) ** 2 * 6
                )
                ev = p_trio * horse_odds
                if ev >= EV_THRESHOLD["sanrenpuku"]:
                    stake = self._kelly_stake(p_trio, horse_odds, budget)
                    recommendations.append({
                        "bet_type": "3連複",
                        "combination": f"{i}-{j}-{k}",
                        "probability": round(p_trio, 4),
                        "odds": horse_odds,
                        "expected_value": round(ev, 3),
                        "stake": int(stake),
                    })

        df_rec = pd.DataFrame(recommendations)
        if not df_rec.empty:
            df_rec = df_rec.sort_values("expected_value", ascending=False)
            # 予算内に収める
            df_rec["stake"] = df_rec["stake"].clip(upper=budget // max(len(df_rec), 1))
            df_rec["stake"] = (df_rec["stake"] // 100) * 100  # 100円単位
            df_rec = df_rec[df_rec["stake"] >= 100]

        return df_rec

    @staticmethod
    def _kelly_stake(p: float, odds: float, budget: float) -> float:
        """フラクショナル・ケリー基準で賭け金計算"""
        b = odds - 1.0
        q = 1.0 - p
        kelly = (b * p - q) / b if b > 0 else 0
        kelly = max(kelly, 0) * KELLY_FRACTION
        return budget * kelly
