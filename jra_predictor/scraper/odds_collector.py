"""
過去レースの複勝・馬連・ワイド・3連複オッズをnetkeibaから収集してDBに保存する
"""
import logging
import time
from tqdm import tqdm
from jra_predictor.data import Database
from jra_predictor.scraper.race_result import RaceResultScraper

logger = logging.getLogger(__name__)


class OddsCollector:
    def __init__(self, db: Database):
        self.db = db
        self.scraper = RaceResultScraper()

    def collect(self, start_year: int, end_year: int, limit: int = 0):
        """
        DBのrace_resultsから対象期間のrace_idを取得してオッズをスクレイピング・保存

        limit: 0=全件, N=N件で止める（動作確認用）
        """
        # 対象 race_id を取得（odds_rawに未保存のものだけ）
        race_ids = self._get_target_race_ids(start_year, end_year)
        logger.info(f"対象レース数: {len(race_ids)} 件")

        if limit > 0:
            race_ids = race_ids[:limit]
            logger.info(f"件数制限: {limit} 件")

        ok, skip, err = 0, 0, 0
        for race_id in tqdm(race_ids, desc="オッズ収集"):
            try:
                saved = self._collect_one(race_id)
                if saved:
                    ok += 1
                else:
                    skip += 1
            except Exception as e:
                logger.warning(f"{race_id} エラー: {e}")
                err += 1

        logger.info(f"完了: 成功={ok}, スキップ={skip}, エラー={err}")

    def _get_target_race_ids(self, start_year: int, end_year: int) -> list[str]:
        """未収集のrace_idを返す"""
        df_results = self.db.read_table("race_results")
        if df_results.empty or "race_id" not in df_results.columns:
            return []

        # 年フィルタ（race_idの先頭4桁が年）
        all_ids = df_results["race_id"].dropna().unique().tolist()
        target = [r for r in all_ids
                  if isinstance(r, str) and len(r) >= 4
                  and start_year <= int(r[:4]) <= end_year]

        # すでにodds_rawに保存済みのrace_idを除外
        df_odds = self.db.read_table("odds_raw")
        if not df_odds.empty:
            done = set(df_odds["race_id"].unique())
            target = [r for r in target if r not in done]

        target.sort()
        return target

    def _collect_one(self, race_id: str) -> bool:
        """1レース分のオッズをスクレイピングしてDBに保存。取得できなければFalseを返す"""
        raw = self.scraper.fetch_odds(race_id)

        # 単勝・複勝オッズ
        tansho = raw.get("tansho", {})
        if not tansho:
            return False  # アクセス失敗 or データなし

        fukusho_dict = {}
        tan_dict = {}
        for horse_num, od in tansho.items():
            if isinstance(od, dict):
                if od.get("win_odds"):
                    tan_dict[horse_num] = od["win_odds"]
                if od.get("place_odds_min"):
                    fukusho_dict[horse_num] = od["place_odds_min"]

        if fukusho_dict:
            self.db.save_odds(race_id, "fukusho", fukusho_dict)
        if tan_dict:
            self.db.save_odds(race_id, "tan", tan_dict)

        # 馬連
        umaren = raw.get("umaren", {})
        if umaren:
            self.db.save_odds(race_id, "umaren", umaren)

        # ワイド
        wide = raw.get("wide", {})
        if wide:
            self.db.save_odds(race_id, "wide", wide)

        # 3連複
        sanrenpuku = raw.get("sanrenpuku", {})
        if sanrenpuku:
            self.db.save_odds(race_id, "sanrenpuku", sanrenpuku)

        return True
