"""
データ収集パイプライン: スクレイピング→DB保存を一括実行
"""
import logging
from tqdm import tqdm
from .database import Database
from jra_predictor.scraper import RaceListScraper, RaceResultScraper, HorseProfileScraper

logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(self):
        self.db = Database()
        self.race_list_scraper = RaceListScraper()
        self.result_scraper = RaceResultScraper()
        self.horse_scraper = HorseProfileScraper()

    def collect_historical(
        self,
        start_year: int,
        end_year: int,
        course_codes: list[str] = None,
        skip_existing: bool = True,
    ):
        """過去レースの一括収集"""
        logger.info(f"Collecting races {start_year}-{end_year}")

        horse_ids_seen = set()
        race_ids_iter = self.race_list_scraper.iter_race_ids(
            start_year, end_year, course_codes
        )

        for race_id in tqdm(race_ids_iter, desc="Races"):
            if skip_existing and self.db.is_race_scraped(race_id):
                continue

            # レース基本情報
            info = self.result_scraper.fetch_race_info(race_id)
            if info:
                self.db.upsert_race_info(info)

            # レース結果（race_infoのフィールドもマージして保存）
            df_result = self.result_scraper.fetch_race_result(race_id)
            if df_result is not None:
                if info:
                    for col in ["date", "course", "course_code", "race_number",
                                "race_name", "distance", "surface", "weather", "track_condition"]:
                        if col in info and col not in df_result.columns:
                            df_result[col] = info[col]
                        elif col in info and (df_result[col].isna().all()):
                            df_result[col] = info[col]
                self.db.upsert_race_results(df_result)

                # 馬プロフィール（まだ取得していない馬のみ）
                for horse_id in df_result["horse_id"].dropna().unique():
                    if horse_id and horse_id not in horse_ids_seen:
                        horse_ids_seen.add(horse_id)
                        profile = self.horse_scraper.fetch_horse_profile(horse_id)
                        if profile:
                            self.db.upsert_horse_profile(profile)
                        history = self.horse_scraper.fetch_horse_history(horse_id)
                        if history is not None:
                            self.db.upsert_horse_history(history)

            # オッズ
            odds = self.result_scraper.fetch_odds(race_id)
            for bet_type, odds_dict in odds.items():
                if odds_dict:
                    self.db.save_odds(race_id, bet_type, odds_dict)

        logger.info("Historical collection complete")

    def collect_upcoming(self, race_ids: list[str]):
        """直近レース（予測用）の出走表・馬過去成績・オッズを収集"""
        horse_ids_seen = set()
        for race_id in tqdm(race_ids, desc="Upcoming races"):
            # レース基本情報・出走表
            info = self.result_scraper.fetch_race_info(race_id)
            if info:
                self.db.upsert_race_info(info)

            # 出走馬の過去成績を取得（未開催は出走表、開催済みは結果ページ）
            df_entry = self.result_scraper.fetch_race_entry(race_id)
            if df_entry is not None and not df_entry.empty:
                self.db.upsert_race_results(df_entry)
                for horse_id in df_entry["horse_id"].dropna().unique():
                    if horse_id and horse_id not in horse_ids_seen:
                        horse_ids_seen.add(horse_id)
                        profile = self.horse_scraper.fetch_horse_profile(horse_id)
                        if profile:
                            self.db.upsert_horse_profile(profile)
                        history = self.horse_scraper.fetch_horse_history(horse_id)
                        if history is not None:
                            self.db.upsert_horse_history(history)

            # オッズ
            odds = self.result_scraper.fetch_odds(race_id)
            for bet_type, odds_dict in odds.items():
                if odds_dict:
                    self.db.save_odds(race_id, bet_type, odds_dict)

        logger.info(f"Upcoming collection complete: {len(race_ids)} races")
