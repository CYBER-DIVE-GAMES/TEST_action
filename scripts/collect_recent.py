"""
直近N週間分の差分スクレイピング
毎週月曜に実行することで常に最新データをDBに保つ

使い方:
  python scripts/collect_recent.py          # デフォルト: 直近2週間
  python scripts/collect_recent.py --weeks 4  # 直近4週間
"""
import sys
import time
import logging
import traceback
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "collect_recent.log", encoding="utf-8"),
    ],
)
error_logger = logging.getLogger("recent_errors")
error_logger.addHandler(logging.FileHandler(LOG_DIR / "collect_errors.log", encoding="utf-8"))

logger = logging.getLogger(__name__)

COURSE_CODES = ["01","02","03","04","05","06","07","08","09","10"]


def get_recent_dates(weeks: int) -> list[str]:
    """直近N週間の土日を返す"""
    today = date.today()
    dates = []
    for i in range(weeks * 7):
        d = today - timedelta(days=i)
        if d.weekday() in (5, 6):  # 土=5, 日=6
            dates.append(d.strftime("%Y%m%d"))
    return sorted(dates)


def main():
    parser = argparse.ArgumentParser(description="直近N週間の差分スクレイピング")
    parser.add_argument("--weeks", type=int, default=2, help="直近何週間分か（デフォルト:2）")
    args = parser.parse_args()

    dates = get_recent_dates(args.weeks)
    logger.info(f"=== 差分スクレイピング開始 ===")
    logger.info(f"対象: 直近{args.weeks}週間 ({len(dates)}日分)")
    logger.info(f"対象日: {dates[0]} 〜 {dates[-1]}")

    from jra_predictor.data import Database
    from jra_predictor.scraper import RaceResultScraper, HorseProfileScraper

    db = Database()
    result_scraper = RaceResultScraper()
    horse_scraper = HorseProfileScraper()

    horse_ids_seen = set()
    total_races = 0
    total_errors = 0
    skipped = 0
    total_empty = 0

    for date_str in dates:
        logger.info(f"\n--- {date_str} ---")

        for cc in COURSE_CODES:
            for r in range(1, 13):
                race_id = f"{date_str}{cc}{r:02d}"
                try:
                    # 既取得チェック（ただし直近は結果が更新される可能性があるので強制再取得しない）
                    if db.is_race_scraped(race_id):
                        skipped += 1
                        continue

                    info = None
                    try:
                        info = result_scraper.fetch_race_info(race_id)
                    except Exception as e:
                        error_logger.warning(f"[{race_id}] race_info失敗: {e}")

                    if not info or not info.get("race_name"):
                        total_empty += 1
                        continue

                    db.upsert_race_info(info)

                    df_result = None
                    try:
                        df_result = result_scraper.fetch_race_result(race_id)
                        if df_result is not None:
                            db.upsert_race_results(df_result)
                    except Exception as e:
                        error_logger.warning(f"[{race_id}] race_result失敗: {e}")

                    if df_result is not None:
                        for horse_id in df_result["horse_id"].dropna().unique():
                            if not horse_id or horse_id in horse_ids_seen:
                                continue
                            horse_ids_seen.add(horse_id)
                            try:
                                profile = horse_scraper.fetch_horse_profile(horse_id)
                                if profile:
                                    db.upsert_horse_profile(profile)
                            except Exception as e:
                                error_logger.warning(f"[{horse_id}] profile失敗: {e}")
                            try:
                                history = horse_scraper.fetch_horse_history(horse_id)
                                if history is not None:
                                    db.upsert_horse_history(history)
                            except Exception as e:
                                error_logger.warning(f"[{horse_id}] history失敗: {e}")

                        try:
                            odds = result_scraper.fetch_odds(race_id)
                            for bet_type, odds_dict in odds.items():
                                if not odds_dict:
                                    continue
                                if bet_type == "tansho":
                                    win_flat = {}
                                    place_flat = {}
                                    for num, v in odds_dict.items():
                                        if isinstance(v, dict):
                                            if v.get("win_odds") is not None:
                                                win_flat[str(num)] = v["win_odds"]
                                            if v.get("place_odds_min") is not None:
                                                place_flat[str(num)] = v["place_odds_min"]
                                        else:
                                            win_flat[str(num)] = v
                                    if win_flat:
                                        db.save_odds(race_id, "tansho", win_flat)
                                    if place_flat:
                                        db.save_odds(race_id, "fukusho", place_flat)
                                else:
                                    db.save_odds(race_id, bet_type, odds_dict)
                        except Exception as e:
                            error_logger.warning(f"[{race_id}] odds失敗: {e}")

                    total_races += 1
                    logger.info(f"  ✅ {race_id} ({info.get('race_name','')})")

                except KeyboardInterrupt:
                    logger.info("\n中断されました")
                    _summary(total_races, total_errors, skipped, total_empty)
                    sys.exit(0)

                except Exception as e:
                    total_errors += 1
                    error_logger.error(f"[{race_id}] 予期しないエラー: {e}")
                    error_logger.error(traceback.format_exc())

    _summary(total_races, total_errors, skipped, total_empty)


def _summary(total_races, total_errors, skipped, total_empty):
    logger.info(f"\n{'='*40}")
    logger.info(f"  差分スクレイピング完了")
    logger.info(f"  取得: {total_races} / スキップ: {skipped} / 空振り: {total_empty} / エラー: {total_errors}")
    logger.info(f"  完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*40}")


if __name__ == "__main__":
    main()
