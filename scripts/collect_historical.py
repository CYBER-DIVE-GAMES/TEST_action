"""
2022〜2025年のレースデータをnetkeibaからスクレイピング
- エラーが起きてもスキップして継続
- logs/collect_errors.log にエラーを記録
- skip_existing=True なので途中再開可能
"""
import sys
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# ログ設定
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "collect_main.log", encoding="utf-8"),
    ],
)
error_logger = logging.getLogger("collect_errors")
error_handler = logging.FileHandler(LOG_DIR / "collect_errors.log", encoding="utf-8")
error_handler.setLevel(logging.WARNING)
error_logger.addHandler(error_handler)
error_logger.addHandler(logging.StreamHandler())

logger = logging.getLogger(__name__)

START_YEAR = 2022
END_YEAR = 2025


def main():
    logger.info(f"=== 収集開始: {START_YEAR}〜{END_YEAR}年 ===")
    logger.info(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    from jra_predictor.data import Database
    from jra_predictor.scraper import RaceListScraper, RaceResultScraper, HorseProfileScraper

    db = Database()
    race_list_scraper = RaceListScraper()
    result_scraper = RaceResultScraper()
    horse_scraper = HorseProfileScraper()

    horse_ids_seen = set()
    total_races = 0
    total_errors = 0
    skipped = 0

    for year in range(START_YEAR, END_YEAR + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"  {year}年 開始")
        logger.info(f"{'='*50}")
        year_races = 0
        year_errors = 0

        try:
            race_ids = list(race_list_scraper.iter_race_ids(year, year))
        except Exception as e:
            msg = f"[{year}] レースID一覧の取得失敗: {e}"
            error_logger.error(msg)
            error_logger.error(traceback.format_exc())
            logger.error(f"❌ {msg}")
            logger.error("=> レースIDの取得に失敗しました。netkeibaへの接続を確認してください。")
            continue

        logger.info(f"  {year}年: {len(race_ids)}レース検出")

        for i, race_id in enumerate(race_ids):
            try:
                # 既取得チェック
                if db.is_race_scraped(race_id):
                    skipped += 1
                    continue

                # レース基本情報
                try:
                    info = result_scraper.fetch_race_info(race_id)
                    if info:
                        db.upsert_race_info(info)
                except Exception as e:
                    error_logger.warning(f"[{race_id}] race_info取得失敗: {e}")

                # レース結果
                df_result = None
                try:
                    df_result = result_scraper.fetch_race_result(race_id)
                    if df_result is not None:
                        db.upsert_race_results(df_result)
                except Exception as e:
                    error_logger.warning(f"[{race_id}] race_result取得失敗: {e}")

                # 馬プロフィール・履歴
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
                            error_logger.warning(f"[{horse_id}] horse_profile取得失敗: {e}")
                        try:
                            history = horse_scraper.fetch_horse_history(horse_id)
                            if history is not None:
                                db.upsert_horse_history(history)
                        except Exception as e:
                            error_logger.warning(f"[{horse_id}] horse_history取得失敗: {e}")

                # オッズ（結果が取れた場合のみ）
                if df_result is not None:
                    try:
                        odds = result_scraper.fetch_odds(race_id)
                        for bet_type, odds_dict in odds.items():
                            if odds_dict:
                                db.save_odds(race_id, bet_type, odds_dict)
                    except Exception as e:
                        error_logger.warning(f"[{race_id}] odds取得失敗: {e}")

                year_races += 1
                total_races += 1

                # 進捗ログ（100レースごと）
                if (i + 1) % 100 == 0:
                    logger.info(f"  進捗: {i+1}/{len(race_ids)} ({(i+1)/len(race_ids)*100:.1f}%) "
                                f"取得:{year_races} エラー:{year_errors} スキップ:{skipped}")

            except KeyboardInterrupt:
                logger.info("\n中断されました。次回は同じコマンドで再開できます（取得済みはスキップ）")
                _print_summary(total_races, total_errors, skipped)
                sys.exit(0)

            except Exception as e:
                year_errors += 1
                total_errors += 1
                msg = f"[{race_id}] 予期しないエラー: {e}"
                error_logger.error(msg)
                error_logger.error(traceback.format_exc())
                logger.warning(f"❌ {msg}")

                # 連続エラーが多い場合は少し待機
                if year_errors > 0 and year_errors % 20 == 0:
                    logger.warning(f"  ⚠️  エラーが{year_errors}件を超えました。30秒待機します...")
                    logger.warning(f"  詳細: logs/collect_errors.log を確認してください")
                    time.sleep(30)

        logger.info(f"  {year}年完了: 取得={year_races}, エラー={year_errors}")

    _print_summary(total_races, total_errors, skipped)


def _print_summary(total_races, total_errors, skipped):
    logger.info(f"\n{'='*50}")
    logger.info(f"  収集完了サマリー")
    logger.info(f"{'='*50}")
    logger.info(f"  取得レース数: {total_races}")
    logger.info(f"  スキップ（既取得）: {skipped}")
    logger.info(f"  エラー数: {total_errors}")
    logger.info(f"  エラー詳細: logs/collect_errors.log")
    logger.info(f"  完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if total_errors > 0:
        logger.warning(f"\n  ⚠️ エラーあり。logs/collect_errors.log を確認してください。")
        logger.warning(f"  再実行すると未取得レースのみ再試行します（取得済みはスキップ）。")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
