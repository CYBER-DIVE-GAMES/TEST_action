"""
2022〜2025年のレースデータをnetkeibaからスクレイピング
- レースID一覧はネットワーク不要で土日から直接生成（フリーズ対策）
- エラーが起きてもスキップして継続
- logs/collect_errors.log にエラーを記録
- skip_existing=True なので途中再開可能
"""
import sys
import time
import calendar
import logging
import traceback
from pathlib import Path
from datetime import datetime, date as _date

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

# JRA 中央10場コード
COURSE_CODES = ["01","02","03","04","05","06","07","08","09","10"]


def generate_weekend_dates(year: int) -> list[str]:
    """ネットワーク不要で土日日程を生成"""
    dates = []
    for month in range(1, 13):
        cal = calendar.monthcalendar(year, month)
        for week in cal:
            for day_idx in [5, 6]:  # 土=5, 日=6
                day = week[day_idx]
                if day != 0:
                    dates.append(f"{year}{month:02d}{day:02d}")
    return sorted(dates)


def generate_race_ids_for_date(date_str: str) -> list[str]:
    """1日分の全候補レースID（各場1〜12R）を生成"""
    ids = []
    for cc in COURSE_CODES:
        for r in range(1, 13):
            ids.append(f"{date_str}{cc}{r:02d}")
    return ids


def main():
    logger.info(f"=== 収集開始: {START_YEAR}〜{END_YEAR}年 ===")
    logger.info(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"方式: 土日日程を直接生成（ネットワーク不要）→ 各レースIDを順次取得")

    from jra_predictor.data import Database
    from jra_predictor.scraper import RaceResultScraper, HorseProfileScraper

    db = Database()
    result_scraper = RaceResultScraper()
    horse_scraper = HorseProfileScraper()

    horse_ids_seen = set()
    total_races = 0
    total_empty = 0   # 実際には開催なかったレースID
    total_errors = 0
    skipped = 0

    for year in range(START_YEAR, END_YEAR + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"  {year}年 開始")
        logger.info(f"{'='*50}")

        dates = generate_weekend_dates(year)
        logger.info(f"  {year}年: 土日{len(dates)}日 × 最大{len(COURSE_CODES)*12}レース/日 を順次確認")

        year_races = 0
        year_errors = 0

        for date_str in dates:
            candidate_ids = generate_race_ids_for_date(date_str)

            date_has_race = False  # この日に1つでもレースがあったか

            for race_id in candidate_ids:
                try:
                    # 既取得チェック
                    if db.is_race_scraped(race_id):
                        skipped += 1
                        date_has_race = True
                        continue

                    # レース基本情報（存在確認も兼ねる）
                    info = None
                    try:
                        info = result_scraper.fetch_race_info(race_id)
                    except Exception as e:
                        error_logger.warning(f"[{race_id}] race_info取得失敗: {e}")

                    # infoが取れなかった = このレースIDは存在しない → スキップ
                    if not info or not info.get("race_name"):
                        total_empty += 1
                        continue

                    db.upsert_race_info(info)
                    date_has_race = True

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

                    # オッズ
                    if df_result is not None:
                        try:
                            odds = result_scraper.fetch_odds(race_id)
                            for bet_type, odds_dict in odds.items():
                                if not odds_dict:
                                    continue
                                # tanshoは {馬番: {'win_odds':x,'place_odds_min':y,...}} 形式
                                # → 単勝/複勝を別々に展開してフラットに保存
                                if bet_type == "tansho":
                                    win_flat = {}
                                    place_min_flat = {}
                                    place_max_flat = {}
                                    for num, v in odds_dict.items():
                                        if isinstance(v, dict):
                                            if v.get("win_odds") is not None:
                                                win_flat[str(num)] = v["win_odds"]
                                            if v.get("place_odds_min") is not None:
                                                place_min_flat[str(num)] = v["place_odds_min"]
                                            if v.get("place_odds_max") is not None:
                                                place_max_flat[str(num)] = v["place_odds_max"]
                                        else:
                                            win_flat[str(num)] = v
                                    if win_flat:
                                        db.save_odds(race_id, "tansho", win_flat)
                                    if place_min_flat:
                                        db.save_odds(race_id, "fukusho", place_min_flat)
                                else:
                                    db.save_odds(race_id, bet_type, odds_dict)
                        except Exception as e:
                            error_logger.warning(f"[{race_id}] odds取得失敗: {e}")

                    year_races += 1
                    total_races += 1
                    logger.info(f"  ✅ {race_id} ({info.get('race_name','')}) 取得完了")

                except KeyboardInterrupt:
                    logger.info("\n中断されました。次回は同じコマンドで再開できます（取得済みはスキップ）")
                    _print_summary(total_races, total_errors, skipped, total_empty)
                    sys.exit(0)

                except Exception as e:
                    year_errors += 1
                    total_errors += 1
                    msg = f"[{race_id}] 予期しないエラー: {e}"
                    error_logger.error(msg)
                    error_logger.error(traceback.format_exc())
                    logger.warning(f"❌ {msg}")

                    if year_errors > 0 and year_errors % 20 == 0:
                        logger.warning(f"  ⚠️  エラーが{year_errors}件。30秒待機...")
                        logger.warning(f"  詳細: logs/collect_errors.log を確認してください")
                        time.sleep(30)

            logger.info(f"  {date_str}: 取得={year_races}件累計 / この日レースあり={date_has_race}")

        logger.info(f"  {year}年完了: 取得={year_races}, エラー={year_errors}")

    _print_summary(total_races, total_errors, skipped, total_empty)


def _print_summary(total_races, total_errors, skipped, total_empty):
    logger.info(f"\n{'='*50}")
    logger.info(f"  収集完了サマリー")
    logger.info(f"{'='*50}")
    logger.info(f"  取得レース数: {total_races}")
    logger.info(f"  スキップ（既取得）: {skipped}")
    logger.info(f"  空振り（開催なし）: {total_empty}")
    logger.info(f"  エラー数: {total_errors}")
    logger.info(f"  エラー詳細: logs/collect_errors.log")
    logger.info(f"  完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if total_errors > 0:
        logger.warning(f"\n  ⚠️ エラーあり。再実行すると未取得レースのみ再試行します。")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
