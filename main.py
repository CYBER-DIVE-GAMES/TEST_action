"""
JRA 予想ツール CLIエントリーポイント

使い方:
  python main.py collect --start 2020 --end 2024    # 過去データ収集
  python main.py train                               # モデル学習
  python main.py backtest                            # バックテスト
  python main.py predict --race_ids YYYYMMDDCCRR    # 直近レース予測
  python main.py report                              # 週次レポート出力
"""
import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/jra_predictor.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def cmd_collect(args):
    from jra_predictor.data import DataPipeline
    pipeline = DataPipeline()
    pipeline.collect_historical(
        start_year=args.start,
        end_year=args.end,
        course_codes=args.courses or None,
        skip_existing=not args.force,
    )


def cmd_train(args):
    from jra_predictor.data import Database
    from jra_predictor.features import FeatureBuilder
    from jra_predictor.models import RacePredictor

    db = Database()
    builder = FeatureBuilder(db)
    df = builder.build()

    if df.empty:
        logger.error("No data. Run 'collect' first.")
        return

    for target in ["is_win", "is_place"]:
        model = RacePredictor(target)
        model.train(df, tune_hyperparams=args.tune)
        model.save()
        logger.info(f"{target} model saved.")


def cmd_backtest(args):
    from jra_predictor.data import Database
    from jra_predictor.backtest import BacktestEngine

    db = Database()
    engine = BacktestEngine(db)

    if args.sweep:
        # EV閾値を段階的に変えて比較
        thresholds = [1.30, 1.20, 1.10, 1.05, 1.00]
        print("\n" + "="*70)
        print("EV閾値スイープ（複勝）")
        print(f"{'EV閾値':>8} {'ベット数':>8} {'的中率':>8} {'回収率':>8} {'収支':>12}")
        print("-"*70)
        for th in thresholds:
            r = engine.run(
                test_years=args.years,
                budget_per_race=args.budget,
                ev_threshold_override={"tan": th, "fukusho": th},
            )
            for bt in ["複勝", "単勝"]:
                if bt in r:
                    d = r[bt]
                    print(f"  {bt} EV>{th:.2f}  {d['n_bets']:>8,}  {d['hit_rate']:>7.1f}%  {d['roi']:>7.1f}%  ¥{d['profit']:>+,}")
        print("="*70)
    else:
        engine.run(
            test_years=args.years,
            budget_per_race=args.budget,
            tune_hyperparams=False,
        )


def cmd_predict(args):
    from jra_predictor.data import Database, DataPipeline
    from jra_predictor.features import FeatureBuilder
    from jra_predictor.models import RacePredictor, ExpectedValueCalculator
    import pandas as pd

    db = Database()
    pipeline = DataPipeline()

    # 出走表・オッズを収集
    pipeline.collect_upcoming(args.race_ids)

    # 特徴量構築
    builder = FeatureBuilder(db)
    df_all = builder.build()

    win_model = RacePredictor("is_win")
    place_model = RacePredictor("is_place")
    win_model.load()
    place_model.load()

    ev_calc = ExpectedValueCalculator(win_model, place_model)

    from jra_predictor.backtest.engine import BacktestEngine
    bt = BacktestEngine(db)

    all_recs = []
    for race_id in args.race_ids:
        df_race = df_all[df_all["race_id"] == race_id]
        if df_race.empty:
            logger.warning(f"No data for race {race_id}")
            continue
        odds = bt._get_odds_for_race(race_id)
        recs = ev_calc.recommend(df_race, odds, budget=args.budget)
        if not recs.empty:
            recs["race_id"] = race_id
            all_recs.append(recs)

    if all_recs:
        df_out = pd.concat(all_recs, ignore_index=True)
        print("\n" + "="*70)
        print("【推奨馬券】")
        print("="*70)
        print(df_out[["race_id", "bet_type", "combination", "odds",
                       "probability", "expected_value", "stake"]].to_string(index=False))
        # CSV出力
        out_path = f"data/predict_{args.race_ids[0]}.csv"
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved to {out_path}")
    else:
        print("推奨馬券なし（期待値閾値を超える馬券が見つかりませんでした）")


def cmd_list_races(args):
    """今日または指定日のレースID一覧を生成して表示"""
    from datetime import date, timedelta

    # 日付決定
    if args.date:
        d = date.fromisoformat(args.date)
    elif args.tomorrow:
        d = date.today() + timedelta(days=1)
    else:
        d = date.today()

    date_str = d.strftime("%Y%m%d")

    # JRA開催場コード（中央10場）
    course_codes = ["01","02","03","04","05","06","07","08","09","10"]

    print(f"\n{d.strftime('%Y年%m月%d日')} のレースID候補")
    print("（実際に開催されているレースのみ有効）")
    print("="*50)

    all_ids = []
    for cc in course_codes:
        for r in range(1, 13):  # 1〜12レース
            race_id = f"{date_str}{cc}{r:02d}"
            all_ids.append(race_id)

    # predict用にスペース区切りで表示
    print("\n【競馬場別】")
    course_names = {
        "01":"札幌","02":"函館","03":"福島","04":"新潟",
        "05":"東京","06":"中山","07":"中京","08":"京都",
        "09":"阪神","10":"小倉"
    }
    for cc in course_codes:
        ids = [f"{date_str}{cc}{r:02d}" for r in range(1, 13)]
        print(f"  {course_names[cc]}({cc}): {ids[0]} 〜 {ids[-1]}")

    print(f"\n【全レース予測コマンド例（東京開催の場合）】")
    tokyo_ids = " ".join([f"{date_str}0501", f"{date_str}0506", f"{date_str}0511"])
    print(f"  python main.py predict {date_str}0501 {date_str}0502 ... {date_str}0512")


def cmd_scrape_odds(args):
    from jra_predictor.data import Database
    from jra_predictor.scraper import OddsCollector
    db = Database()
    collector = OddsCollector(db)
    collector.collect(
        start_year=args.start,
        end_year=args.end,
        limit=args.limit,
    )


def cmd_import(args):
    from jra_predictor.data import Database
    from jra_predictor.data.kaggle_importer import KaggleJraImporter
    db = Database()
    importer = KaggleJraImporter(db)
    from pathlib import Path
    p = Path(args.path)
    if p.is_dir():
        importer.import_all(args.path)
    else:
        importer.import_race_results(args.path)
    logger.info("インポート完了")


def main():
    parser = argparse.ArgumentParser(description="JRA 中央競馬 予想ツール")
    sub = parser.add_subparsers(dest="command")

    # collect
    p_collect = sub.add_parser("collect", help="過去データ収集")
    p_collect.add_argument("--start", type=int, default=2019, help="開始年")
    p_collect.add_argument("--end", type=int, default=2024, help="終了年")
    p_collect.add_argument("--courses", nargs="*", help="競馬場コード（省略時:全場）")
    p_collect.add_argument("--force", action="store_true", help="既存データ上書き")

    # train
    p_train = sub.add_parser("train", help="モデル学習")
    p_train.add_argument("--tune", action="store_true", help="ハイパーパラメータ自動チューニング")

    # backtest
    p_bt = sub.add_parser("backtest", help="バックテスト実行")
    p_bt.add_argument("--years", type=int, default=2, help="テスト期間（年）")
    p_bt.add_argument("--budget", type=float, default=10000, help="1レースあたりの予算（円）")
    p_bt.add_argument("--sweep", action="store_true", help="EV閾値を段階的に変えて比較")

    # list-races
    p_list = sub.add_parser("list-races", help="今日のレースID一覧を表示")
    p_list.add_argument("--date", type=str, default="", help="日付指定 YYYY-MM-DD（省略時:今日）")
    p_list.add_argument("--tomorrow", action="store_true", help="明日のレースを表示")

    # scrape-odds
    p_odds = sub.add_parser("scrape-odds", help="過去レースのオッズをnetkeibaから収集")
    p_odds.add_argument("--start", type=int, default=2019, help="開始年")
    p_odds.add_argument("--end", type=int, default=2021, help="終了年")
    p_odds.add_argument("--limit", type=int, default=0, help="件数制限（0=全件、50等で動作確認可）")

    # import
    p_imp = sub.add_parser("import", help="CSVからデータをインポート")
    p_imp.add_argument("path", help="CSVファイルまたはフォルダのパス")

    # predict
    p_pred = sub.add_parser("predict", help="直近レース予測")
    p_pred.add_argument("race_ids", nargs="+", help="レースID（12桁）")
    p_pred.add_argument("--budget", type=float, default=10000, help="予算（円）")

    args = parser.parse_args()

    if args.command == "list-races":
        cmd_list_races(args)
    elif args.command == "scrape-odds":
        cmd_scrape_odds(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
