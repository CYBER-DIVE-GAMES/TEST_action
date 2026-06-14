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


def cmd_predict_today(args):
    """今日の開催レースを自動検出→データ取得→予測まで一括実行"""
    from datetime import date, timedelta
    from jra_predictor.data import Database, DataPipeline
    from jra_predictor.features import FeatureBuilder
    from jra_predictor.models import RacePredictor, ExpectedValueCalculator
    from jra_predictor.backtest.engine import BacktestEngine
    from jra_predictor.scraper.race_list import RaceListScraper
    import pandas as pd

    if args.date:
        d = date.fromisoformat(args.date)
    else:
        d = date.today()
    date_str = d.strftime("%Y%m%d")

    course_codes = args.courses or ["01","02","03","04","05","06","07","08","09","10"]
    course_names = {
        "01":"札幌","02":"函館","03":"福島","04":"新潟",
        "05":"東京","06":"中山","07":"中京","08":"京都",
        "09":"阪神","10":"小倉"
    }

    print(f"\n{d.strftime('%Y年%m月%d日')} の開催レースを検索中...")

    # 当日のレースIDを検出
    scraper = RaceListScraper()
    race_ids = scraper._fetch_date_race_ids(date_str, course_codes)

    if not race_ids:
        print("本日の開催レースが見つかりませんでした。")
        print(f"  --date YYYY-MM-DD で日付を指定するか、")
        print(f"  --courses 06 09 で競馬場コードを指定してください。")
        return

    print(f"  {len(race_ids)}レース検出: {race_ids[0]} 〜 {race_ids[-1]}")

    # データ収集（出走表・馬過去成績・オッズ）
    print("\n出走馬データを収集中（数分かかります）...")
    db = Database()
    pipeline = DataPipeline()
    pipeline.collect_upcoming(race_ids)

    # 特徴量構築
    print("\n特徴量を計算中...")
    builder = FeatureBuilder(db)
    df_all = builder.build()

    # モデルロード
    win_model = RacePredictor("is_win")
    place_model = RacePredictor("is_place")
    win_model.load()
    place_model.load()

    ev_calc = ExpectedValueCalculator(win_model, place_model)
    bt = BacktestEngine(db)

    # 各レースを予測
    print("\n" + "="*70)
    print(f"【{d.strftime('%Y年%m月%d日')} 予測結果】")
    print("="*70)

    all_recs = []
    for race_id in race_ids:
        df_race = df_all[df_all["race_id"] == race_id]
        if df_race.empty:
            continue

        first = df_race.iloc[0]
        course = first.get("course", "")
        race_name = first.get("race_name", race_id)
        race_num = first.get("race_number", "")

        odds = bt._get_odds_for_race(race_id)
        if not any(odds.values()):
            odds = bt._build_odds_from_df(df_race)

        recs = ev_calc.recommend(df_race, odds, budget=args.budget)

        print(f"\n【{course} {race_num}R {race_name}】 {len(df_race)}頭")

        if recs.empty:
            print("  → 買い目なし（期待値閾値を超える馬券なし）")
        else:
            # 複勝・ワイド・3連複のみ表示
            allowed = {"複勝", "ワイド", "3連複"}
            recs_show = recs[recs["bet_type"].isin(allowed)]
            if recs_show.empty:
                print("  → 買い目なし")
            else:
                for _, r in recs_show.iterrows():
                    print(f"  {r['bet_type']:4s}  {r['combination']:12s}  "
                          f"オッズ:{r['odds']:5.1f}  EV:{r['expected_value']:.3f}  "
                          f"購入:¥{int(r['stake']):,}")
                recs["race_id"] = race_id
                all_recs.append(recs_show)

    if all_recs:
        df_out = pd.concat(all_recs, ignore_index=True)
        out_path = f"data/predict_{date_str}.csv"
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n買い目をCSVに保存: {out_path}")
        print(f"合計購入額: ¥{df_out['stake'].sum():,}")
    else:
        print("\n本日の全レースで買い目なし")

    print("="*70)


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


def cmd_debug_scrape(args):
    """shutuba.htmlの実際のHTML構造を診断する"""
    import logging
    logging.getLogger("jra_predictor").setLevel(logging.DEBUG)
    from jra_predictor.scraper import RaceResultScraper
    from config.settings import NETKEIBA_RACE

    s = RaceResultScraper()
    race_id = args.race_id
    url = f"{NETKEIBA_RACE}/race/shutuba.html"
    print(f"Fetching: {url}?race_id={race_id}")
    soup = s.get(url, params={"race_id": race_id})
    if soup is None:
        print("ERROR: got None response")
        return

    print("\n=== ALL TABLES ===")
    for i, t in enumerate(soup.find_all("table")):
        print(f"  [{i}] class={t.get('class')} id={t.get('id')}")
        rows = t.find_all("tr")
        print(f"       rows={len(rows)}")
        if rows:
            first_tds = rows[0].find_all(["td", "th"])
            print(f"       first row cells={len(first_tds)}: {[c.get_text(strip=True)[:15] for c in first_tds[:5]]}")

    print("\n=== PAGE TITLE ===")
    print(soup.title.string if soup.title else "(none)")

    print("\n=== HORSE LINKS (first 5) ===")
    for a in soup.select("a[href*='/horse/']")[:5]:
        print(f"  {a.get('href')} -> {a.get_text(strip=True)}")

    print("\n=== JOCKEY LINKS (first 3) ===")
    for a in soup.select("a[href*='/jockey/']")[:3]:
        print(f"  {a.get('href')} -> {a.get_text(strip=True)}")


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

    # debug-scrape
    p_dbg = sub.add_parser("debug-scrape", help="shutuba.htmlのHTML構造を診断")
    p_dbg.add_argument("race_id", help="レースID（12桁）")

    # predict-today
    p_today = sub.add_parser("predict-today", help="今日の開催レースを自動検出して予測")
    p_today.add_argument("--date", type=str, default="", help="日付 YYYY-MM-DD（省略時:今日）")
    p_today.add_argument("--courses", nargs="*", help="競馬場コード（省略時:全場）")
    p_today.add_argument("--budget", type=float, default=10000, help="予算（円）")

    args = parser.parse_args()

    if args.command == "debug-scrape":
        cmd_debug_scrape(args)
    elif args.command == "list-races":
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
    elif args.command == "predict-today":
        cmd_predict_today(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
