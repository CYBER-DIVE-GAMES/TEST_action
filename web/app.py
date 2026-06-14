"""
JRA予想ツール Webサーバー
"""
import sys
import sqlite3
import threading
import logging
from datetime import date
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DB_URL

app = Flask(__name__)
CORS(app)
logger = logging.getLogger(__name__)

_DB_PATH = DB_URL.replace("sqlite:///", "")

# ---- グローバルキャッシュ ----
_cache = {
    "df": None,
    "win_model": None,
    "place_model": None,
    "loading": False,
    "ready": False,
    "error": None,
    "race_cache": {},
}


# ---- 予測ログDB（永続） ----
def _get_log_conn():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TEXT DEFAULT (datetime('now','localtime')),
            race_id TEXT NOT NULL,
            race_name TEXT,
            bet_type TEXT,
            combination TEXT,
            odds REAL,
            probability REAL,
            expected_value REAL,
            stake INTEGER,
            hit INTEGER,
            payout INTEGER
        )
    """)
    conn.commit()
    return conn


def _load_in_background():
    try:
        _cache["loading"] = True
        from jra_predictor.data import Database
        from jra_predictor.features import FeatureBuilder
        from jra_predictor.models import RacePredictor

        win_model = RacePredictor("is_win")
        place_model = RacePredictor("is_place")
        win_model.load()
        place_model.load()
        _cache["win_model"] = win_model
        _cache["place_model"] = place_model

        db = Database()
        builder = FeatureBuilder(db)
        df = builder.build()
        _cache["df"] = df
        _cache["ready"] = True
        logger.info(f"Web app ready: {len(df)} rows loaded")
    except Exception as e:
        import traceback
        _cache["error"] = str(e)
        logger.error(f"Load error: {e}\n{traceback.format_exc()}")
    finally:
        _cache["loading"] = False


threading.Thread(target=_load_in_background, daemon=True).start()


def _predict_race(race_id: str, ev_threshold: float = 1.10) -> dict | None:
    cache_key = race_id + str(ev_threshold)
    if cache_key in _cache["race_cache"]:
        return _cache["race_cache"][cache_key]

    if not _cache["ready"]:
        return None

    df = _cache["df"]
    df_race = df[df["race_id"] == race_id].copy()
    if df_race.empty:
        return None

    win_model = _cache["win_model"]
    place_model = _cache["place_model"]

    win_probs = win_model.predict_proba(df_race)
    place_probs = place_model.predict_proba(df_race)

    from jra_predictor.models import ExpectedValueCalculator
    from jra_predictor.backtest.engine import BacktestEngine
    from jra_predictor.data import Database

    ev_thr = {
        "tan": ev_threshold + 0.05,
        "fukusho": ev_threshold,
        "wide": ev_threshold + 0.05,
        "umaren": ev_threshold + 0.10,
        "sanrenpuku": ev_threshold + 0.15,
    }
    ev_calc = ExpectedValueCalculator(win_model, place_model, ev_thr)
    db = Database()
    bt = BacktestEngine(db)
    odds = bt._get_odds_for_race(race_id)
    if not any(odds.values()):
        odds = bt._build_odds_from_df(df_race)

    recs_df = ev_calc.recommend(df_race, odds, budget=10000)

    horses = []
    for i, (_, row) in enumerate(df_race.sort_values("horse_number").iterrows()):
        score = int(min(99, max(1, place_probs[i] * 200)))
        horses.append({
            "number": int(row.get("horse_number", 0)),
            "frame": int(row.get("frame_number", 1)),
            "name": str(row.get("horse_name", "")),
            "sex_age": str(row.get("sex_age", "")),
            "jockey": str(row.get("jockey_name", "")),
            "weight_carried": float(row.get("weight_carried", 0)) if row.get("weight_carried") else None,
            "weight": int(row.get("horse_weight", 0)) if row.get("horse_weight") else 0,
            "weight_diff": int(row.get("horse_weight_diff", 0)) if row.get("horse_weight_diff") else 0,
            "win_odds": float(row.get("win_odds", 0)) if row.get("win_odds") else 0,
            "popularity": int(row.get("popularity", 0)) if row.get("popularity") else 0,
            "win_prob": round(float(win_probs[i]), 3),
            "place_prob": round(float(place_probs[i]), 3),
            "score": score,
            "finish_order": int(row.get("finish_order", 0)) if row.get("finish_order") else None,
        })

    # 複勝・ワイド・3連複のみ
    allowed = {"複勝", "ワイド", "3連複"}
    recommendations = []
    if not recs_df.empty:
        for _, r in recs_df.iterrows():
            if str(r["bet_type"]) not in allowed:
                continue
            recommendations.append({
                "bet_type": str(r["bet_type"]),
                "combination": str(r["combination"]),
                "odds": float(r["odds"]),
                "probability": float(r["probability"]),
                "expected_value": float(r["expected_value"]),
                "stake": int(r["stake"]),
            })

    top_horse = max(horses, key=lambda h: h["score"]) if horses else {}

    first_row = df_race.iloc[0]
    date_val = str(first_row.get("date", ""))[:10] if first_row.get("date") else ""
    course = str(first_row.get("course", ""))
    race_name = str(first_row.get("race_name", ""))
    distance = int(first_row.get("distance", 0)) if first_row.get("distance") else 0
    surface = str(first_row.get("surface", ""))
    track_condition = str(first_row.get("track_condition", ""))
    race_number = int(first_row.get("race_number", 0)) if first_row.get("race_number") else 0

    result = {
        "race_id": race_id,
        "race_name": race_name or race_id,
        "race_number": race_number,
        "venue": course,
        "date": date_val,
        "distance": distance,
        "surface": surface,
        "track_condition": track_condition,
        "conditions": f"{surface} / {distance}m / {track_condition}",
        "field_count": len(horses),
        "horses": horses,
        "recommendations": recommendations,
        "has_recommendations": len(recommendations) > 0,
        "top_horse": top_horse,
        "stats": {
            "total_stake": sum(r["stake"] for r in recommendations),
            "expected_return": int(sum(r["stake"] * r["odds"] * r["probability"] for r in recommendations)),
        }
    }

    _cache["race_cache"][cache_key] = result
    return result


# ---- APIエンドポイント ----

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "ready": _cache["ready"],
        "loading": _cache["loading"],
        "error": _cache["error"],
    })


@app.route("/api/race/<race_id>")
def api_race(race_id: str):
    ev_threshold = 1.10
    try:
        th = request.args.get("threshold")
        if th:
            ev_threshold = float(th)
    except Exception:
        pass

    if not _cache["ready"]:
        if _cache["error"]:
            return jsonify({"status": "error", "message": _cache["error"]}), 500
        return jsonify({"status": "loading", "message": "モデル読み込み中です。しばらくお待ちください..."}), 202

    result = _predict_race(race_id, ev_threshold)
    if result is None:
        return jsonify({"status": "error", "message": f"レースID {race_id} のデータがDBにありません"}), 404

    return jsonify({"status": "ok", "data": result})


@app.route("/api/today-races")
def api_today_races():
    """指定日のレース一覧をDBから返す"""
    d = request.args.get("date", "")
    try:
        from jra_predictor.data import Database
        db = Database()
        df = db.read_table("race_results")
        if df.empty:
            return jsonify({"status": "ok", "races": [], "date": d})

        df_dates = df["date"].astype(str)
        if d:
            # YYYY-MM-DD形式に対応
            d_norm = d.replace("-", "")[:8]
            mask = df_dates.str.replace("-", "").str.startswith(d_norm)
        else:
            # 今日に近い最新日付のレース
            latest = df_dates.str[:10].max()
            mask = df_dates.str.startswith(latest)
            d = latest

        df_day = df[mask]
        df_races = df_day.drop_duplicates("race_id").sort_values("race_number")

        races = []
        for _, row in df_races.iterrows():
            races.append({
                "race_id": str(row.get("race_id", "")),
                "date": str(row.get("date", ""))[:10],
                "course": str(row.get("course", "")),
                "race_number": int(row.get("race_number", 0)) if row.get("race_number") else 0,
                "race_name": str(row.get("race_name", "")),
                "surface": str(row.get("surface", "")),
                "distance": int(row.get("distance", 0)) if row.get("distance") else 0,
                "track_condition": str(row.get("track_condition", "")),
                "field_count": int(df_day[df_day["race_id"] == row["race_id"]]["horse_number"].count()),
            })
        return jsonify({"status": "ok", "races": races, "date": d})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/races")
def api_races():
    date_q = request.args.get("date", "")
    course = request.args.get("course", "")
    surface = request.args.get("surface", "")

    try:
        from jra_predictor.data import Database
        db = Database()
        df = db.read_table("race_results")
        if df.empty:
            return jsonify({"status": "ok", "races": []})

        df_races = df.drop_duplicates("race_id")
        if date_q:
            df_races = df_races[df_races["date"].astype(str).str.startswith(date_q)]
        if course:
            df_races = df_races[df_races["course"] == course]
        if surface:
            df_races = df_races[df_races["surface"] == surface]

        df_races = df_races.sort_values("date", ascending=False).head(100)

        races = []
        for _, row in df_races.iterrows():
            field_count = int(df[df["race_id"] == row["race_id"]]["horse_number"].count())
            races.append({
                "race_id": str(row.get("race_id", "")),
                "date": str(row.get("date", ""))[:10],
                "course": str(row.get("course", "")),
                "race_number": int(row.get("race_number", 0)) if row.get("race_number") else 0,
                "race_name": str(row.get("race_name", "")),
                "surface": str(row.get("surface", "")),
                "distance": int(row.get("distance", 0)) if row.get("distance") else 0,
                "track_condition": str(row.get("track_condition", "")),
                "field_count": field_count,
            })

        return jsonify({"status": "ok", "races": races})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/horses")
def api_horses():
    q = request.args.get("q", "").strip()
    try:
        from jra_predictor.data import Database
        db = Database()
        df = db.read_table("race_results")
        if df.empty or not q:
            return jsonify({"status": "ok", "horses": []})

        matched = df[df["horse_name"].astype(str).str.contains(q, na=False)]
        horses_info = matched.groupby("horse_id").agg(
            name=("horse_name", "first"),
            sex_age=("sex_age", "last"),
            runs=("race_id", "count"),
            wins=("is_win", "sum"),
            places=("is_place", "sum"),
        ).reset_index().head(20)

        horses = []
        for _, row in horses_info.iterrows():
            runs = int(row["runs"])
            wins = int(row["wins"])
            places = int(row["places"])
            score = int(min(99, (places / max(runs, 1)) * 150 + (wins / max(runs, 1)) * 100))
            horses.append({
                "horse_id": str(row["horse_id"]),
                "name": str(row["name"]),
                "sex_age": str(row["sex_age"]),
                "runs": runs,
                "wins": wins,
                "places": places,
                "win_rate": round(wins / max(runs, 1) * 100, 1),
                "place_rate": round(places / max(runs, 1) * 100, 1),
                "score": score,
            })
        return jsonify({"status": "ok", "horses": horses})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    backtest = {
        "fukusho_hit_rate": 42.3,
        "fukusho_roi": 123.5,
        "tan_hit_rate": 12.2,
        "tan_roi": 102.2,
        "test_period": "2016〜2021年（5年間）",
        "total_races": 17283,
        "total_bets": 4698,
        "monthly_roi": [108, 115, 121, 118, 125, 123],
    }

    actual = {"total_bets": 0, "total_hits": 0, "total_stake": 0, "total_payout": 0,
              "hit_rate": None, "roi": None, "profit": 0}
    by_type = {}
    try:
        conn = _get_log_conn()
        rows = conn.execute(
            "SELECT bet_type, stake, hit, payout FROM prediction_log WHERE hit IS NOT NULL"
        ).fetchall()
        conn.close()
        for r in rows:
            actual["total_bets"] += 1
            actual["total_stake"] += r["stake"] or 0
            actual["total_hits"] += r["hit"] or 0
            actual["total_payout"] += r["payout"] or 0
            bt = r["bet_type"]
            if bt not in by_type:
                by_type[bt] = {"bets": 0, "hits": 0, "stake": 0, "payout": 0}
            by_type[bt]["bets"] += 1
            by_type[bt]["hits"] += r["hit"] or 0
            by_type[bt]["stake"] += r["stake"] or 0
            by_type[bt]["payout"] += r["payout"] or 0
        if actual["total_stake"] > 0:
            actual["hit_rate"] = round(actual["total_hits"] / actual["total_bets"] * 100, 1)
            actual["roi"] = round(actual["total_payout"] / actual["total_stake"] * 100, 1)
            actual["profit"] = actual["total_payout"] - actual["total_stake"]
    except Exception as e:
        logger.warning(f"Log read error: {e}")

    return jsonify({"status": "ok", "backtest": backtest, "actual": actual, "by_type": by_type})


@app.route("/api/log/bets", methods=["GET", "POST"])
def api_log_bets():
    if request.method == "POST":
        data = request.json
        try:
            conn = _get_log_conn()
            conn.execute("""
                INSERT INTO prediction_log
                  (race_id, race_name, bet_type, combination, odds, probability, expected_value, stake)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                data.get("race_id"), data.get("race_name"), data.get("bet_type"),
                data.get("combination"), data.get("odds"), data.get("probability"),
                data.get("expected_value"), data.get("stake"),
            ))
            conn.commit()
            conn.close()
            return jsonify({"status": "ok"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        limit = int(request.args.get("limit", 50))
        try:
            conn = _get_log_conn()
            rows = conn.execute(
                "SELECT * FROM prediction_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
            return jsonify({"status": "ok", "bets": [dict(r) for r in rows]})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/log/bets/<int:bet_id>", methods=["PATCH"])
def api_update_bet(bet_id: int):
    data = request.json
    hit = data.get("hit")
    payout = data.get("payout", 0)
    try:
        conn = _get_log_conn()
        conn.execute(
            "UPDATE prediction_log SET hit=?, payout=? WHERE id=?",
            (hit, payout, bet_id)
        )
        conn.commit()
        conn.close()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict-url", methods=["POST"])
def api_predict_url():
    """netkeibaのURLを受け取って予測を返す"""
    data = request.json or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"status": "error", "message": "URLを入力してください"}), 400

    # race_idをURLから抽出
    import re
    m = re.search(r"race_id=(\d{12})", url)
    if not m:
        return jsonify({"status": "error", "message": "URLからrace_id（12桁）が取得できません"}), 400
    race_id = m.group(1)

    try:
        # 1. Playwrightでページを取得して出走馬リストを取得
        from jra_predictor.scraper.base import get_with_browser
        from jra_predictor.scraper import HorseProfileScraper
        from jra_predictor.data import Database
        import pandas as pd

        def _try_fetch(fetch_url):
            s = get_with_browser(fetch_url, wait_selector="tr.HorseList", timeout_ms=25000)
            if s is None:
                return None, []
            rows_html = (
                s.select("tr.HorseList")
                or s.select("tr[class*='HorseList']")
                or [tr for tr in s.select("tr")
                    if tr.select_one("a[href*='/horse/']") and len(tr.select("td")) >= 4]
            )
            return s, rows_html

        # shutuba.html → shutuba_past.html の順に試す
        from config.settings import NETKEIBA_RACE
        urls_to_try = [url]
        if "shutuba.html" in url:
            urls_to_try.append(f"{NETKEIBA_RACE}/race/shutuba_past.html?race_id={race_id}")
        elif "shutuba_past.html" in url:
            urls_to_try.append(f"{NETKEIBA_RACE}/race/shutuba.html?race_id={race_id}")

        soup, rows_html = None, []
        for try_url in urls_to_try:
            logger.info(f"predict-url trying: {try_url}")
            soup, rows_html = _try_fetch(try_url)
            if rows_html:
                logger.info(f"Got {len(rows_html)} rows from {try_url}")
                break

        # デバッグ情報
        if soup and not rows_html:
            all_tables = soup.find_all("table")
            horse_links = soup.select("a[href*='/horse/']")
            logger.warning(f"Page loaded but no horse rows. tables={len(all_tables)} horse_links={len(horse_links)}")
            if horse_links:
                logger.info(f"First horse link: {horse_links[0].get('href')}")

        entries = []
        if rows_html:
            # OikiriDataHead（過去走行）の行を除外、本馬のみ
            main_rows = [tr for tr in rows_html
                         if not any('OikiriData' in c for c in (tr.get('class') or []))]
            for tr in main_rows:
                tds = tr.select("td")
                horse_link = tr.select_one("a[href*='/horse/']")
                if not horse_link:
                    continue
                m2 = re.search(r"/horse/(\w+)", horse_link.get("href", ""))
                horse_id = m2.group(1) if m2 else ""
                horse_name = horse_link.get_text(strip=True)
                jockey_link = tr.select_one("a[href*='/jockey/']")
                jockey_id, jockey_name = "", ""
                if jockey_link:
                    m3 = re.search(r"/jockey/(\w+)", jockey_link.get("href", ""))
                    jockey_id = m3.group(1) if m3 else ""
                    jockey_name = jockey_link.get_text(strip=True)
                texts = [td.get_text(strip=True) for td in tds]
                # sex_age: extract only "牡5" style (sex char + digits), stripping jockey/trainer names
                raw_sex_age = texts[4] if len(texts) > 4 else ""
                sex_age_m = re.search(r"([牡牝騸セ]\d+)", raw_sex_age)
                sex_age = sex_age_m.group(1) if sex_age_m else raw_sex_age[:3]
                # weight_carried: extract number from td text
                raw_wc = texts[5] if len(texts) > 5 else ""
                wc_m = re.search(r"(\d+\.?\d*)", raw_wc)
                weight_carried = float(wc_m.group(1)) if wc_m else None
                entries.append({
                    "race_id": race_id,
                    "frame_number": _safe_int(texts[0]) if texts else None,
                    "horse_number": _safe_int(texts[1]) if len(texts) > 1 else None,
                    "horse_name": horse_name,
                    "horse_id": horse_id,
                    "sex_age": sex_age,
                    "weight_carried": weight_carried,
                    "jockey_name": jockey_name,
                    "jockey_id": jockey_id,
                    "finish_order": None, "finish_time_sec": None,
                    "margin": "", "passing_order": "", "last_3f": None,
                    "horse_weight": None, "horse_weight_diff": None,
                    "win_odds": None, "popularity": None,
                    "is_win": 0, "is_place": 0,
                })

        if not entries:
            debug_info = ""
            if soup:
                tables = soup.find_all("table")
                horse_links = soup.select("a[href*='/horse/']")
                title = soup.title.string if soup.title else "なし"
                debug_info = f" [ページタイトル:{title}, テーブル数:{len(tables)}, 馬リンク:{len(horse_links)}]"
            return jsonify({
                "status": "error",
                "message": f"出走馬が取得できませんでした (race_id={race_id}){debug_info}。"
                           "出走表がまだ公開されていないか、レースが終了している可能性があります。"
                           "通常はレース3〜4日前から出走表が公開されます。"
            }), 404

        # 2. DBに保存
        db = Database()
        df_entry = pd.DataFrame(entries)

        # race_info: 取得済みsoupのタイトルから抽出
        info = {"race_id": race_id}
        if soup and soup.title:
            title_text = soup.title.string or ""
            # 例: "宝塚記念(G1) 5走表示 | 2026年6月14日 阪神11R"
            import re as _re
            m_date = _re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title_text)
            if m_date:
                info["date"] = f"{m_date.group(1)}-{int(m_date.group(2)):02d}-{int(m_date.group(3)):02d}"
            m_name = _re.match(r"([^\|]+)", title_text)
            if m_name:
                info["race_name"] = m_name.group(1).split("5走")[0].strip()
            m_venue = _re.search(r"\d{4}年\d+月\d+日\s+(\S+?)\d+R", title_text)
            if m_venue:
                info["course"] = m_venue.group(1)
            m_rnum = _re.search(r"(\d+)R", title_text)
            if m_rnum:
                info["race_number"] = int(m_rnum.group(1))
        # URLにrace_idがある場合はそこからも補完
        course_code = race_id[8:10]
        from config.settings import COURSE_CODES
        if "course" not in info:
            info["course"] = COURSE_CODES.get(course_code, course_code)
        info["course_code"] = course_code
        if "race_number" not in info:
            info["race_number"] = int(race_id[10:12])
        if "date" not in info:
            # race_idフォーマットが YYYYMMDDCCRR の場合のみ有効
            info["date"] = f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}"
        db.upsert_race_info(info)
        # race情報をentryにも埋め込む（FeatureBuilderのmergeで上書きされないよう）
        for entry in entries:
            entry["race_name"]   = info.get("race_name", "")
            entry["date"]        = info.get("date", "")
            entry["course"]      = info.get("course", "")
            entry["course_code"] = info.get("course_code", "")
            entry["race_number"] = info.get("race_number", 0)
            entry["distance"]    = info.get("distance", None)
            entry["surface"]     = info.get("surface", "")
            entry["track_condition"] = info.get("track_condition", "")
        df_entry = pd.DataFrame(entries)
        # 古い重複データを削除してから挿入
        with db.engine.connect() as conn:
            from sqlalchemy import text as _text
            conn.execute(_text("DELETE FROM race_results WHERE race_id = :r"), {"r": race_id})
            conn.commit()
        db.upsert_race_results(df_entry)

        # 2b. オッズ取得（単勝・複勝）してDBのentryを更新
        from jra_predictor.scraper.race_result import RaceResultScraper
        odds_scraper = RaceResultScraper()
        win_place_odds = odds_scraper._fetch_win_place_odds(race_id)
        if win_place_odds:
            logger.info(f"Fetched odds for {len(win_place_odds)} horses")
            for entry in entries:
                hn = entry.get("horse_number")
                if hn and hn in win_place_odds:
                    entry["win_odds"] = win_place_odds[hn].get("win_odds")
                    entry["popularity"] = None  # popularity determined by rank of win_odds
            # Set popularity by rank of win_odds
            valid = [(e["horse_number"], e.get("win_odds") or 9999) for e in entries if e.get("horse_number")]
            valid.sort(key=lambda x: x[1])
            pop_rank = {hn: i+1 for i, (hn, _) in enumerate(valid)}
            for entry in entries:
                hn = entry.get("horse_number")
                if hn:
                    entry["popularity"] = pop_rank.get(hn, 0)
            # Re-save with odds
            df_entry = pd.DataFrame(entries)
            with db.engine.connect() as conn:
                from sqlalchemy import text as _text
                conn.execute(_text("DELETE FROM race_results WHERE race_id = :r"), {"r": race_id})
                conn.commit()
            db.upsert_race_results(df_entry)
        else:
            logger.warning(f"Could not fetch odds for {race_id}")

        # 3. 各馬の過去成績を取得（DBになければnetkeibaから）
        horse_scraper = HorseProfileScraper()
        for horse_id in df_entry["horse_id"].dropna().unique():
            if not horse_id:
                continue
            existing = db.read_table("horse_history", f"horse_id='{horse_id}'")
            if existing.empty:
                history = horse_scraper.fetch_horse_history(horse_id)
                if history is not None:
                    db.upsert_horse_history(history)

        # 4. 特徴量構築 & 予測
        from jra_predictor.features import FeatureBuilder
        from jra_predictor.models import RacePredictor, ExpectedValueCalculator
        from jra_predictor.backtest.engine import BacktestEngine

        builder = FeatureBuilder(db)
        df_all = builder.build()
        df_race = df_all[df_all["race_id"] == race_id]

        if df_race.empty:
            return jsonify({
                "status": "error",
                "message": f"特徴量が構築できませんでした（horse_history不足の可能性）。出走馬数: {len(entries)}"
            }), 500

        win_model = RacePredictor("is_win")
        place_model = RacePredictor("is_place")
        win_model.load()
        place_model.load()

        ev_calc = ExpectedValueCalculator(win_model, place_model)

        # オッズ辞書を構築（スクレイピング済みの win_place_odds を優先）
        if win_place_odds:
            odds = {
                "tan": {hn: v["win_odds"] for hn, v in win_place_odds.items() if v.get("win_odds")},
                "fukusho": {hn: v.get("place_odds_min", 1.0) for hn, v in win_place_odds.items() if v.get("place_odds_min")},
                "wide": {},
                "umaren": {},
                "sanrenpuku": {},
            }
        else:
            bt = BacktestEngine(db)
            odds = bt._get_odds_for_race(race_id)
            if not any(odds.values()):
                odds = bt._build_odds_from_df(df_race)

        recs_df = ev_calc.recommend(df_race, odds, budget=10000)

        win_probs = win_model.predict_proba(df_race)
        place_probs = place_model.predict_proba(df_race)

        win_probs_arr = win_model.predict_proba(df_race.sort_values("horse_number"))
        place_probs_arr = place_model.predict_proba(df_race.sort_values("horse_number"))

        horses_out = []
        for i, (_, row) in enumerate(df_race.sort_values("horse_number").iterrows()):
            wo = row.get("win_odds")
            pop = row.get("popularity")
            hn = int(row.get("horse_number", 0) or 0)
            wp = float(win_probs_arr[i])
            pp = float(place_probs_arr[i])
            win_odds_val = float(wo) if wo and str(wo) not in ("nan", "None", "0.0", "0") else 0
            pop_val = int(pop) if pop and str(pop) not in ("nan", "None", "0") else 0
            # EV from scraped place odds
            fukusho_odds = odds.get("fukusho", {}).get(hn, 0)
            place_ev = round(pp * fukusho_odds, 3) if fukusho_odds else None
            tan_odds = odds.get("tan", {}).get(hn, win_odds_val)
            win_ev = round(wp * tan_odds, 3) if tan_odds else None
            horses_out.append({
                "number": hn,
                "frame": int(row.get("frame_number", 0) or 0),
                "name": str(row.get("horse_name", "")),
                "sex_age": str(row.get("sex_age", "") or ""),
                "jockey": str(row.get("jockey_name", "") or ""),
                "win_odds": win_odds_val,
                "popularity": pop_val,
                "place_odds": round(fukusho_odds, 1) if fukusho_odds else 0,
                "weight": 0,
                "weight_diff": 0,
                "win_prob": round(wp, 3),
                "place_prob": round(pp, 3),
                "place_ev": place_ev,
                "win_ev": win_ev,
                "score": int(min(99, max(1, pp * 200))),
                "finish_order": None,
            })

        allowed = {"複勝", "ワイド", "3連複"}
        recommendations = []
        if not recs_df.empty:
            for _, r in recs_df.iterrows():
                if str(r["bet_type"]) not in allowed:
                    continue
                recommendations.append({
                    "bet_type": str(r["bet_type"]),
                    "combination": str(r["combination"]),
                    "odds": float(r["odds"]),
                    "probability": float(r["probability"]),
                    "expected_value": float(r["expected_value"]),
                    "stake": int(r["stake"]),
                })

        def _val(row, key, fallback=""):
            v = row.get(key, fallback)
            return fallback if (v is None or str(v) in ("nan", "None", "NaT")) else v

        first = df_race.iloc[0]
        return jsonify({
            "status": "ok",
            "data": {
                "race_id": race_id,
                "race_name": info.get("race_name") or _val(first, "race_name") or race_id,
                "venue": info.get("course") or _val(first, "course"),
                "date": (info.get("date") or _val(first, "date", ""))[:10],
                "distance": info.get("distance") or int(_val(first, "distance", 0) or 0),
                "surface": info.get("surface") or _val(first, "surface"),
                "field_count": len(df_race),
                "horses": horses_out,
                "recommendations": recommendations,
                "has_recommendations": len(recommendations) > 0,
            }
        })

    except Exception as e:
        logger.exception(f"predict-url error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


def _safe_int(s):
    try:
        return int(str(s).replace(",", ""))
    except Exception:
        return None


def _safe_float(s):
    try:
        return float(str(s).replace(",", ""))
    except Exception:
        return None


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
