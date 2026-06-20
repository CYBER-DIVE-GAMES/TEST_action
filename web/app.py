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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS registered_races (
            race_id TEXT PRIMARY KEY,
            registered_at TEXT DEFAULT (datetime('now','localtime')),
            race_name TEXT,
            date TEXT,
            course TEXT,
            race_number INTEGER,
            surface TEXT,
            distance INTEGER,
            field_count INTEGER,
            url TEXT
        )
    """)
    # 既存DBへのカラム追加（エラー無視）
    try:
        conn.execute("ALTER TABLE registered_races ADD COLUMN url TEXT")
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS race_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            registered_at TEXT DEFAULT (datetime('now','localtime')),
            race_id TEXT NOT NULL,
            race_name TEXT,
            date TEXT,
            course TEXT,
            race_number INTEGER,
            surface TEXT,
            distance INTEGER,
            bet_type TEXT,
            combination TEXT,
            odds REAL,
            expected_value REAL,
            hit INTEGER,
            payout INTEGER
        )
    """)
    conn.commit()
    return conn


def _load_in_background():
    try:
        _cache["loading"] = True
        from jra_predictor.models import RacePredictor

        win_model = RacePredictor("is_win")
        place_model = RacePredictor("is_place")
        win_model.load()
        place_model.load()
        _cache["win_model"] = win_model
        _cache["place_model"] = place_model
        _cache["ready"] = True
        logger.info("Models loaded successfully")
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
    """登録済みレースをDBから読み込んで予測（再予測用）"""
    if not _cache["ready"]:
        if _cache["error"]:
            return jsonify({"status": "error", "message": _cache["error"]}), 500
        return jsonify({"status": "loading", "message": "モデル読み込み中です。しばらくお待ちください..."}), 202

    try:
        import pandas as pd
        from jra_predictor.data import Database
        from jra_predictor.models import RacePredictor, ExpectedValueCalculator
        from jra_predictor.backtest.engine import BacktestEngine
        from config.settings import COURSE_CODES

        db = Database()
        df_entry = db.read_table("race_results", f"race_id='{race_id}'")
        if df_entry.empty:
            return jsonify({"status": "error", "message": f"レースID {race_id} のデータがDBにありません"}), 404

        # インライン特徴量計算（predict-urlと同じ処理）
        df = df_entry.copy()
        for col in ["horse_number","frame_number","win_odds","popularity","weight_carried","horse_weight","distance"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0)

        n = len(df)
        df["sex"]               = df["sex_age"].astype(str).str.extract(r"([牡牝騸セ])")
        df["age"]               = pd.to_numeric(df["sex_age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        df["field_count"]       = n
        df["horse_number_ratio"]= df["horse_number"] / max(n, 1)
        df["frame_number_norm"] = df["frame_number"] / 8.0
        df["weight_handicap"]   = df["weight_carried"] - df["weight_carried"].mean()
        df["is_win"]  = 0
        df["is_place"] = 0
        df["distance"] = df["distance"].replace(0, 1600)
        df["distance_cat"] = pd.cut(df["distance"], bins=[0,1400,1800,2200,9999], labels=["sprint","mile","middle","long"])

        pop_min, pop_max = df["popularity"].min(), df["popularity"].max()
        df["popularity_norm"] = (df["popularity"] - pop_min) / (pop_max - pop_min + 1e-9)
        fav = df[df["popularity"] == 1]
        fav_o = float(fav["win_odds"].values[0]) if len(fav) else float(df["win_odds"].max() or 1)
        df["fav_odds"]      = fav_o
        df["relative_odds"] = df["win_odds"] / (fav_o + 1e-9)

        HIST_COLS = ["win_rate_3","win_rate_5","win_rate_10","place_rate_3","place_rate_5","place_rate_10",
                     "avg_popularity_3","avg_popularity_5","avg_odds_5","odds_change",
                     "prev_finish","prev2_finish","avg_last3f_5","days_since_last","career_runs",
                     "jockey_win_rate_30","jockey_win_rate_100","jockey_place_rate_30","jockey_place_rate_100",
                     "jockey_course_wins","jockey_dist_wins","trainer_win_rate_50","trainer_place_rate_50",
                     "horse_course_wins","horse_course_place","horse_dist_wins","horse_surface_wins","horse_condition_wins",
                     "avg_running_style","sire_win_rate","sire_place_rate","sire_dist_win_rate","avg_weight_3","weight_vs_avg"]
        for col in HIST_COLS:
            if col not in df.columns:
                df[col] = float("nan")

        # DBから過去成績を取得して特徴量を埋める
        race_date_str = str(df["date"].iloc[0])[:10] if "date" in df.columns and len(df) else str(date.today())
        course_code_h = race_id[4:6] if len(race_id) >= 6 else ""
        surface_h = str(df["surface"].iloc[0]) if "surface" in df.columns and len(df) else ""
        dist_h = int(float(df["distance"].iloc[0])) if "distance" in df.columns and len(df) else 1600
        hist = _fetch_history_features(
            df["horse_id"].tolist() if "horse_id" in df.columns else [],
            df["jockey_id"].tolist() if "jockey_id" in df.columns else [],
            df["trainer_id"].tolist() if "trainer_id" in df.columns else [],
            race_date_str, course_code_h, surface_h, dist_h
        )
        horse_hist   = hist.get("horse", {})
        jockey_hist  = hist.get("jockey", {})
        trainer_hist = hist.get("trainer", {})
        for col in ["win_rate_3","win_rate_5","win_rate_10","place_rate_3","place_rate_5","place_rate_10",
                    "avg_popularity_3","avg_popularity_5","avg_odds_5",
                    "prev_finish","prev2_finish","avg_last3f_5","days_since_last","career_runs",
                    "horse_course_wins","horse_course_place","horse_surface_wins",
                    "horse_condition_wins","avg_running_style",
                    "sire_win_rate","sire_place_rate","sire_dist_win_rate"]:
            df[col] = df["horse_id"].map(lambda hid: horse_hist.get(hid, {}).get(col, float("nan"))) \
                if "horse_id" in df.columns else float("nan")
        df["odds_change"] = df.apply(
            lambda row: (row["win_odds"] - horse_hist.get(row.get("horse_id",""), {}).get("_prev_odds", float("nan")))
            if horse_hist.get(row.get("horse_id",""), {}).get("_prev_odds") is not None else float("nan"), axis=1
        )
        # 馬体重トレンド
        df["avg_weight_3"] = df["horse_id"].map(
            lambda hid: horse_hist.get(hid, {}).get("_avg_weight_3", float("nan"))
        ) if "horse_id" in df.columns else float("nan")
        df["weight_vs_avg"] = df.apply(
            lambda row: float(row["horse_weight"]) - row["avg_weight_3"]
            if pd.notna(row.get("horse_weight")) and pd.notna(row.get("avg_weight_3")) else float("nan"), axis=1
        )
        for col in ["jockey_win_rate_30","jockey_win_rate_100","jockey_place_rate_30",
                    "jockey_place_rate_100","jockey_course_wins","jockey_dist_wins"]:
            df[col] = df["jockey_id"].map(lambda jid: jockey_hist.get(jid, {}).get(col, float("nan"))) \
                if "jockey_id" in df.columns else float("nan")
        for col in ["trainer_win_rate_50","trainer_place_rate_50"]:
            df[col] = df["trainer_id"].map(lambda tid: trainer_hist.get(tid, {}).get(col, float("nan"))) \
                if "trainer_id" in df.columns else float("nan")

        # クラス・変化系特徴量を追加
        race_name_val = str(df["race_name"].iloc[0]) if "race_name" in df.columns and len(df) else ""
        dist_val_int = int(float(df["distance"].iloc[0])) if "distance" in df.columns and len(df) else 1600
        df = _add_prediction_class_features(df, race_name_val, dist_val_int)
        # 前走クラス・距離変化・馬場変化をhorse_histから補完
        df["prev_race_class"] = df["horse_id"].map(
            lambda hid: horse_hist.get(hid, {}).get("prev_race_class", float("nan"))
        ) if "horse_id" in df.columns else float("nan")
        df["class_change"] = df["race_class"] - df["prev_race_class"]
        df["class_finish_index"] = df.apply(
            lambda r: r["class_change"] * (6 - min(r.get("prev_finish", 6) or 6, 6))
            if pd.notna(r.get("class_change")) and pd.notna(r.get("prev_finish")) else float("nan"), axis=1
        )
        df["distance_change"] = df["horse_id"].map(
            lambda hid: (dist_val_int - horse_hist.get(hid, {}).get("_prev_distance", float("nan")))
            if pd.notna(horse_hist.get(hid, {}).get("_prev_distance")) else float("nan")
        ) if "horse_id" in df.columns else float("nan")
        df["surface_change"] = df["horse_id"].map(
            lambda hid: (0 if horse_hist.get(hid, {}).get("_prev_surface") == surface_h else 1)
            if horse_hist.get(hid, {}).get("_prev_surface") else float("nan")
        ) if "horse_id" in df.columns else float("nan")

        NON_NUMERIC = {"race_id","horse_name","horse_id","jockey_name","jockey_id","trainer_name",
                       "race_name","course","course_code","surface","track_condition","sex_age","sex",
                       "margin","passing_order","distance_cat","date"}
        for col in df.columns:
            if df[col].dtype == object and col not in NON_NUMERIC:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        win_model   = _cache.get("win_model")   or RacePredictor("is_win")
        place_model = _cache.get("place_model") or RacePredictor("is_place")
        if not _cache.get("win_model"):   win_model.load()
        if not _cache.get("place_model"): place_model.load()

        df_sorted = df.sort_values("horse_number").reset_index(drop=True)
        win_probs   = win_model.predict_proba(df_sorted)
        place_probs = place_model.predict_proba(df_sorted)

        bt = BacktestEngine(db)
        odds_dict = bt._get_odds_for_race(race_id)
        if not any(odds_dict.values()):
            odds_dict = bt._build_odds_from_df(df_sorted)

        ev_calc = ExpectedValueCalculator(win_model, place_model)
        recs_df = ev_calc.recommend(df_sorted, odds_dict, budget=10000)

        # レース情報
        first = df_sorted.iloc[0]
        def sv(key, fallback=""):
            v = first.get(key, fallback)
            return fallback if (v is None or str(v) in ("nan","None","NaT","0")) else v

        horses_out = []
        for i, row in df_sorted.iterrows():
            hn = int(row["horse_number"] or 0)
            wp = float(win_probs[i])
            pp = float(place_probs[i])
            fo = odds_dict.get("fukusho", {}).get(hn, 0) or float(row.get("place_odds_min") or 0)
            to = odds_dict.get("tan", {}).get(hn, 0) or float(row.get("win_odds") or 0)
            wo = float(row.get("win_odds") or 0)
            horses_out.append({
                "number": hn, "frame": int(row.get("frame_number") or 1),
                "name": str(row.get("horse_name") or ""), "sex_age": str(row.get("sex_age") or ""),
                "jockey": str(row.get("jockey_name") or ""), "win_odds": wo,
                "popularity": int(row.get("popularity") or 0),
                "place_odds": round(fo, 1) if fo else 0,
                "weight": 0, "weight_diff": 0,
                "win_prob": round(wp,3), "place_prob": round(pp,3),
                "place_ev": round(pp*fo,3) if fo else None,
                "win_ev":   round(wp*to,3) if to else None,
                "score": int(min(99,max(1,pp*200))),
                "finish_order": int(row.get("finish_order") or 0) or None,
            })

        allowed = {"複勝","ワイド","3連複"}
        recs = [{"bet_type":str(r["bet_type"]),"combination":str(r["combination"]),
                 "odds":float(r["odds"]),"probability":float(r["probability"]),
                 "expected_value":float(r["expected_value"]),"stake":int(r["stake"])}
                for _,r in recs_df.iterrows() if str(r["bet_type"]) in allowed] if not recs_df.empty else []

        course_code = race_id[4:6] if len(race_id) >= 6 else ""
        return jsonify({"status": "ok", "data": {
            "race_id": race_id,
            "race_name": sv("race_name") or race_id,
            "venue": sv("course") or COURSE_CODES.get(course_code, course_code),
            "date": str(sv("date",""))[:10],
            "distance": int(float(sv("distance",0) or 0)),
            "surface": sv("surface"),
            "field_count": n,
            "horses": horses_out,
            "recommendations": recs,
            "has_recommendations": len(recs) > 0,
        }})
    except Exception as e:
        logger.exception(f"api_race error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


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


@app.route("/api/register-race", methods=["POST"])
def api_register_race():
    """レースを登録し、複勝買い目をrace_betsに保存"""
    data = request.json or {}
    race_id = data.get("race_id", "")
    if not race_id:
        return jsonify({"status": "error", "message": "race_id missing"}), 400
    try:
        conn = _get_log_conn()
        conn.execute("""
            INSERT OR REPLACE INTO registered_races
              (race_id, race_name, date, course, race_number, surface, distance, field_count, url)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            race_id,
            data.get("race_name", ""),
            data.get("date", ""),
            data.get("venue", ""),
            data.get("race_number", 0),
            data.get("surface", ""),
            data.get("distance", 0),
            data.get("field_count", 0),
            data.get("url", ""),
        ))
        # 複勝推奨を race_bets に自動保存
        recs = [r for r in (data.get("recommendations") or []) if r.get("bet_type") == "複勝"]
        for r in recs:
            conn.execute("""
                INSERT INTO race_bets
                  (race_id, race_name, date, course, race_number, surface, distance, bet_type, combination, odds, expected_value)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                race_id,
                data.get("race_name", ""),
                data.get("date", ""),
                data.get("venue", ""),
                data.get("race_number", 0),
                data.get("surface", ""),
                data.get("distance", 0),
                "複勝",
                r.get("combination", ""),
                r.get("odds", 0),
                r.get("expected_value", 0),
            ))
        conn.commit()
        conn.close()
        return jsonify({"status": "ok", "saved_bets": len(recs)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/registered-races")
def api_registered_races():
    """登録済みレース一覧（date未指定=今日JST、date=''=全件）"""
    from datetime import datetime, timezone, timedelta
    JST = timezone(timedelta(hours=9))
    today = datetime.now(JST).strftime("%Y-%m-%d")
    date_param = request.args.get("date", None)
    # dateパラメータなし→今日、空文字→全件
    if date_param is None:
        date_q = today
        all_records = False
    elif date_param == "":
        date_q = None
        all_records = True
    else:
        date_q = date_param
        all_records = False
    try:
        conn = _get_log_conn()
        if all_records:
            rows = conn.execute(
                "SELECT * FROM registered_races ORDER BY date DESC, race_number ASC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM registered_races WHERE date=? ORDER BY race_number ASC",
                (date_q,)
            ).fetchall()
        conn.close()
        return jsonify({"status": "ok", "races": [dict(r) for r in rows], "date": date_q or "all"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/registered-races/<race_id>", methods=["DELETE"])
def api_delete_registered_race(race_id: str):
    """登録済みレースを削除"""
    try:
        conn = _get_log_conn()
        conn.execute("DELETE FROM registered_races WHERE race_id=?", (race_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/race-bets")
def api_race_bets():
    """買い目ログ（複勝のみ、買い目があるレースのみ）"""
    date_q = request.args.get("date", "")
    try:
        conn = _get_log_conn()
        if date_q:
            rows = conn.execute(
                "SELECT * FROM race_bets WHERE date=? ORDER BY date DESC, race_number DESC, id DESC",
                (date_q,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM race_bets ORDER BY date DESC, race_number DESC, id DESC LIMIT 500"
            ).fetchall()
        conn.close()
        return jsonify({"status": "ok", "bets": [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/race-bets/<int:bet_id>", methods=["PATCH"])
def api_update_race_bet(bet_id: int):
    data = request.json
    hit = data.get("hit")
    payout = data.get("payout", 0)
    try:
        conn = _get_log_conn()
        conn.execute("UPDATE race_bets SET hit=?, payout=? WHERE id=?", (hit, payout, bet_id))
        conn.commit()
        conn.close()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/refresh-odds/<race_id>", methods=["POST"])
def api_refresh_odds(race_id: str):
    """オッズを再取得（登録済みURLから）"""
    try:
        conn = _get_log_conn()
        row = conn.execute("SELECT url FROM registered_races WHERE race_id=?", (race_id,)).fetchone()
        conn.close()
        stored_url = row["url"] if row and row["url"] else None
    except Exception:
        stored_url = None

    try:
        from jra_predictor.scraper.race_result import RaceResultScraper
        from jra_predictor.data import Database
        import pandas as pd

        scraper = RaceResultScraper()
        win_place_odds = scraper._fetch_win_place_odds(race_id)
        if not win_place_odds:
            return jsonify({"status": "error", "message": "オッズ取得失敗（レース前後はオッズページがない場合があります）"}), 500

        db = Database()
        df = db.read_table("race_results")
        df_race = df[df["race_id"] == race_id].copy()
        if df_race.empty:
            return jsonify({"status": "error", "message": "レースデータなし"}), 404

        valid = [(hn, win_place_odds[hn].get("win_odds", 9999)) for hn in win_place_odds if win_place_odds[hn].get("win_odds")]
        valid.sort(key=lambda x: x[1])
        pop_rank = {hn: i+1 for i, (hn, _) in enumerate(valid)}

        entries = df_race.to_dict("records")
        for entry in entries:
            hn = int(entry.get("horse_number") or 0)
            if hn in win_place_odds:
                entry["win_odds"] = win_place_odds[hn].get("win_odds")
                entry["place_odds_min"] = win_place_odds[hn].get("place_odds_min")
                entry["popularity"] = pop_rank.get(hn, 0)

        from sqlalchemy import text as _text
        with db.engine.connect() as conn2:
            conn2.execute(_text("DELETE FROM race_results WHERE race_id = :r"), {"r": race_id})
            conn2.commit()
        db.upsert_race_results(pd.DataFrame(entries))

        return jsonify({"status": "ok", "updated": len(win_place_odds)})
    except Exception as e:
        logger.exception(f"refresh-odds error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/repredict/<race_id>", methods=["POST"])
def api_repredict(race_id: str):
    """登録済みURLから再予測"""
    try:
        conn = _get_log_conn()
        row = conn.execute("SELECT url FROM registered_races WHERE race_id=?", (race_id,)).fetchone()
        conn.close()
        if not row or not row["url"]:
            return jsonify({"status": "error", "message": "URLが保存されていません。ホームからURLを貼って再予測してください。"}), 404
        url = row["url"]
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    # predict-url と同じロジックを内部で呼ぶ
    with app.test_request_context('/api/predict-url', method='POST',
                                   json={"url": url},
                                   content_type='application/json'):
        return api_predict_url()


@app.route("/api/actual-stats")
def api_actual_stats():
    """登録レースの複勝実績統計"""
    try:
        conn = _get_log_conn()
        rows = conn.execute(
            "SELECT date, hit, payout, odds FROM race_bets WHERE bet_type='複勝' AND hit IS NOT NULL ORDER BY date"
        ).fetchall()
        conn.close()
        total = len(rows)
        hits = sum(1 for r in rows if r["hit"] == 1)
        payout = sum(r["payout"] or 0 for r in rows)
        # 1betあたり100円として計算
        stake = total * 100
        hit_rate = round(hits / total * 100, 1) if total else None
        roi = round(payout / stake * 100, 1) if stake else None
        # 月別集計
        from collections import defaultdict
        monthly = defaultdict(lambda: {"bets": 0, "hits": 0, "payout": 0})
        for r in rows:
            m = str(r["date"])[:7] if r["date"] else "不明"
            monthly[m]["bets"] += 1
            monthly[m]["hits"] += r["hit"] or 0
            monthly[m]["payout"] += r["payout"] or 0
        months_sorted = sorted(monthly.keys())
        return jsonify({
            "status": "ok",
            "total_bets": total,
            "hits": hits,
            "hit_rate": hit_rate,
            "roi": roi,
            "profit": payout - stake,
            "monthly_labels": months_sorted[-12:],
            "monthly_roi": [round(monthly[m]["payout"] / (monthly[m]["bets"] * 100) * 100, 1) if monthly[m]["bets"] else 0 for m in months_sorted[-12:]],
            "monthly_hit_rate": [round(monthly[m]["hits"] / monthly[m]["bets"] * 100, 1) if monthly[m]["bets"] else 0 for m in months_sorted[-12:]],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



def _add_prediction_class_features(df, race_name: str, distance: int):
    """予測時のクラス・変化系特徴量を追加（_fetch_history_featuresの補完）"""
    import re

    def _race_class(name):
        if not isinstance(name, str):
            return 5
        if any(k in name for k in ["GI","G1","有馬","天皇賞","ジャパン","宝塚","安田"]):
            return 1
        if any(k in name for k in ["GII","G2"]):
            return 2
        if any(k in name for k in ["GIII","G3"]):
            return 3
        if "(L)" in name or "リステッド" in name:
            return 4
        if "OP" in name or "オープン" in name:
            return 5
        if "3勝" in name or "1600万" in name:
            return 6
        if "2勝" in name or "1000万" in name:
            return 7
        if "1勝" in name or "500万" in name:
            return 8
        if "未勝利" in name:
            return 9
        if "新馬" in name or "メイクデビュー" in name:
            return 10
        return 5

    df["race_class"]     = _race_class(race_name)
    df["is_shinsoba"]    = 1 if isinstance(race_name, str) and ("新馬" in race_name or "メイクデビュー" in race_name) else 0
    df["place_base_rate"] = 3.0 / df["field_count"].clip(lower=4)

    # 前走クラス・距離・馬場はDBから取得済みの_prev系を使う
    # （_fetch_history_featuresで_prev_race_class等を取得していないためNaN）
    for col in ["prev_race_class","class_change","class_finish_index",
                "distance_change","surface_change","course_change",
                "n_frontrunners","pace_pressure"]:
        if col not in df.columns:
            df[col] = float("nan")

    return df


def _fetch_history_features(horse_ids: list, jockey_ids: list, trainer_ids: list,
                             race_date: str, course_code: str, surface: str, distance: int) -> dict:
    """DBから馬・騎手・調教師の過去成績を取得して特徴量dictを返す

    Returns: {horse_id: {feature_name: value, ...}, ...}
    """
    try:
        import sqlite3 as _sqlite3
        import numpy as np
        conn = _sqlite3.connect(_DB_PATH)
        conn.row_factory = _sqlite3.Row
    except Exception:
        return {}

    result = {hid: {} for hid in horse_ids}

    try:
        # ---- 馬の過去成績（race_results から） ----
        for horse_id in horse_ids:
            if not horse_id:
                continue
            rows = conn.execute(
                """SELECT finish_order, win_odds, popularity, last_3f, date,
                          course_code, surface, is_win, is_place,
                          distance, track_condition, passing_order, horse_weight,
                          race_name
                   FROM race_results
                   WHERE horse_id=? AND date < ?
                   ORDER BY date DESC LIMIT 15""",
                (horse_id, race_date)
            ).fetchall()

            if not rows:
                continue

            feats = result[horse_id]
            fo = [r["finish_order"] for r in rows if r["finish_order"] is not None]
            is_win_list  = [r["is_win"]   or 0 for r in rows]
            is_place_list = [r["is_place"] or 0 for r in rows]
            odds_list    = [r["win_odds"]  for r in rows if r["win_odds"] is not None]
            pop_list     = [r["popularity"] for r in rows if r["popularity"] is not None]
            lf_list      = [r["last_3f"]   for r in rows if r["last_3f"] is not None]

            feats["prev_finish"]  = float(fo[0])  if len(fo) > 0 else np.nan
            feats["prev2_finish"] = float(fo[1])  if len(fo) > 1 else np.nan

            for w, col in [(3,"3"),(5,"5"),(10,"10")]:
                w_win  = is_win_list[:w]
                w_plc  = is_place_list[:w]
                w_pop  = pop_list[:w]
                w_odds = odds_list[:w]
                feats[f"win_rate_{col}"]        = float(np.mean(w_win))  if w_win  else np.nan
                feats[f"place_rate_{col}"]      = float(np.mean(w_plc))  if w_plc  else np.nan
                feats[f"avg_popularity_{col}"]  = float(np.mean(w_pop))  if w_pop  else np.nan
                if col in ("5",):
                    feats["avg_odds_5"] = float(np.mean(w_odds)) if w_odds else np.nan

            feats["avg_last3f_5"] = float(np.mean(lf_list[:5])) if lf_list else np.nan
            feats["career_runs"]  = len(rows)

            # 前走からの休養日数
            if rows:
                from datetime import date as _date
                try:
                    prev_d = _date.fromisoformat(rows[0]["date"])
                    this_d = _date.fromisoformat(race_date)
                    feats["days_since_last"] = (this_d - prev_d).days
                except Exception:
                    pass

            # オッズ変化（前走オッズは rows[0] に入っているが今のオッズは呼び出し側で設定済み）
            if odds_list:
                feats["_prev_odds"] = float(odds_list[0])  # 呼び出し側でodds_changeを計算

            # コース別複勝率
            course_rows = [r for r in rows if r["course_code"] == course_code]
            feats["horse_course_wins"]  = float(np.mean([r["is_win"]   or 0 for r in course_rows])) if course_rows else np.nan
            feats["horse_course_place"] = float(np.mean([r["is_place"] or 0 for r in course_rows])) if course_rows else np.nan

            # 馬場（芝/ダ）別勝率
            surf_rows = [r for r in rows if r["surface"] == surface]
            feats["horse_surface_wins"] = float(np.mean([r["is_win"] or 0 for r in surf_rows])) if surf_rows else np.nan

            # 馬場状態別（良・稍重・重・不良）勝率
            cond_rows_map = {}
            for r in rows:
                tc = r["track_condition"] if r["track_condition"] else None
                if tc:
                    cond_rows_map.setdefault(tc, []).append(r)
            # 全馬場状態の平均（当日の馬場状態が予測時点で不明のため全体平均）
            all_cond = [r for r in rows if r["track_condition"]]
            feats["horse_condition_wins"] = float(np.mean([r["is_win"] or 0 for r in all_cond])) if all_cond else np.nan

            # 脚質（passing_orderから推定）
            def _est_style(passing):
                if not passing:
                    return np.nan
                positions = [int(x) for x in str(passing).split("-") if x.strip().isdigit()]
                if not positions:
                    return np.nan
                avg_pos = np.mean(positions)
                return 1 if avg_pos <= 3 else (2 if avg_pos <= 6 else 3)

            styles = [_est_style(r["passing_order"]) for r in rows if r["passing_order"]]
            styles = [s for s in styles if not np.isnan(s)]
            feats["avg_running_style"] = float(np.mean(styles[:5])) if styles else np.nan

            # 馬体重トレンド
            weights = [r["horse_weight"] for r in rows if r["horse_weight"]]
            feats["_avg_weight_3"] = float(np.mean(weights[:3])) if weights else np.nan

            # 前走クラス・距離・馬場変化
            def _race_class_from_name(name):
                if not isinstance(name, str): return 5
                if any(k in name for k in ["GI","G1","有馬","天皇賞","宝塚","安田"]): return 1
                if any(k in name for k in ["GII","G2"]): return 2
                if any(k in name for k in ["GIII","G3"]): return 3
                if "(L)" in name or "リステッド" in name: return 4
                if "OP" in name or "オープン" in name: return 5
                if "3勝" in name: return 6
                if "2勝" in name: return 7
                if "1勝" in name: return 8
                if "未勝利" in name: return 9
                if "新馬" in name or "メイクデビュー" in name: return 10
                return 5

            if rows:
                prev = rows[0]
                prev_cls = _race_class_from_name(prev["race_name"]) if prev["race_name"] else 5
                feats["prev_race_class"] = float(prev_cls)
                feats["_prev_distance"]  = float(prev["distance"]) if prev["distance"] else np.nan
                feats["_prev_surface"]   = prev["surface"] if prev["surface"] else None

        # ---- 騎手の過去成績 ----
        jockey_feats = {}
        for jid in set(jockey_ids):
            if not jid:
                continue
            rows = conn.execute(
                """SELECT is_win, is_place, course_code, distance
                   FROM race_results
                   WHERE jockey_id=? AND date < ?
                   ORDER BY date DESC LIMIT 120""",
                (jid, race_date)
            ).fetchall()
            if not rows:
                continue

            iw  = [r["is_win"]   or 0 for r in rows]
            ipl = [r["is_place"] or 0 for r in rows]

            f = {}
            f["jockey_win_rate_30"]   = float(np.mean(iw[:30]))  if len(iw) >= 5 else np.nan
            f["jockey_win_rate_100"]  = float(np.mean(iw[:100])) if len(iw) >= 5 else np.nan
            f["jockey_place_rate_30"] = float(np.mean(ipl[:30])) if len(ipl) >= 5 else np.nan
            f["jockey_place_rate_100"]= float(np.mean(ipl[:100]))if len(ipl) >= 5 else np.nan

            # コース別
            cc_rows = [r for r in rows if r["course_code"] == course_code]
            f["jockey_course_wins"] = float(np.mean([r["is_win"] or 0 for r in cc_rows])) if cc_rows else np.nan

            # 距離帯別
            dist_cat = "sprint" if distance <= 1400 else ("mile" if distance <= 1800 else ("middle" if distance <= 2200 else "long"))
            dist_ranges = {"sprint":(0,1400),"mile":(1401,1800),"middle":(1801,2200),"long":(2201,9999)}
            lo, hi = dist_ranges[dist_cat]
            dc_rows = [r for r in rows if r["distance"] and lo <= r["distance"] <= hi]
            f["jockey_dist_wins"] = float(np.mean([r["is_win"] or 0 for r in dc_rows])) if dc_rows else np.nan

            jockey_feats[jid] = f

        # ---- 調教師の過去成績 ----
        trainer_feats = {}
        for tid in set(trainer_ids):
            if not tid:
                continue
            rows = conn.execute(
                """SELECT is_win, is_place FROM race_results
                   WHERE trainer_id=? AND date < ?
                   ORDER BY date DESC LIMIT 60""",
                (tid, race_date)
            ).fetchall()
            if not rows:
                continue
            iw  = [r["is_win"]   or 0 for r in rows]
            ipl = [r["is_place"] or 0 for r in rows]
            trainer_feats[tid] = {
                "trainer_win_rate_50":   float(np.mean(iw[:50]))  if len(iw) >= 5 else np.nan,
                "trainer_place_rate_50": float(np.mean(ipl[:50])) if len(ipl) >= 5 else np.nan,
            }

        # ---- 血統（父別成績）----
        sire_feats = {}
        # horse_profileから父名を取得
        placeholders = ",".join("?" * len(horse_ids))
        if horse_ids:
            profile_rows = conn.execute(
                f"SELECT horse_id, sire FROM horse_profile WHERE horse_id IN ({placeholders})",
                horse_ids
            ).fetchall() if placeholders else []

            # 父名 → horse_idリスト
            sire_map = {}  # horse_id -> sire
            for pr in profile_rows:
                if pr["sire"]:
                    sire_map[pr["horse_id"]] = pr["sire"]

            # 父別の全成績を集計
            sires = list(set(sire_map.values()))
            for sire in sires:
                sire_rows = conn.execute(
                    """SELECT r.is_win, r.is_place, r.distance
                       FROM race_results r
                       JOIN horse_profile p ON r.horse_id = p.horse_id
                       WHERE p.sire=? AND r.date < ?
                       ORDER BY r.date DESC LIMIT 500""",
                    (sire, race_date)
                ).fetchall()
                if len(sire_rows) < 10:
                    continue
                iw  = [r["is_win"]   or 0 for r in sire_rows]
                ipl = [r["is_place"] or 0 for r in sire_rows]
                dist_cat = "sprint" if distance <= 1400 else ("mile" if distance <= 1800 else ("middle" if distance <= 2200 else "long"))
                dist_ranges = {"sprint":(0,1400),"mile":(1401,1800),"middle":(1801,2200),"long":(2201,9999)}
                lo, hi = dist_ranges[dist_cat]
                dist_rows_iw = [r["is_win"] or 0 for r in sire_rows if r["distance"] and lo <= r["distance"] <= hi]
                sire_feats[sire] = {
                    "sire_win_rate":      float(np.mean(iw)),
                    "sire_place_rate":    float(np.mean(ipl)),
                    "sire_dist_win_rate": float(np.mean(dist_rows_iw)) if dist_rows_iw else np.nan,
                }

            # 馬ごとに血統特徴量をセット
            for horse_id in horse_ids:
                sire = sire_map.get(horse_id)
                if sire and sire in sire_feats:
                    result[horse_id].update(sire_feats[sire])

    except Exception as e:
        logger.warning(f"_fetch_history_features error: {e}")
    finally:
        conn.close()

    return {"horse": result, "jockey": jockey_feats, "trainer": trainer_feats}


@app.route("/api/predict-url", methods=["POST"])
def api_predict_url():
    """netkeibaのURLを受け取って予測を返す"""
    import re, math
    import pandas as pd
    data = request.json or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"status": "error", "message": "URLを入力してください"}), 400

    m = re.search(r"race_id=(\d+)", url)
    if not m:
        return jsonify({"status": "error", "message": "URLにrace_idが見つかりません"}), 400
    race_id = m.group(1)

    try:
        from jra_predictor.scraper.base import get_with_browser
        from jra_predictor.models import RacePredictor, ExpectedValueCalculator
        from jra_predictor.scraper.race_result import RaceResultScraper
        from config.settings import COURSE_CODES

        # ── Step 1: ユーザーのURLをそのままPlaywrightで開く ──────────────────
        logger.info(f"predict-url: loading {url}")
        soup = get_with_browser(url, wait_selector="tr.HorseList", timeout_ms=30000)
        if soup is None:
            return jsonify({"status": "error", "message": "ページを開けませんでした。ネット接続を確認してください。"}), 500

        # ── Step 2: 馬リストをパース ──────────────────────────────────────────
        rows = [tr for tr in soup.select("tr.HorseList")
                if not any("OikiriData" in c for c in (tr.get("class") or []))]

        if not rows:
            return jsonify({
                "status": "error",
                "message": "出走馬が取得できませんでした。出走表がまだ公開されていないか、レースが終了している可能性があります。"
            }), 404

        entries = []
        for tr in rows:
            tds = tr.select("td")
            texts = [td.get_text(" ", strip=True) for td in tds]
            if len(texts) < 6:
                continue
            horse_link = tr.select_one("a[href*='/horse/']")
            jockey_link = tr.select_one("a[href*='/jockey/']")

            horse_id = ""
            horse_name = horse_link.get_text(strip=True) if horse_link else texts[3] if len(texts) > 3 else ""
            if horse_link:
                hm = re.search(r"/horse/(\w+)", horse_link["href"])
                horse_id = hm.group(1) if hm else ""

            jockey_name = jockey_link.get_text(strip=True) if jockey_link else ""
            jockey_id = ""
            if jockey_link:
                jm = re.search(r"/jockey/(\w+)", jockey_link["href"])
                jockey_id = jm.group(1) if jm else ""

            # 性齢: "牡5" パターンを抽出
            raw_sex_age = texts[4] if len(texts) > 4 else ""
            sa_m = re.search(r"([牡牝騸セ]\d+)", raw_sex_age)
            sex_age = sa_m.group(1) if sa_m else raw_sex_age[:3]

            # 斤量
            raw_wc = texts[5] if len(texts) > 5 else ""
            wc_m = re.search(r"(\d+\.?\d*)", raw_wc)
            weight_carried = float(wc_m.group(1)) if wc_m else 55.0

            entries.append({
                "race_id": race_id,
                "frame_number": int(texts[0]) if texts[0].isdigit() else None,
                "horse_number": int(texts[1]) if len(texts) > 1 and texts[1].isdigit() else None,
                "horse_name": horse_name,
                "horse_id": horse_id,
                "sex_age": sex_age,
                "weight_carried": weight_carried,
                "jockey_name": jockey_name,
                "jockey_id": jockey_id,
                "finish_order": None, "horse_weight": None,
                "win_odds": None, "popularity": None,
                "is_win": 0, "is_place": 0,
            })

        logger.info(f"predict-url: parsed {len(entries)} horses")

        # ── Step 3: レース情報をページタイトルから取得 ──────────────────────
        course_code = race_id[4:6] if len(race_id) >= 6 else ""
        race_number = int(race_id[10:12]) if len(race_id) >= 12 else 0
        info = {
            "race_id": race_id,
            "course": COURSE_CODES.get(course_code, course_code),
            "course_code": course_code,
            "race_number": race_number,
            "race_name": "",
            "date": "",
            "distance": 0,
            "surface": "",
        }
        title_tag = soup.select_one("title")
        title_text = title_tag.get_text() if title_tag else ""
        # タイトル例: "津軽海峡特別(2勝クラス) 5走表示 | 2026年6月14日 函館11R"
        date_m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title_text)
        if date_m:
            info["date"] = f"{date_m.group(1)}-{int(date_m.group(2)):02d}-{int(date_m.group(3)):02d}"

        name_m = re.match(r"(.+?)(?:\s*(?:出走表|出馬表)\s*|\s*5走|\s*\|)", title_text)
        if name_m:
            info["race_name"] = name_m.group(1).strip()

        # 距離・芝ダート
        race_data_div = soup.select_one(".RaceData01") or soup.select_one(".race_data")
        if race_data_div:
            rd_text = race_data_div.get_text()
            dist_m = re.search(r"(芝|ダ|障)(\d+)m", rd_text)
            if dist_m:
                info["surface"] = dist_m.group(1)
                info["distance"] = int(dist_m.group(2))

        for entry in entries:
            entry["race_name"]   = info["race_name"]
            entry["date"]        = info["date"]
            entry["course"]      = info["course"]
            entry["course_code"] = info["course_code"]
            entry["distance"]    = info["distance"] or None
            entry["surface"]     = info["surface"]

        # ── Step 4: オッズ取得 ────────────────────────────────────────────────
        scraper = RaceResultScraper()
        win_place_odds = scraper._fetch_win_place_odds(race_id)
        if win_place_odds:
            for entry in entries:
                hn = entry.get("horse_number")
                if hn and hn in win_place_odds:
                    entry["win_odds"] = win_place_odds[hn].get("win_odds")
            # 人気順を単勝オッズ順で設定
            valid = sorted(
                [(e["horse_number"], e.get("win_odds") or 9999) for e in entries if e.get("horse_number")],
                key=lambda x: x[1]
            )
            pop_rank = {hn: i+1 for i, (hn, _) in enumerate(valid)}
            for entry in entries:
                if entry.get("horse_number"):
                    entry["popularity"] = pop_rank.get(entry["horse_number"], 0)
        else:
            logger.warning(f"オッズ取得失敗: {race_id}")

        # ── Step 5: インライン特徴量計算 ──────────────────────────────────────
        df = pd.DataFrame(entries)
        df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce").fillna(0)
        df["frame_number"] = pd.to_numeric(df["frame_number"], errors="coerce").fillna(1)
        df["win_odds"]     = pd.to_numeric(df["win_odds"], errors="coerce").fillna(0)
        df["popularity"]   = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
        df["weight_carried"] = pd.to_numeric(df["weight_carried"], errors="coerce").fillna(55)
        df["horse_weight"] = pd.to_numeric(df.get("horse_weight"), errors="coerce").fillna(0)
        df["distance"]     = pd.to_numeric(df["distance"], errors="coerce").fillna(info.get("distance") or 1600)

        n = len(df)
        df["sex"]               = df["sex_age"].str.extract(r"([牡牝騸セ])")
        df["age"]               = pd.to_numeric(df["sex_age"].str.extract(r"(\d+)")[0], errors="coerce")
        df["field_count"]       = n
        df["horse_number_ratio"]= df["horse_number"] / n
        df["frame_number_norm"] = df["frame_number"] / 8.0
        df["weight_handicap"]   = df["weight_carried"] - df["weight_carried"].mean()
        df["is_win"]  = 0
        df["is_place"] = 0

        pop_min, pop_max = df["popularity"].min(), df["popularity"].max()
        df["popularity_norm"] = (df["popularity"] - pop_min) / (pop_max - pop_min + 1e-9)

        fav = df[df["popularity"] == 1]
        fav_o = float(fav["win_odds"].values[0]) if len(fav) else float(df["win_odds"].max() or 1)
        df["fav_odds"]      = fav_o
        df["relative_odds"] = df["win_odds"] / (fav_o + 1e-9)

        df["distance_cat"] = pd.cut(df["distance"], bins=[0, 1400, 1800, 2200, 9999],
                                     labels=["sprint", "mile", "middle", "long"])

        # ── DBから過去成績を取得して特徴量を埋める ──────────────────────────
        horse_ids_list   = df["horse_id"].tolist()
        jockey_ids_list  = df["jockey_id"].tolist() if "jockey_id" in df.columns else []
        trainer_ids_list = df["trainer_id"].tolist() if "trainer_id" in df.columns else []
        race_date_str = info.get("date") or str(date.today())
        dist_val = int(df["distance"].iloc[0]) if len(df) else 1600

        hist = _fetch_history_features(
            horse_ids_list, jockey_ids_list, trainer_ids_list,
            race_date_str, info.get("course_code",""), info.get("surface",""), dist_val
        )
        horse_hist  = hist.get("horse", {})
        jockey_hist = hist.get("jockey", {})
        trainer_hist= hist.get("trainer", {})

        # 馬の特徴量をDFに書き込む
        for col in ["win_rate_3","win_rate_5","win_rate_10","place_rate_3","place_rate_5","place_rate_10",
                    "avg_popularity_3","avg_popularity_5","avg_odds_5",
                    "prev_finish","prev2_finish","avg_last3f_5","days_since_last","career_runs",
                    "horse_course_wins","horse_course_place","horse_surface_wins",
                    "horse_condition_wins","avg_running_style",
                    "sire_win_rate","sire_place_rate","sire_dist_win_rate"]:
            df[col] = df["horse_id"].map(lambda hid: horse_hist.get(hid, {}).get(col, float("nan")))

        # odds_change = 今のオッズ - 前走オッズ
        df["odds_change"] = df.apply(
            lambda row: (row["win_odds"] - horse_hist.get(row["horse_id"], {}).get("_prev_odds", float("nan")))
            if horse_hist.get(row["horse_id"], {}).get("_prev_odds") is not None else float("nan"), axis=1
        )

        # 馬体重トレンド
        df["avg_weight_3"] = df["horse_id"].map(
            lambda hid: horse_hist.get(hid, {}).get("_avg_weight_3", float("nan"))
        )
        df["weight_vs_avg"] = df.apply(
            lambda row: float(row["horse_weight"]) - row["avg_weight_3"]
            if pd.notna(row.get("horse_weight")) and pd.notna(row.get("avg_weight_3")) else float("nan"), axis=1
        )

        # 騎手特徴量
        for col in ["jockey_win_rate_30","jockey_win_rate_100","jockey_place_rate_30",
                    "jockey_place_rate_100","jockey_course_wins","jockey_dist_wins"]:
            df[col] = df["jockey_id"].map(lambda jid: jockey_hist.get(jid, {}).get(col, float("nan"))) \
                if "jockey_id" in df.columns else float("nan")

        # 調教師特徴量
        for col in ["trainer_win_rate_50","trainer_place_rate_50"]:
            df[col] = df["trainer_id"].map(lambda tid: trainer_hist.get(tid, {}).get(col, float("nan"))) \
                if "trainer_id" in df.columns else float("nan")

        # クラス・変化系特徴量
        df = _add_prediction_class_features(df, info.get("race_name",""), dist_val)
        df["prev_race_class"] = df["horse_id"].map(
            lambda hid: horse_hist.get(hid, {}).get("prev_race_class", float("nan"))
        )
        df["class_change"] = df["race_class"] - df["prev_race_class"]
        df["class_finish_index"] = df.apply(
            lambda r: r["class_change"] * (6 - min(r.get("prev_finish", 6) or 6, 6))
            if pd.notna(r.get("class_change")) and pd.notna(r.get("prev_finish")) else float("nan"), axis=1
        )
        surf_val = info.get("surface","")
        df["distance_change"] = df["horse_id"].map(
            lambda hid: (dist_val - horse_hist.get(hid, {}).get("_prev_distance", float("nan")))
            if pd.notna(horse_hist.get(hid, {}).get("_prev_distance")) else float("nan")
        )
        df["surface_change"] = df["horse_id"].map(
            lambda hid: (0 if horse_hist.get(hid, {}).get("_prev_surface") == surf_val else 1)
            if horse_hist.get(hid, {}).get("_prev_surface") else float("nan")
        )
        for col in ["horse_dist_wins","n_frontrunners","pace_pressure","course_change"]:
            if col not in df.columns:
                df[col] = float("nan")

        filled = df[["horse_id","prev_finish","place_rate_5","jockey_win_rate_30","trainer_win_rate_50"]].head(3)
        logger.info(f"DB history features sample:\n{filled.to_string()}")

        # object → numeric 変換
        NON_NUMERIC = {"race_id","horse_name","horse_id","jockey_name","jockey_id",
                       "trainer_name","race_name","course","course_code","surface",
                       "track_condition","sex_age","sex","margin","passing_order",
                       "distance_cat","date"}
        for col in df.columns:
            if df[col].dtype == object and col not in NON_NUMERIC:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Step 6: モデル予測 ────────────────────────────────────────────────
        win_model   = _cache.get("win_model")   or RacePredictor("is_win")
        place_model = _cache.get("place_model") or RacePredictor("is_place")
        if not _cache.get("win_model"):   win_model.load()
        if not _cache.get("place_model"): place_model.load()

        df_sorted = df.sort_values("horse_number").reset_index(drop=True)
        win_probs   = win_model.predict_proba(df_sorted)
        place_probs = place_model.predict_proba(df_sorted)
        # オッズ系feature除外の純粋能力スコア（EV計算に使用）
        ai_scores   = place_model.predict_proba(df_sorted, exclude_odds=True)

        # AIスコアをレース内で正規化して0-100の相対スコアに変換
        score_sum = ai_scores.sum()
        ai_scores_norm = ai_scores / score_sum if score_sum > 0 else ai_scores

        ev_calc = ExpectedValueCalculator(win_model, place_model)
        odds_dict = {
            "tan":       {hn: v["win_odds"]       for hn, v in (win_place_odds or {}).items() if v.get("win_odds")},
            "fukusho":   {hn: v.get("place_odds_min", 0) for hn, v in (win_place_odds or {}).items() if v.get("place_odds_min")},
            "wide": {}, "umaren": {}, "sanrenpuku": {},
        }
        # AIスコアベースのfukusho probabilityでEV計算
        ai_score_map = {int(df_sorted.iloc[i]["horse_number"]): float(ai_scores[i])
                        for i in range(len(df_sorted))}
        recs_df = ev_calc.recommend(df_sorted, odds_dict, budget=10000,
                                    place_prob_override=ai_score_map)

        horses_out = []
        for i, row in df_sorted.iterrows():
            hn = int(row["horse_number"] or 0)
            wp = float(win_probs[i])
            pp = float(place_probs[i])
            ai_s = float(ai_scores[i])
            fukusho_odds = odds_dict["fukusho"].get(hn, 0)
            tan_odds     = odds_dict["tan"].get(hn, 0)
            wo = float(row["win_odds"] or 0)
            horses_out.append({
                "number":     hn,
                "frame":      int(row.get("frame_number") or 1),
                "name":       str(row.get("horse_name") or ""),
                "sex_age":    str(row.get("sex_age") or ""),
                "jockey":     str(row.get("jockey_name") or ""),
                "win_odds":   wo,
                "popularity": int(row.get("popularity") or 0),
                "place_odds": round(fukusho_odds, 1) if fukusho_odds else 0,
                "weight": 0, "weight_diff": 0,
                "win_prob":   round(wp, 3),
                "place_prob": round(pp, 3),
                "ai_score":   round(ai_s, 4),
                "place_ev":   round(ai_s * fukusho_odds, 3) if fukusho_odds else None,
                "win_ev":     round(wp * tan_odds, 3) if tan_odds else None,
                "score":      int(min(99, max(1, ai_s * 200))),
                "finish_order": None,
            })

        allowed = {"複勝", "ワイド", "3連複"}
        recommendations = [
            {"bet_type": str(r["bet_type"]), "combination": str(r["combination"]),
             "odds": float(r["odds"]), "probability": float(r["probability"]),
             "expected_value": float(r["expected_value"]), "stake": int(r["stake"])}
            for _, r in recs_df.iterrows() if str(r["bet_type"]) in allowed
        ] if not recs_df.empty else []

        return jsonify({
            "status": "ok",
            "data": {
                "race_id":     race_id,
                "race_name":   info["race_name"] or race_id,
                "race_number": race_number,
                "venue":       info["course"],
                "date":        info["date"],
                "distance":    info["distance"],
                "surface":     info["surface"],
                "field_count": n,
                "horses":      horses_out,
                "recommendations":     recommendations,
                "has_recommendations": len(recommendations) > 0,
            }
        })

    except Exception as e:
        logger.exception(f"predict-url error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    try:
        from jra_predictor.scraper.race_result import RaceResultScraper
        from jra_predictor.scraper.base import get_with_browser
        from jra_predictor.data import Database
        from config.settings import NETKEIBA_RACE, NETKEIBA_BASE, COURSE_CODES
        import pandas as pd

        scraper = RaceResultScraper()

        # 1. 出走表取得
        # fetch_race_entry: shutuba.html を試し、失敗したら fetch_race_result (result.html) に自動フォールバック
        logger.info(f"predict-url: fetching entries for {race_id}")
        df_entry = scraper.fetch_race_entry(race_id)

        if df_entry is None or df_entry.empty:
            return jsonify({
                "status": "error",
                "message": f"出走馬が取得できませんでした (race_id={race_id})。"
                           "出走表がまだ公開されていないか、URLが正しくない可能性があります。"
                           "通常はレース3〜4日前から出走表が公開されます。"
            }), 404

        entries = df_entry.to_dict("records")
        logger.info(f"predict-url: got {len(entries)} horses")

        # 2. race_info取得（日付・レース名・距離はスクレイピングで取得、race_idからは取らない）
        db = Database()
        info = scraper.fetch_race_info(race_id) or {"race_id": race_id}

        # race_idから会場コード・レース番号だけ補完（日付はスクレイピング結果を使う）
        course_code = race_id[4:6]
        if not info.get("course"):
            info["course"] = COURSE_CODES.get(course_code, course_code)
        info["course_code"] = course_code
        if not info.get("race_number"):
            info["race_number"] = int(race_id[10:12])
        # 日付が取れなかった場合のみ今日の日付で補完
        if not info.get("date"):
            from datetime import date
            info["date"] = date.today().isoformat()
        db.upsert_race_info(info)

        # race情報をentryにも埋め込む
        for entry in entries:
            entry["race_id"]     = race_id
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

        # 3. 特徴量をインラインで構築（DBフル再構築を避けて高速化）
        from jra_predictor.models import RacePredictor, ExpectedValueCalculator
        from jra_predictor.backtest.engine import BacktestEngine

        df_race = df_entry.copy()
        df_race["date"] = pd.to_datetime(df_race.get("date", ""), errors="coerce")
        df_race["horse_number"] = pd.to_numeric(df_race["horse_number"], errors="coerce").fillna(0)
        df_race["frame_number"] = pd.to_numeric(df_race.get("frame_number", 1), errors="coerce").fillna(1)
        df_race["win_odds"] = pd.to_numeric(df_race.get("win_odds", None), errors="coerce").fillna(0)
        df_race["popularity"] = pd.to_numeric(df_race.get("popularity", None), errors="coerce").fillna(0)
        df_race["weight_carried"] = pd.to_numeric(df_race.get("weight_carried", 55), errors="coerce").fillna(55)
        df_race["horse_weight"] = pd.to_numeric(df_race.get("horse_weight", None), errors="coerce").fillna(0)
        df_race["horse_weight_diff"] = pd.to_numeric(df_race.get("horse_weight_diff", None), errors="coerce").fillna(0)
        df_race["finish_order"] = pd.to_numeric(df_race.get("finish_order", None), errors="coerce")
        df_race["last_3f"] = pd.to_numeric(df_race.get("last_3f", None), errors="coerce")
        df_race["horse_number"] = df_race["horse_number"].astype(float)
        df_race["frame_number"] = df_race["frame_number"].astype(float)

        df_race["sex"] = df_race["sex_age"].str.extract(r"([牡牝騸セ])")
        df_race["age"] = pd.to_numeric(df_race["sex_age"].str.extract(r"(\d+)")[0], errors="coerce")
        df_race["field_count"] = len(df_race)
        df_race["horse_number_ratio"] = df_race["horse_number"] / df_race["field_count"]
        df_race["frame_number_norm"] = df_race["frame_number"] / 8.0
        df_race["is_win"] = 0
        df_race["is_place"] = 0

        dist_val = pd.to_numeric(df_race.get("distance", None), errors="coerce").fillna(
            info.get("distance", 1600)
        )
        df_race["distance"] = dist_val
        df_race["distance_cat"] = pd.cut(
            df_race["distance"], bins=[0, 1400, 1800, 2200, 9999],
            labels=["sprint", "mile", "middle", "long"]
        )

        wc = df_race["weight_carried"]
        df_race["weight_handicap"] = wc - wc.mean()

        pop_min = df_race["popularity"].min()
        pop_max = df_race["popularity"].max()
        df_race["popularity_norm"] = (df_race["popularity"] - pop_min) / (pop_max - pop_min + 1e-9)

        fav_rows = df_race[df_race["popularity"] == 1]
        fav_o = float(fav_rows["win_odds"].values[0]) if len(fav_rows) else float(df_race["win_odds"].min())
        df_race["fav_odds"] = fav_o
        df_race["relative_odds"] = df_race["win_odds"] / (fav_o + 1e-9)

        # 過去統計列はNaNのまま（モデルはNaN許容）
        for col in ["win_rate_3", "win_rate_5", "win_rate_10",
                    "place_rate_3", "place_rate_5", "place_rate_10",
                    "avg_popularity_3", "avg_popularity_5", "avg_popularity_10",
                    "avg_odds_3", "avg_odds_5", "avg_odds_10",
                    "prev_finish", "prev2_finish", "prev_odds", "odds_change",
                    "avg_last3f_5", "days_since_last", "career_runs",
                    "jockey_win_rate_30", "jockey_win_rate_100",
                    "jockey_place_rate_30", "jockey_place_rate_100",
                    "jockey_course_wins", "jockey_dist_wins",
                    "trainer_win_rate_50", "trainer_place_rate_50",
                    "horse_course_wins", "horse_course_place",
                    "horse_dist_wins", "horse_surface_wins", "horse_condition_wins",
                    "avg_running_style", "sire_win_rate", "sire_place_rate",
                    "sire_dist_win_rate", "avg_weight_3", "weight_vs_avg"]:
            if col not in df_race.columns:
                df_race[col] = float("nan")

        win_model = _cache.get("win_model") or RacePredictor("is_win")
        place_model = _cache.get("place_model") or RacePredictor("is_place")
        if not _cache.get("win_model"):
            win_model.load()
        if not _cache.get("place_model"):
            place_model.load()

        ev_calc = ExpectedValueCalculator(win_model, place_model)

        # object型列を強制数値変換（LightGBMはobjectを受け付けない）
        NON_NUMERIC = {"race_id", "horse_name", "horse_id", "jockey_name", "jockey_id",
                       "trainer_name", "trainer_id", "race_name", "course", "course_code",
                       "surface", "track_condition", "sex_age", "sex", "margin",
                       "passing_order", "distance_cat", "sire", "dam_sire", "birth_date",
                       "owner", "date"}
        for col in df_race.columns:
            if df_race[col].dtype == object and col not in NON_NUMERIC:
                df_race[col] = pd.to_numeric(df_race[col], errors="coerce")

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
                "number": _safe_num(row.get("horse_number", 0)),
                "frame": _safe_num(row.get("frame_number", 1), default=1),
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


def _safe_num(v, default=0, as_int=True):
    try:
        import math
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return int(f) if as_int else f
    except (TypeError, ValueError):
        return default


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
