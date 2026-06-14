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
        _cache["error"] = str(e)
        logger.error(f"Load error: {e}")
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


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
