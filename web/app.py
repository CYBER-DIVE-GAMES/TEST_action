"""
JRA予想ツール Webサーバー
"""
import sys
import json
import random
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
CORS(app)


def mock_race_data(race_id: str) -> dict:
    """
    DB・モデルが未構築の場合でも画面確認できるモックデータ
    実運用時は real_race_data() に差し替える
    """
    horses = [
        {"number": 1, "frame": 1, "name": "ステラリア",    "sex_age": "牝5", "jockey": "川田将雅", "trainer": "高橋康之", "weight": 456, "weight_diff": -2,  "win_odds": 2.1,  "place_odds": 1.3, "popularity": 1, "win_prob": 0.312, "place_prob": 0.641, "score": 93},
        {"number": 2, "frame": 1, "name": "グランアレグリア","sex_age": "牝6", "jockey": "福永祐一", "trainer": "藤沢和雄", "weight": 464, "weight_diff": +4, "win_odds": 3.4,  "place_odds": 1.6, "popularity": 2, "win_prob": 0.198, "place_prob": 0.512, "score": 87},
        {"number": 3, "frame": 2, "name": "ソングライン",   "sex_age": "牝4", "jockey": "池添謙一", "trainer": "林徹",    "weight": 448, "weight_diff": 0,   "win_odds": 5.8,  "place_odds": 2.1, "popularity": 3, "win_prob": 0.143, "place_prob": 0.389, "score": 81},
        {"number": 4, "frame": 2, "name": "シュネルマイスター","sex_age": "牡4","jockey": "C.ルメール","trainer": "手塚貴久","weight": 490, "weight_diff": -6, "win_odds": 7.2,  "place_odds": 2.4, "popularity": 4, "win_prob": 0.112, "place_prob": 0.334, "score": 78},
        {"number": 5, "frame": 3, "name": "サリオス",       "sex_age": "牡5", "jockey": "松山弘平", "trainer": "堀宣行",  "weight": 498, "weight_diff": +2,  "win_odds": 9.1,  "place_odds": 2.8, "popularity": 5, "win_prob": 0.089, "place_prob": 0.298, "score": 74},
        {"number": 6, "frame": 3, "name": "ダノンザキッド", "sex_age": "牡4", "jockey": "戸崎圭太", "trainer": "安田翔伍","weight": 484, "weight_diff": +8,  "win_odds": 12.4, "place_odds": 3.5, "popularity": 6, "win_prob": 0.062, "place_prob": 0.221, "score": 68},
        {"number": 7, "frame": 4, "name": "ホウオウアマゾン","sex_age": "牡4","jockey": "岩田康誠", "trainer": "西村真幸","weight": 476, "weight_diff": 0,   "win_odds": 18.7, "place_odds": 4.2, "popularity": 7, "win_prob": 0.041, "place_prob": 0.178, "score": 62},
        {"number": 8, "frame": 4, "name": "カテドラル",     "sex_age": "牡6", "jockey": "横山武史", "trainer": "小島茂之","weight": 488, "weight_diff": -4,  "win_odds": 24.5, "place_odds": 5.1, "popularity": 8, "win_prob": 0.031, "place_prob": 0.142, "score": 58},
        {"number": 9, "frame": 5, "name": "インディチャンプ","sex_age": "牡6","jockey": "福永祐一", "trainer": "音無秀孝","weight": 502, "weight_diff": +6,  "win_odds": 31.2, "place_odds": 6.3, "popularity": 9, "win_prob": 0.024, "place_prob": 0.118, "score": 54},
        {"number": 10,"frame": 5, "name": "ロータスランド",  "sex_age": "牝5","jockey": "和田竜二", "trainer": "吉田直弘","weight": 444, "weight_diff": -2,  "win_odds": 42.1, "place_odds": 7.8, "popularity":10, "win_prob": 0.018, "place_prob": 0.098, "score": 49},
        {"number": 11,"frame": 6, "name": "ケイデンスコール","sex_age": "牡5","jockey": "藤岡佑介", "trainer": "奥村武",  "weight": 470, "weight_diff": 0,   "win_odds": 56.8, "place_odds": 9.2, "popularity":11, "win_prob": 0.012, "place_prob": 0.078, "score": 44},
        {"number": 12,"frame": 6, "name": "ファインルージュ","sex_age": "牝4","jockey": "三浦皇成", "trainer": "黒岩陽一","weight": 452, "weight_diff": +4,  "win_odds": 68.3, "place_odds":11.4, "popularity":12, "win_prob": 0.009, "place_prob": 0.063, "score": 41},
        {"number": 13,"frame": 7, "name": "レシステンシア", "sex_age": "牝5","jockey": "北村友一", "trainer": "松下武士","weight": 438, "weight_diff": -8,  "win_odds": 82.1, "place_odds":13.5, "popularity":13, "win_prob": 0.007, "place_prob": 0.051, "score": 38},
        {"number": 14,"frame": 7, "name": "カラテ",         "sex_age": "牡5", "jockey": "菅原明良", "trainer": "中井裕二","weight": 480, "weight_diff": +2,  "win_odds":103.4, "place_odds":16.2, "popularity":14, "win_prob": 0.005, "place_prob": 0.042, "score": 35},
        {"number": 15,"frame": 8, "name": "ビアンフェ",     "sex_age": "牡5", "jockey": "浜中俊",  "trainer": "寺島良",  "weight": 474, "weight_diff": 0,   "win_odds":145.6, "place_odds":21.3, "popularity":15, "win_prob": 0.003, "place_prob": 0.031, "score": 32},
        {"number": 16,"frame": 8, "name": "アンドラステ",   "sex_age": "牝4","jockey": "岩田望来", "trainer": "西村真幸","weight": 446, "weight_diff": -4,  "win_odds":188.2, "place_odds":26.8, "popularity":16, "win_prob": 0.002, "place_prob": 0.024, "score": 28},
    ]

    # 期待値の高い推奨馬券
    recommendations = [
        {"bet_type": "複勝",  "combination": "1",    "horse_names": "ステラリア",              "odds": 1.3,  "probability": 0.641, "expected_value": 1.53, "stake": 2000},
        {"bet_type": "複勝",  "combination": "2",    "horse_names": "グランアレグリア",         "odds": 1.6,  "probability": 0.512, "expected_value": 1.42, "stake": 1500},
        {"bet_type": "ワイド","combination": "1-2",  "horse_names": "ステラリア - グランアレグリア","odds": 2.4, "probability": 0.298, "expected_value": 1.38, "stake": 1200},
        {"bet_type": "ワイド","combination": "1-3",  "horse_names": "ステラリア - ソングライン", "odds": 3.8,  "probability": 0.198, "expected_value": 1.31, "stake": 900},
        {"bet_type": "馬連",  "combination": "1-2",  "horse_names": "ステラリア - グランアレグリア","odds": 5.2, "probability": 0.198, "expected_value": 1.28, "stake": 800},
        {"bet_type": "馬連",  "combination": "1-3",  "horse_names": "ステラリア - ソングライン", "odds": 9.1,  "probability": 0.112, "expected_value": 1.24, "stake": 600},
        {"bet_type": "3連複", "combination": "1-2-3","horse_names": "ステラリア - グランアレグリア - ソングライン","odds": 14.8,"probability": 0.089,"expected_value": 1.32,"stake": 500},
    ]

    # コース適性レーダーチャート用
    top_horse = horses[0]
    radar = {
        "labels": ["芝適性", "距離適性", "コース実績", "騎手相性", "馬場適性", "近走状態"],
        "values": [88, 92, 95, 85, 78, 93],
    }

    return {
        "race_id": race_id,
        "race_name": "ヴィクトリアマイルG1",
        "venue": "東京11R",
        "date": "2024年5月12日（日）",
        "conditions": "芝 / 1600m / 良 / 晴",
        "field_count": len(horses),
        "horses": horses,
        "recommendations": recommendations,
        "radar": radar,
        "top_horse": top_horse,
        "stats": {
            "total_stake": sum(r["stake"] for r in recommendations),
            "expected_return": int(sum(r["stake"] for r in recommendations) * 1.24),
            "hit_rate": 24.3,
            "roi": 112.6,
        }
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/race/<race_id>")
def api_race(race_id):
    try:
        # 実運用時はここでDBから取得・モデル予測
        # from jra_predictor.data import Database
        # from jra_predictor.features import FeatureBuilder
        # from jra_predictor.models import RacePredictor, ExpectedValueCalculator
        data = mock_race_data(race_id)
        return jsonify({"status": "ok", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json()
    race_id = body.get("race_id", "202405050811")
    data = mock_race_data(race_id)
    return jsonify({"status": "ok", "data": data})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
