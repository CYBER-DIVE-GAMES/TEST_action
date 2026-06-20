"""
グローバル設定
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models_saved"
LOG_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

# スクレイピング設定
SCRAPER_DELAY = 2.0          # リクエスト間隔（秒）
SCRAPER_TIMEOUT = 30
SCRAPER_MAX_RETRY = 3
NETKEIBA_BASE = "https://db.netkeiba.com"
NETKEIBA_RACE = "https://race.netkeiba.com"

# DB設定
DB_PATH = DATA_DIR / "jra.db"
DB_URL = f"sqlite:///{DB_PATH}"

# モデル設定
RANDOM_SEED = 42
CV_FOLDS = 5
TEST_YEARS = 2               # バックテスト用直近N年

# 馬券種別設定
BET_TYPES = ["fukusho", "wide", "umaren", "sanrenpuku"]

# 期待値閾値（この値以上の馬券のみ購入推奨）
EV_THRESHOLD = {
    "tan": 1.15,
    "fukusho": 1.10,
    "wide": 1.20,
    "umaren": 1.25,
    "sanrenpuku": 1.30,
}

# ケリー基準の分数（過度な賭けを防ぐ）
KELLY_FRACTION = 0.25

# 対象競馬場コード
COURSE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}
