"""
SQLiteデータベース管理
"""
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from config.settings import DB_URL

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.engine = create_engine(DB_URL, echo=False)
        self._init_tables()

    def _init_tables(self):
        ddl = """
        CREATE TABLE IF NOT EXISTS race_info (
            race_id TEXT PRIMARY KEY,
            race_name TEXT,
            date TEXT,
            course TEXT,
            course_code TEXT,
            race_number INTEGER,
            surface TEXT,
            distance INTEGER,
            weather TEXT,
            track_condition TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS race_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL,
            date TEXT,
            course TEXT,
            course_code TEXT,
            race_number INTEGER,
            race_name TEXT,
            distance INTEGER,
            surface TEXT,
            weather TEXT,
            track_condition TEXT,
            finish_order INTEGER,
            frame_number INTEGER,
            horse_number INTEGER,
            horse_name TEXT,
            horse_id TEXT,
            sex_age TEXT,
            weight_carried REAL,
            jockey_name TEXT,
            jockey_id TEXT,
            finish_time_sec REAL,
            margin TEXT,
            passing_order TEXT,
            last_3f REAL,
            horse_weight INTEGER,
            horse_weight_diff INTEGER,
            win_odds REAL,
            popularity INTEGER,
            trainer_name TEXT,
            trainer_id TEXT,
            owner TEXT,
            prize REAL,
            UNIQUE(race_id, horse_number)
        );

        CREATE TABLE IF NOT EXISTS horse_profile (
            horse_id TEXT PRIMARY KEY,
            horse_name TEXT,
            birth_date TEXT,
            trainer TEXT,
            trainer_id TEXT,
            owner TEXT,
            breeder TEXT,
            birth_place TEXT,
            sire TEXT,
            dam_sire TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS horse_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            horse_id TEXT NOT NULL,
            race_id TEXT,
            date TEXT,
            course TEXT,
            weather TEXT,
            race_name TEXT,
            field_count INTEGER,
            frame_number INTEGER,
            horse_number INTEGER,
            odds REAL,
            popularity INTEGER,
            finish_order INTEGER,
            jockey TEXT,
            weight_carried REAL,
            distance INTEGER,
            surface TEXT,
            track_condition TEXT,
            finish_time_sec REAL,
            margin TEXT,
            passing_order TEXT,
            last_3f REAL,
            horse_weight INTEGER,
            horse_weight_diff INTEGER,
            prize REAL,
            UNIQUE(horse_id, race_id)
        );

        CREATE TABLE IF NOT EXISTS odds_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL,
            bet_type TEXT NOT NULL,
            combination TEXT NOT NULL,
            odds REAL,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(race_id, bet_type, combination)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT NOT NULL,
            horse_number INTEGER,
            win_prob REAL,
            place_prob REAL,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_results_race ON race_results(race_id);
        CREATE INDEX IF NOT EXISTS idx_results_horse ON race_results(horse_id);
        CREATE INDEX IF NOT EXISTS idx_history_horse ON horse_history(horse_id);
        CREATE INDEX IF NOT EXISTS idx_odds_race ON odds_raw(race_id);
        """
        with self.engine.connect() as conn:
            for stmt in ddl.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
            conn.commit()
        logger.info("Database initialized")

    def upsert_race_info(self, info: dict):
        df = pd.DataFrame([info])
        df.to_sql("race_info", self.engine, if_exists="append", index=False,
                  method="replace_on_conflict" if False else None)
        # SQLite upsert
        cols = list(info.keys())
        placeholders = ", ".join(f":{c}" for c in cols)
        col_str = ", ".join(cols)
        sql = f"INSERT OR REPLACE INTO race_info ({col_str}) VALUES ({placeholders})"
        with self.engine.connect() as conn:
            conn.execute(text(sql), info)
            conn.commit()

    def upsert_race_results(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        with self.engine.connect() as conn:
            for _, row in df.iterrows():
                d = row.to_dict()
                cols = [c for c in d if d[c] is not None]
                placeholders = ", ".join(f":{c}" for c in cols)
                col_str = ", ".join(cols)
                sql = f"INSERT OR REPLACE INTO race_results ({col_str}) VALUES ({placeholders})"
                conn.execute(text(sql), {c: d[c] for c in cols})
            conn.commit()

    def upsert_horse_profile(self, profile: dict):
        cols = list(profile.keys())
        placeholders = ", ".join(f":{c}" for c in cols)
        col_str = ", ".join(cols)
        sql = f"INSERT OR REPLACE INTO horse_profile ({col_str}) VALUES ({placeholders})"
        with self.engine.connect() as conn:
            conn.execute(text(sql), profile)
            conn.commit()

    def upsert_horse_history(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        with self.engine.connect() as conn:
            for _, row in df.iterrows():
                d = row.to_dict()
                cols = [c for c in d if d[c] is not None]
                placeholders = ", ".join(f":{c}" for c in cols)
                col_str = ", ".join(cols)
                sql = f"INSERT OR REPLACE INTO horse_history ({col_str}) VALUES ({placeholders})"
                conn.execute(text(sql), {c: d[c] for c in cols})
            conn.commit()

    def save_odds(self, race_id: str, bet_type: str, odds_dict: dict):
        with self.engine.connect() as conn:
            for combo, odds in odds_dict.items():
                combo_str = str(combo) if isinstance(combo, tuple) else str(combo)
                sql = """INSERT OR REPLACE INTO odds_raw
                         (race_id, bet_type, combination, odds)
                         VALUES (:race_id, :bet_type, :combo, :odds)"""
                conn.execute(text(sql), {
                    "race_id": race_id, "bet_type": bet_type,
                    "combo": combo_str, "odds": odds
                })
            conn.commit()

    def read_table(self, table: str, where: str = "") -> pd.DataFrame:
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return pd.read_sql(sql, self.engine)

    def is_race_scraped(self, race_id: str) -> bool:
        with self.engine.connect() as conn:
            r = conn.execute(
                text("SELECT 1 FROM race_info WHERE race_id = :r"),
                {"r": race_id}
            ).fetchone()
        return r is not None
