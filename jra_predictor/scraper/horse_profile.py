"""
馬プロフィール・過去成績のスクレイピング
"""
import re
import logging
import pandas as pd
from .base import BaseScaper
from config.settings import NETKEIBA_BASE

logger = logging.getLogger(__name__)


class HorseProfileScraper(BaseScaper):

    def fetch_horse_history(self, horse_id: str) -> pd.DataFrame | None:
        """馬の全レース履歴を取得"""
        url = f"{NETKEIBA_BASE}/horse/{horse_id}/"
        soup = self.get(url)
        if soup is None:
            return None

        table = soup.select_one("table.db_h_race_results")
        if table is None:
            return None

        rows = []
        for tr in table.select("tr"):
            tds = tr.select("td")
            if len(tds) < 15:
                continue
            # ヘッダー行をスキップ（最初のtdが数字の日付でなければスキップ）
            first = tds[0].get_text(strip=True)
            if not re.match(r"\d{4}/\d{2}/\d{2}", first):
                continue
            row = self._parse_history_row(tds, horse_id)
            if row:
                rows.append(row)

        return pd.DataFrame(rows) if rows else None

    def _parse_history_row(self, tds, horse_id: str) -> dict | None:
        try:
            race_link = tds[4].select_one("a")
            race_id = ""
            if race_link:
                m = re.search(r"/race/(\d{12})", race_link.get("href", ""))
                if m:
                    race_id = m.group(1)

            return {
                "horse_id": horse_id,
                "race_id": race_id,
                "date": tds[0].get_text(strip=True),
                "course": tds[1].get_text(strip=True),
                "weather": tds[2].get_text(strip=True),
                "race_number": tds[3].get_text(strip=True),
                "race_name": tds[4].get_text(strip=True),
                "field_count": self._safe_int(tds[5].get_text(strip=True)),
                "frame_number": self._safe_int(tds[6].get_text(strip=True)),
                "horse_number": self._safe_int(tds[7].get_text(strip=True)),
                "odds": self._safe_float(tds[8].get_text(strip=True)),
                "popularity": self._safe_int(tds[9].get_text(strip=True)),
                "finish_order": self._safe_int(tds[10].get_text(strip=True)),
                "jockey": tds[11].get_text(strip=True),
                "weight_carried": self._safe_float(tds[12].get_text(strip=True)),
                "distance": self._parse_distance(tds[13].get_text(strip=True)),
                "surface": self._parse_surface(tds[13].get_text(strip=True)),
                "track_condition": tds[14].get_text(strip=True),
                "finish_time_sec": self._parse_time(tds[17].get_text(strip=True)) if len(tds) > 17 else None,
                "margin": tds[18].get_text(strip=True) if len(tds) > 18 else "",
                "passing_order": tds[20].get_text(strip=True) if len(tds) > 20 else "",
                "last_3f": self._safe_float(tds[22].get_text(strip=True)) if len(tds) > 22 else None,
                "horse_weight": self._parse_horse_weight(tds[23].get_text(strip=True)) if len(tds) > 23 else None,
                "horse_weight_diff": self._parse_horse_weight_diff(tds[23].get_text(strip=True)) if len(tds) > 23 else None,
                "prize": self._safe_float(tds[25].get_text(strip=True).replace(",", "")) if len(tds) > 25 else None,
            }
        except Exception as e:
            logger.debug(f"History row parse error {horse_id}: {e}")
            return None

    def fetch_horse_profile(self, horse_id: str) -> dict | None:
        """馬の基本情報（血統・馬主など）を取得"""
        url = f"{NETKEIBA_BASE}/horse/{horse_id}/"
        soup = self.get(url)
        if soup is None:
            return None

        info = {"horse_id": horse_id}
        try:
            name_tag = soup.select_one("div.horse_title h1")
            info["horse_name"] = name_tag.get_text(strip=True) if name_tag else ""

            for tr in soup.select("table.db_prof_table tr"):
                th = tr.select_one("th")
                td = tr.select_one("td")
                if not th or not td:
                    continue
                key = th.get_text(strip=True)
                val = td.get_text(strip=True)
                if key == "生年月日":
                    info["birth_date"] = val
                elif key == "調教師":
                    info["trainer"] = val
                    link = td.select_one("a")
                    if link:
                        m = re.search(r"/trainer/(\w+)", link.get("href", ""))
                        if m:
                            info["trainer_id"] = m.group(1)
                elif key == "馬主":
                    info["owner"] = val
                elif key == "生産者":
                    info["breeder"] = val
                elif key == "産地":
                    info["birth_place"] = val
                elif key == "セリ取引価格":
                    info["auction_price"] = val

            # 父・母父（血統）
            pedigree = soup.select("table.blood_table td a")
            if len(pedigree) >= 2:
                info["sire"] = pedigree[0].get_text(strip=True)
                info["dam_sire"] = pedigree[2].get_text(strip=True) if len(pedigree) > 2 else ""
        except Exception as e:
            logger.warning(f"Profile parse error {horse_id}: {e}")

        return info

    @staticmethod
    def _parse_distance(s: str) -> int | None:
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None

    @staticmethod
    def _parse_surface(s: str) -> str:
        if "芝" in s:
            return "芝"
        elif "ダ" in s:
            return "ダート"
        elif "障" in s:
            return "障害"
        return ""

    @staticmethod
    def _parse_time(s: str) -> float | None:
        m = re.match(r"(\d+):(\d+\.\d+)", s)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_horse_weight(s: str) -> int | None:
        m = re.match(r"(\d+)", s)
        return int(m.group(1)) if m else None

    @staticmethod
    def _parse_horse_weight_diff(s: str) -> int | None:
        m = re.search(r"\(([+-]?\d+)\)", s)
        return int(m.group(1)) if m else None

    @staticmethod
    def _safe_int(s: str) -> int | None:
        try:
            return int(s.replace(",", ""))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(s: str) -> float | None:
        try:
            return float(s.replace(",", ""))
        except (ValueError, TypeError):
            return None
