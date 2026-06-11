"""
レース結果・オッズのスクレイピング
"""
import re
import logging
import pandas as pd
from .base import BaseScaper
from config.settings import NETKEIBA_BASE, NETKEIBA_RACE

logger = logging.getLogger(__name__)


class RaceResultScraper(BaseScaper):

    def fetch_race_result(self, race_id: str) -> pd.DataFrame | None:
        """着順・タイム・馬情報を取得"""
        url = f"{NETKEIBA_BASE}/race/{race_id}/"
        soup = self.get(url)
        if soup is None:
            return None

        table = soup.select_one("table.race_table_01")
        if table is None:
            logger.warning(f"Race table not found: {race_id}")
            return None

        rows = []
        for tr in table.select("tr")[1:]:
            tds = tr.select("td")
            if len(tds) < 10:
                continue
            row = self._parse_result_row(tds, race_id)
            if row:
                rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["race_id"] = race_id
        return df

    def _parse_result_row(self, tds, race_id: str) -> dict | None:
        try:
            horse_link = tds[3].select_one("a")
            horse_id = ""
            if horse_link:
                m = re.search(r"/horse/(\w+)", horse_link.get("href", ""))
                if m:
                    horse_id = m.group(1)

            jockey_link = tds[6].select_one("a")
            jockey_id = ""
            if jockey_link:
                m = re.search(r"/jockey/(\w+)", jockey_link.get("href", ""))
                if m:
                    jockey_id = m.group(1)

            trainer_link = tds[18].select_one("a") if len(tds) > 18 else None
            trainer_id = ""
            if trainer_link:
                m = re.search(r"/trainer/(\w+)", trainer_link.get("href", ""))
                if m:
                    trainer_id = m.group(1)

            time_str = tds[7].get_text(strip=True)
            time_sec = self._parse_time(time_str)

            return {
                "finish_order": self._safe_int(tds[0].get_text(strip=True)),
                "frame_number": self._safe_int(tds[1].get_text(strip=True)),
                "horse_number": self._safe_int(tds[2].get_text(strip=True)),
                "horse_name": tds[3].get_text(strip=True),
                "horse_id": horse_id,
                "sex_age": tds[4].get_text(strip=True),
                "weight_carried": self._safe_float(tds[5].get_text(strip=True)),
                "jockey_name": tds[6].get_text(strip=True),
                "jockey_id": jockey_id,
                "finish_time_sec": time_sec,
                "margin": tds[8].get_text(strip=True),
                "passing_order": tds[10].get_text(strip=True) if len(tds) > 10 else "",
                "last_3f": self._safe_float(tds[11].get_text(strip=True)) if len(tds) > 11 else None,
                "horse_weight": self._parse_horse_weight(tds[14].get_text(strip=True)) if len(tds) > 14 else None,
                "horse_weight_diff": self._parse_horse_weight_diff(tds[14].get_text(strip=True)) if len(tds) > 14 else None,
                "win_odds": self._safe_float(tds[12].get_text(strip=True)) if len(tds) > 12 else None,
                "popularity": self._safe_int(tds[13].get_text(strip=True)) if len(tds) > 13 else None,
                "trainer_name": tds[18].get_text(strip=True) if len(tds) > 18 else "",
                "trainer_id": trainer_id,
                "owner": tds[19].get_text(strip=True) if len(tds) > 19 else "",
                "prize": self._safe_float(tds[20].get_text(strip=True).replace(",", "")) if len(tds) > 20 else None,
            }
        except Exception as e:
            logger.debug(f"Row parse error in {race_id}: {e}")
            return None

    def fetch_race_info(self, race_id: str) -> dict | None:
        """レース基本情報（距離・馬場・天候など）を取得"""
        url = f"{NETKEIBA_BASE}/race/{race_id}/"
        soup = self.get(url)
        if soup is None:
            return None

        info = {"race_id": race_id}
        try:
            title = soup.select_one("div.race_head_inner h1")
            info["race_name"] = title.get_text(strip=True) if title else ""

            data_intro = soup.select_one("div.data_intro")
            if data_intro:
                text = data_intro.get_text()
                # コース情報: 例) 芝1600m
                m = re.search(r"(芝|ダ|障)(\d+)m", text)
                if m:
                    info["surface"] = m.group(1)
                    info["distance"] = int(m.group(2))

                # 天候
                m = re.search(r"天候：(\S+)", text)
                info["weather"] = m.group(1) if m else ""

                # 馬場状態
                m = re.search(r"馬場：(\S+)", text)
                info["track_condition"] = m.group(1) if m else ""

                # 日付
                m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
                if m:
                    info["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

                # 開催場所
                course_code = race_id[4:6]
                from config.settings import COURSE_CODES
                info["course"] = COURSE_CODES.get(course_code, course_code)
                info["course_code"] = course_code

                # レース番号
                info["race_number"] = int(race_id[10:12]) if len(race_id) >= 12 else 0
        except Exception as e:
            logger.warning(f"Race info parse error {race_id}: {e}")

        return info

    def fetch_odds(self, race_id: str) -> dict:
        """単勝・複勝・馬連・ワイド・3連複オッズを取得"""
        odds = {}
        # 単勝・複勝
        odds["tansho"] = self._fetch_win_place_odds(race_id)
        # 馬連
        odds["umaren"] = self._fetch_quinella_odds(race_id)
        # ワイド
        odds["wide"] = self._fetch_wide_odds(race_id)
        # 3連複
        odds["sanrenpuku"] = self._fetch_trio_odds(race_id)
        return odds

    def _fetch_win_place_odds(self, race_id: str) -> dict:
        url = f"{NETKEIBA_RACE}/odds/index.html"
        soup = self.get(url, params={"race_id": race_id, "type": "b1"})
        if soup is None:
            return {}
        result = {}
        for tr in soup.select("tr.HorseList"):
            tds = tr.select("td")
            if len(tds) < 4:
                continue
            try:
                num = self._safe_int(tds[0].get_text(strip=True))
                win = self._safe_float(tds[1].get_text(strip=True))
                place_min = self._safe_float(tds[2].get_text(strip=True))
                place_max = self._safe_float(tds[3].get_text(strip=True))
                result[num] = {
                    "win_odds": win,
                    "place_odds_min": place_min,
                    "place_odds_max": place_max,
                }
            except Exception:
                pass
        return result

    def _fetch_quinella_odds(self, race_id: str) -> dict:
        url = f"{NETKEIBA_RACE}/odds/index.html"
        soup = self.get(url, params={"race_id": race_id, "type": "b4"})
        if soup is None:
            return {}
        return self._parse_pair_odds(soup)

    def _fetch_wide_odds(self, race_id: str) -> dict:
        url = f"{NETKEIBA_RACE}/odds/index.html"
        soup = self.get(url, params={"race_id": race_id, "type": "b5"})
        if soup is None:
            return {}
        return self._parse_pair_odds(soup)

    def _fetch_trio_odds(self, race_id: str) -> dict:
        url = f"{NETKEIBA_RACE}/odds/index.html"
        soup = self.get(url, params={"race_id": race_id, "type": "b7"})
        if soup is None:
            return {}
        result = {}
        for td in soup.select("td.Odds"):
            try:
                combo_td = td.find_previous("td", class_="Num")
                if combo_td:
                    combo = tuple(int(x) for x in combo_td.get_text(strip=True).split("-"))
                    odds_val = self._safe_float(td.get_text(strip=True))
                    result[combo] = odds_val
            except Exception:
                pass
        return result

    def _parse_pair_odds(self, soup) -> dict:
        result = {}
        for tr in soup.select("tr"):
            tds = tr.select("td")
            if len(tds) < 2:
                continue
            try:
                combo_text = tds[0].get_text(strip=True)
                m = re.match(r"(\d+)-(\d+)", combo_text)
                if m:
                    combo = (int(m.group(1)), int(m.group(2)))
                    result[combo] = self._safe_float(tds[1].get_text(strip=True))
            except Exception:
                pass
        return result

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
