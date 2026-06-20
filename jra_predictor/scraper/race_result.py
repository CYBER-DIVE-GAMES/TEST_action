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

    def fetch_race_entry(self, race_id: str) -> pd.DataFrame | None:
        """出走表から出走馬一覧を取得

        shutuba_past.html → shutuba.html → result.html の順で試す。
        netkeibaはJSレンダリングのためPlaywrightヘッドレスブラウザを使用する。
        """
        rows_html = None
        for page_type in ["shutuba_past.html", "shutuba.html"]:
            url = f"{NETKEIBA_RACE}/race/{page_type}?race_id={race_id}"
            soup = self.get_browser(url, wait_selector="tr.HorseList")
            if soup is None:
                continue
            rows_html = (
                soup.select("tr.HorseList")
                or soup.select("tr[class*='HorseList']")
                or [tr for tr in soup.select("table.Shutuba_Table tr")
                    if len(tr.select("td")) >= 6 and tr.select_one("a[href*='/horse/']")]
            )
            if rows_html:
                logger.info(f"Got {len(rows_html)} rows from {page_type} for {race_id}")
                break

        if not rows_html:
            logger.info(f"出走表なし→結果ページにフォールバック: {race_id}")
            return self.fetch_race_result(race_id)

        rows = []
        for tr in rows_html:
            tds = tr.select("td")
            if len(tds) < 6:
                continue
            try:
                horse_link = tr.select_one("a[href*='/horse/']")
                horse_id, horse_name = "", ""
                if horse_link:
                    m = re.search(r"/horse/(\w+)", horse_link.get("href", ""))
                    if m:
                        horse_id = m.group(1)
                    horse_name = horse_link.get_text(strip=True)

                jockey_link = tr.select_one("a[href*='/jockey/']")
                jockey_id, jockey_name = "", ""
                if jockey_link:
                    m = re.search(r"/jockey/(\w+)", jockey_link.get("href", ""))
                    if m:
                        jockey_id = m.group(1)
                    jockey_name = jockey_link.get_text(strip=True)

                trainer_link = tr.select_one("a[href*='/trainer/']")
                trainer_id, trainer_name = "", ""
                if trainer_link:
                    m = re.search(r"/trainer/(\w+)", trainer_link.get("href", ""))
                    if m:
                        trainer_id = m.group(1)
                    trainer_name = trainer_link.get_text(strip=True)

                texts = [td.get_text(strip=True) for td in tds]
                rows.append({
                    "race_id": race_id,
                    "frame_number": self._safe_int(texts[0]) if texts else None,
                    "horse_number": self._safe_int(texts[1]) if len(texts) > 1 else None,
                    "horse_name": horse_name or (texts[3] if len(texts) > 3 else ""),
                    "horse_id": horse_id,
                    "sex_age": texts[4] if len(texts) > 4 else "",
                    "weight_carried": self._safe_float(texts[5]) if len(texts) > 5 else None,
                    "jockey_name": jockey_name,
                    "jockey_id": jockey_id,
                    "trainer_name": trainer_name,
                    "trainer_id": trainer_id,
                    "win_odds": self._safe_float(texts[-3]) if len(texts) > 3 else None,
                    "popularity": self._safe_int(texts[-2]) if len(texts) > 2 else None,
                    "horse_weight": None,
                    "horse_weight_diff": None,
                    "finish_order": None,
                    "finish_time_sec": None,
                    "margin": "",
                    "passing_order": "",
                    "last_3f": None,
                    "is_win": 0,
                    "is_place": 0,
                })
            except Exception as e:
                logger.debug(f"Entry row parse error: {e}")
                continue

        if not rows:
            logger.warning(f"Entry rows empty after Playwright: {race_id}")
            return None

        logger.info(f"fetch_race_entry: {len(rows)} horses for {race_id}")
        return pd.DataFrame(rows)

    def _fetch_race_page(self, race_id: str):
        """レースページのsoupを取得（requests→Playwright順に試す）"""
        url = f"{NETKEIBA_BASE}/race/{race_id}/"
        soup = self.get(url)
        if soup is None or not soup.select("a[href*='/horse/']"):
            logger.info(f"Result page: falling back to Playwright for {race_id}")
            soup = self.get_browser(url, wait_selector="table.race_table_01")
        return soup

    def _parse_race_info_from_soup(self, soup, race_id: str) -> dict:
        """soupからレース基本情報を抽出"""
        from config.settings import COURSE_CODES
        info = {"race_id": race_id}
        try:
            # レース名
            for sel in ["div.race_head_inner h1", "h1.RaceName", "h1"]:
                el = soup.select_one(sel)
                if el and el.get_text(strip=True):
                    info["race_name"] = el.get_text(strip=True)
                    break

            # 距離・馬場・天候・馬場状態・日付
            text = ""
            for sel in ["div.data_intro", "div.RaceData01", "div.race_data", "div.mainrace_data"]:
                el = soup.select_one(sel)
                if el:
                    text = el.get_text()
                    break
            # どのdivも取れない場合はページ全体のテキストで試す
            if not text:
                text = soup.get_text()

            m = re.search(r"(芝|ダ|障)[^\d]*(\d{3,4})m", text)
            if m:
                info["surface"] = m.group(1)
                info["distance"] = int(m.group(2))

            m = re.search(r"天候\s*[:：]\s*(\S+)", text)
            info["weather"] = m.group(1) if m else ""

            m = re.search(r"馬場\s*[:：]\s*(\S+)", text)
            info["track_condition"] = m.group(1) if m else ""

            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
            if m:
                info["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

            course_code = race_id[8:10]
            info["course"] = COURSE_CODES.get(course_code, course_code)
            info["course_code"] = course_code
            info["race_number"] = int(race_id[10:12]) if len(race_id) >= 12 else 0
        except Exception as e:
            logger.warning(f"Race info parse error {race_id}: {e}")
        return info

    def fetch_race_result(self, race_id: str) -> pd.DataFrame | None:
        """着順・タイム・馬情報を取得（race_infoも同時に更新）"""
        soup = self._fetch_race_page(race_id)
        if soup is None:
            return None

        table = (
            soup.select_one("table.race_table_01")
            or soup.select_one("table.result_table_02")
            or next((t for t in soup.find_all("table")
                     if t.select("a[href*='/horse/']")), None)
        )
        if table is None:
            logger.warning(f"Race table not found: {race_id}")
            return None

        rows = []
        for tr in table.select("tr")[1:]:
            tds = tr.select("td")
            if len(tds) < 6:
                continue
            if not tr.select_one("a[href*='/horse/']"):
                continue
            row = self._parse_result_row(tds, race_id)
            if row:
                rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["race_id"] = race_id

        # 同じsoupからrace_info情報もDFに付与
        info = self._parse_race_info_from_soup(soup, race_id)
        for col in ["date", "course", "course_code", "race_number", "race_name",
                    "distance", "surface", "weather", "track_condition"]:
            if info.get(col):
                df[col] = info[col]

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
        soup = self._fetch_race_page(race_id)
        if soup is None:
            return None
        return self._parse_race_info_from_soup(soup, race_id)

    def fetch_odds(self, race_id: str) -> dict:
        """単勝・複勝・馬連・ワイド・3連複オッズを取得"""
        odds = {}
        odds["tansho"] = self._fetch_win_place_odds(race_id)
        odds["umaren"] = self._fetch_quinella_odds(race_id)
        odds["wide"] = self._fetch_wide_odds(race_id)
        odds["sanrenpuku"] = self._fetch_trio_odds(race_id)
        return odds

    def _fetch_win_place_odds(self, race_id: str) -> dict:
        url = f"{NETKEIBA_RACE}/odds/index.html?race_id={race_id}&type=b1"
        soup = self.get_browser(url, wait_selector="table.RaceOdds_HorseList_Table")
        if soup is None:
            soup = self.get(f"{NETKEIBA_RACE}/odds/index.html", params={"race_id": race_id, "type": "b1"})
        if soup is None:
            return {}

        tables = soup.select("table.RaceOdds_HorseList_Table")
        # tables[0]=単勝, tables[1]=複勝
        result = {}

        def parse_table(table, idx):
            for tr in table.select("tr")[1:]:  # skip header
                tds = tr.select("td")
                if len(tds) < 6:
                    continue
                num = self._safe_int(tds[1].get_text(strip=True))
                odds_text = tds[5].get_text(strip=True)
                if num is None:
                    continue
                if num not in result:
                    result[num] = {}
                if idx == 0:
                    result[num]["win_odds"] = self._safe_float(odds_text)
                else:
                    # "1.7 - 2.6" 形式
                    parts = odds_text.replace("–", "-").split("-")
                    result[num]["place_odds_min"] = self._safe_float(parts[0].strip()) if parts else None
                    result[num]["place_odds_max"] = self._safe_float(parts[-1].strip()) if len(parts) > 1 else result[num]["place_odds_min"]

        for i, t in enumerate(tables[:2]):
            parse_table(t, i)

        logger.info(f"Win/place odds fetched: {len(result)} horses for {race_id}")
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
