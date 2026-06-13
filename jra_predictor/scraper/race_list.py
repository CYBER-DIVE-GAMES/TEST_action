"""
レースID一覧の取得

netkeibaのレースIDは YYYYMMDDCCRR の12桁。
開催カレンダーから日付を取得し、各日のレース一覧からIDを収集する。
"""
import re
import calendar
import logging
from typing import Generator
from .base import BaseScaper
from config.settings import COURSE_CODES

logger = logging.getLogger(__name__)


class RaceListScraper(BaseScaper):

    def iter_race_ids(
        self,
        start_year: int,
        end_year: int,
        course_codes: list[str] = None,
    ) -> Generator[str, None, None]:
        if course_codes is None:
            course_codes = list(COURSE_CODES.keys())

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                dates = self._get_kaisai_dates(year, month)
                for date_str in dates:
                    ids = self._fetch_date_race_ids(date_str, course_codes)
                    for race_id in ids:
                        yield race_id
            logger.info(f"{year}: collection complete")

    def _get_kaisai_dates(self, year: int, month: int) -> list[str]:
        """
        土日をベースに開催候補日を生成する。
        netkeibaのカレンダーから取得を試み、失敗時は土日を返す。
        """
        dates = []

        # netkeibaの月別開催一覧ページを試す
        url = f"https://race.netkeiba.com/top/race_list.html"
        # 月初の土曜日または日曜日を指定してページを取得
        for day in range(1, 8):
            d = f"{year}{month:02d}{day:02d}"
            soup = self.get(url, params={"kaisai_date": d})
            if soup:
                found = self._parse_dates_from_page(soup, year, month)
                if found:
                    dates.extend(found)
                    break

        if not dates:
            # フォールバック: その月の土日を全部試す
            cal = calendar.monthcalendar(year, month)
            for week in cal:
                for day_idx in [5, 6]:  # 土=5, 日=6
                    day = week[day_idx]
                    if day != 0:
                        dates.append(f"{year}{month:02d}{day:02d}")

        return sorted(set(dates))

    def _parse_dates_from_page(self, soup, year: int, month: int) -> list[str]:
        """ページ内から YYYYMMDD 形式の開催日を抽出"""
        dates = []
        prefix = f"{year}{month:02d}"

        # kaisai_date=YYYYMMDD パターン
        for a in soup.find_all("a", href=re.compile(r"kaisai_date=\d{8}")):
            m = re.search(r"kaisai_date=(\d{8})", a["href"])
            if m and m.group(1).startswith(prefix):
                dates.append(m.group(1))

        # data-date属性
        for el in soup.find_all(attrs={"data-date": True}):
            d = str(el.get("data-date", ""))
            if re.match(r"\d{8}", d) and d.startswith(prefix):
                dates.append(d)

        return list(set(dates))

    def _fetch_date_race_ids(self, date_str: str, course_codes: list[str]) -> list[str]:
        """1日分の全レースIDを取得"""
        race_ids = []

        # 方法1: race_list.html から取得
        url = "https://race.netkeiba.com/top/race_list.html"
        soup = self.get(url, params={"kaisai_date": date_str})
        if soup:
            race_ids = self._parse_race_ids_from_page(soup, course_codes)

        # 方法2: db.netkeiba の日別一覧
        if not race_ids:
            url2 = f"https://db.netkeiba.com/race/list/{date_str}/"
            soup2 = self.get(url2)
            if soup2:
                race_ids = self._parse_race_ids_from_page(soup2, course_codes)

        # 方法3: 直接生成（最終手段）
        if not race_ids:
            race_ids = self._probe_race_ids(date_str, course_codes)

        if race_ids:
            logger.debug(f"  {date_str}: {len(race_ids)} races")
        return race_ids

    def _parse_race_ids_from_page(self, soup, course_codes: list[str]) -> list[str]:
        """ページ内の全レースIDリンクを抽出"""
        race_ids = []
        patterns = [
            re.compile(r"race_id=(\d{12})"),
            re.compile(r"/race/(\d{12})"),
            re.compile(r"race/result\.html\?race_id=(\d{12})"),
        ]
        for a in soup.find_all("a", href=True):
            href = a["href"]
            for pat in patterns:
                m = pat.search(href)
                if m:
                    race_id = m.group(1)
                    course_code = race_id[4:6]
                    if course_code in course_codes and race_id not in race_ids:
                        race_ids.append(race_id)
                    break
        return race_ids

    def _probe_race_ids(self, date_str: str, course_codes: list[str]) -> list[str]:
        """
        IDを直接生成してアクセス確認（最終手段）
        土日の2場開催を前提に、各競馬場の1〜12Rを確認
        """
        found = []
        for course in course_codes:
            course_found = False
            for r_num in range(1, 13):
                race_id = f"{date_str}{course}{r_num:02d}"
                url = f"https://db.netkeiba.com/race/{race_id}/"
                soup = self.get(url)
                if soup and soup.select_one("table.race_table_01, div.race_head_inner"):
                    found.append(race_id)
                    course_found = True
                elif course_found and r_num >= 3:
                    # この競馬場での開催が終わったと判断
                    break
        return found
