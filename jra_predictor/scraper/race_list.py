"""
レースID一覧の取得
netkeibaのレース検索ページからIDを収集する
"""
import re
import logging
from typing import Generator
from .base import BaseScaper
from config.settings import NETKEIBA_BASE, COURSE_CODES

logger = logging.getLogger(__name__)


class RaceListScraper(BaseScaper):
    """年・競馬場・開催回からレースIDを列挙する"""

    def iter_race_ids(
        self,
        start_year: int,
        end_year: int,
        course_codes: list[str] = None,
    ) -> Generator[str, None, None]:
        if course_codes is None:
            course_codes = list(COURSE_CODES.keys())

        for year in range(start_year, end_year + 1):
            for course in course_codes:
                yield from self._fetch_course_race_ids(year, course)

    def _fetch_course_race_ids(self, year: int, course: str) -> list[str]:
        url = f"{NETKEIBA_BASE}/race/list/{year}{course}/"
        # 全開催ページをページネーションで巡回
        race_ids = []
        page = 1
        while True:
            soup = self.get(url, params={"page": page})
            if soup is None:
                break
            ids = self._parse_race_ids(soup)
            if not ids:
                break
            race_ids.extend(ids)
            # 次ページリンクがなければ終了
            next_link = soup.select_one("a.page_next")
            if not next_link:
                break
            page += 1

        logger.info(f"{year} course={course}: {len(race_ids)} races found")
        return race_ids

    def _parse_race_ids(self, soup) -> list[str]:
        race_ids = []
        for a in soup.select("dl.race_top_hold_info dt a, ul.race_top_hold_info li a"):
            href = a.get("href", "")
            m = re.search(r"/race/(\d{12})", href)
            if m:
                race_ids.append(m.group(1))
        # より一般的なパターンも検索
        for a in soup.find_all("a", href=re.compile(r"/race/\d{12}")):
            m = re.search(r"/race/(\d{12})", a["href"])
            if m and m.group(1) not in race_ids:
                race_ids.append(m.group(1))
        return list(set(race_ids))
