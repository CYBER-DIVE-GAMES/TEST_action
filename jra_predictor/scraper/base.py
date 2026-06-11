"""
スクレイピング基底クラス
"""
import time
import random
import logging
import requests
from bs4 import BeautifulSoup
from config.settings import SCRAPER_DELAY, SCRAPER_TIMEOUT, SCRAPER_MAX_RETRY

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}


class BaseScaper:
    def __init__(self, delay: float = SCRAPER_DELAY):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay

    def get(self, url: str, params: dict = None) -> BeautifulSoup | None:
        for attempt in range(SCRAPER_MAX_RETRY):
            try:
                resp = self.session.get(url, params=params, timeout=SCRAPER_TIMEOUT)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
                time.sleep(self.delay + random.uniform(0, 1))
                return BeautifulSoup(resp.text, "lxml")
            except requests.RequestException as e:
                wait = 2 ** attempt
                logger.warning(f"Attempt {attempt+1}/{SCRAPER_MAX_RETRY} failed: {e}. Retrying in {wait}s")
                time.sleep(wait)
        logger.error(f"All retries failed for {url}")
        return None

    def get_text(self, url: str, params: dict = None) -> str | None:
        for attempt in range(SCRAPER_MAX_RETRY):
            try:
                resp = self.session.get(url, params=params, timeout=SCRAPER_TIMEOUT)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
                time.sleep(self.delay + random.uniform(0, 1))
                return resp.text
            except requests.RequestException as e:
                wait = 2 ** attempt
                logger.warning(f"Attempt {attempt+1}/{SCRAPER_MAX_RETRY} failed: {e}. Retrying in {wait}s")
                time.sleep(wait)
        return None
