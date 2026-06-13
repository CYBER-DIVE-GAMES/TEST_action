"""
スクレイピング基底クラス
netkeibaはトップページで Cookie を取得してから各ページにアクセスする必要がある
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
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


class BaseScaper:
    def __init__(self, delay: float = SCRAPER_DELAY):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay
        self._initialized = False

    def _init_session(self):
        """netkeibaのトップページを訪問してCookieを取得"""
        if self._initialized:
            return
        try:
            logger.info("netkeibaへの接続を初期化中...")
            # まずトップページを訪問
            r = self.session.get("https://www.netkeiba.com/", timeout=SCRAPER_TIMEOUT)
            time.sleep(2)
            # 次にレースDBトップを訪問
            self.session.headers.update({"Referer": "https://www.netkeiba.com/"})
            r2 = self.session.get("https://db.netkeiba.com/", timeout=SCRAPER_TIMEOUT)
            time.sleep(2)
            self._initialized = True
            logger.info("接続初期化完了")
        except Exception as e:
            logger.warning(f"セッション初期化エラー: {e}")

    def get(self, url: str, params: dict = None) -> BeautifulSoup | None:
        self._init_session()

        for attempt in range(SCRAPER_MAX_RETRY):
            try:
                # Refererを設定
                if "race.netkeiba" in url:
                    self.session.headers.update({"Referer": "https://race.netkeiba.com/top/"})
                elif "db.netkeiba" in url:
                    self.session.headers.update({"Referer": "https://db.netkeiba.com/"})

                resp = self.session.get(url, params=params, timeout=SCRAPER_TIMEOUT)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding

                # アクセス間隔（ランダム）
                wait = self.delay + random.uniform(0.5, 1.5)
                time.sleep(wait)

                soup = BeautifulSoup(resp.text, "lxml")

                # アクセス拒否チェック
                title = soup.title.string if soup.title else ""
                if "403" in title or "Access Denied" in title or "アクセスが拒否" in title:
                    logger.warning(f"アクセス拒否: {url}")
                    time.sleep(10)
                    continue

                return soup

            except requests.RequestException as e:
                wait = 2 ** attempt
                logger.warning(f"試行 {attempt+1}/{SCRAPER_MAX_RETRY} 失敗: {e}. {wait}秒後リトライ")
                time.sleep(wait)

        logger.error(f"全リトライ失敗: {url}")
        return None
