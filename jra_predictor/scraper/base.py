"""
スクレイピング基底クラス
netkeibaはJSレンダリングのため、Playwrightヘッドレスブラウザを使用する
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

_playwright_instance = None
_pw_browser = None


def _get_browser():
    """Playwrightブラウザのシングルトン取得"""
    global _playwright_instance, _pw_browser
    if _pw_browser is None:
        try:
            from playwright.sync_api import sync_playwright
            _playwright_instance = sync_playwright().start()
            _pw_browser = _playwright_instance.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            logger.info("Playwrightブラウザ起動完了")
        except Exception as e:
            logger.error(f"Playwright起動失敗: {e}")
            _pw_browser = None
    return _pw_browser


def get_with_browser(url: str, wait_selector: str = None, timeout_ms: int = 15000) -> BeautifulSoup | None:
    """Playwrightでページを取得してBeautifulSoupに変換"""
    browser = _get_browser()
    if browser is None:
        return None
    page = None
    try:
        page = browser.new_page()
        page.set_extra_http_headers({"Accept-Language": "ja,en-US;q=0.9"})
        page.goto(url, timeout=timeout_ms, wait_until="networkidle")
        if wait_selector:
            try:
                page.wait_for_selector(wait_selector, timeout=12000)
            except Exception:
                pass
        page.wait_for_timeout(2000)
        html = page.content()
        return BeautifulSoup(html, "lxml")
    except Exception as e:
        logger.warning(f"Playwright取得失敗 {url}: {e}")
        return None
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass


class BaseScaper:
    def __init__(self, delay: float = SCRAPER_DELAY):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay
        self._initialized = False

    def _init_session(self):
        if self._initialized:
            return
        try:
            logger.info("netkeibaへの接続を初期化中...")
            self.session.get("https://www.netkeiba.com/", timeout=SCRAPER_TIMEOUT)
            time.sleep(2)
            self.session.headers.update({"Referer": "https://www.netkeiba.com/"})
            self.session.get("https://db.netkeiba.com/", timeout=SCRAPER_TIMEOUT)
            time.sleep(2)
            self._initialized = True
            logger.info("接続初期化完了")
        except Exception as e:
            logger.warning(f"セッション初期化エラー: {e}")

    def get(self, url: str, params: dict = None) -> BeautifulSoup | None:
        self._init_session()

        for attempt in range(SCRAPER_MAX_RETRY):
            try:
                if "race.netkeiba" in url:
                    self.session.headers.update({"Referer": "https://race.netkeiba.com/top/"})
                elif "db.netkeiba" in url:
                    self.session.headers.update({"Referer": "https://db.netkeiba.com/"})

                resp = self.session.get(url, params=params, timeout=SCRAPER_TIMEOUT)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding

                wait = self.delay + random.uniform(0.5, 1.5)
                time.sleep(wait)

                soup = BeautifulSoup(resp.text, "lxml")

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

    def get_browser(self, url: str, wait_selector: str = None) -> BeautifulSoup | None:
        """Playwrightヘッドレスブラウザでページを取得（JSレンダリング対応）"""
        logger.info(f"Playwright取得: {url}")
        return get_with_browser(url, wait_selector=wait_selector)
