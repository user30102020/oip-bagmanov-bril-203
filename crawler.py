"""
1) Берём список URL из urls.txt
2) Скачиваем HTML-страницы
3) Сохраняем каждую страницу в отдельный файл
4) Создаём index.txt: "0001.html<TAB>URL"
"""

from pathlib import Path
import time
import requests


URLS_FILE = "urls.txt"  # входной файл со списком ссылок
PROJECT_ROOT = Path(__file__).resolve().parent
DUMP_DIR = PROJECT_ROOT / "dump"     # папка, куда сохраняем html страницы
INDEX_FILE = PROJECT_ROOT / "index.txt"

SLEEP_SECONDS = 0.6             # пауза между запросами
TIMEOUT = 15                    # таймаут запроса


def read_urls(path: str) -> list[str]:
    """
    Считываем URLы из файла. Пропускаем пустые строки
    """
    urls = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        urls.append(line)
    return urls


def make_filename(i: int) -> str:
    """
    Делаем аккуратные имена файлов:
    0001.html, 0002.html, ...
    """
    return f"{i:04d}.html"


def download_page(session: requests.Session, url: str) -> tuple[bool, bytes, str]:
    """
    Скачивает страницу и возвращает:
    - ok: успешно ли (True/False)
    - content: байты страницы (HTML)
    - info: строка с пояснением при ошибке
    """
    try:
        resp = session.get(url, timeout=TIMEOUT, allow_redirects=True)

        #  проверяем код ответа
        if resp.status_code != 200:
            return False, b"", f"HTTP {resp.status_code}"

        return True, resp.content, "OK"

    except Exception as e:
        return False, b"", f"ERROR {type(e).__name__}: {e}"


def main():
    # 1) Читаем список ссылок
    urls = read_urls(URLS_FILE)

    # 2) Готовим папки
    DUMP_DIR.mkdir(parents=True, exist_ok=True)

    # 3) Создаём HTTP-сессию
    session = requests.Session()
    session.headers.update({
        "User-Agent": "SimpleUniCrawler/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    index_lines = []

    # 4) Идём по ссылкам и скачиваем
    for i, url in enumerate(urls, start=1):
        filename = make_filename(i)
        filepath = DUMP_DIR / filename

        ok, content, info = download_page(session, url)

        if ok:
            # Сохраняем HTML
            filepath.write_bytes(content)
            index_lines.append(f"{filename}\t{url}")
            print(f"{i}/{len(urls)} OK")
        else:
            # В случае ошибки файл не создаваем, но строку в index добавляем с пометкой FAIL
            index_lines.append(f"{filename}\t{url}\tFAIL {info}")
            print(f"[{i}/{len(urls)}] FAIL")

        # Пауза между запросами
        time.sleep(SLEEP_SECONDS)

    # 5) сохраняем index.txt
    INDEX_FILE.write_text("\n".join(index_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
