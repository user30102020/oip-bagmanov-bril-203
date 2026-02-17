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


def is_html_response(resp: requests.Response) -> bool:
    """
    Проверяем, что скачали именно HTML-текст
    """
    content_type = resp.headers.get("Content-Type", "").lower()
    return "text/html" in content_type


def download_page(session: requests.Session, url: str) -> tuple[bool, bytes, str]:
    """
    Скачивает страницу и возвращает:
    - ok: успешно ли (True/False)
    - content: байты страницы (HTML как есть)
    - info: строка с пояснением (для логов/index при ошибке)
    """
    try:
        resp = session.get(url, timeout=TIMEOUT, allow_redirects=True)

        # 1) код ответа должен быть 200
        if resp.status_code != 200:
            return False, b"", f"HTTP {resp.status_code}"

        # 2) должен быть HTML (текстовая страница)
        if not is_html_response(resp):
            ct = resp.headers.get("Content-Type", "")
            return False, b"", f"Not HTML (Content-Type: {ct})"

        # 3) контент должен быть не пустой (минимальная проверка)
        content = resp.content
        if len(content) < 100:
            return False, b"", "Too small content"

        return True, content, "OK"

    except Exception as e:
        return False, b"", f"ERROR {type(e).__name__}: {e}"


def main():
    # 1) Читаем список ссылок
    urls = read_urls(URLS_FILE)
    if not urls:
        print(f"Файл {URLS_FILE} пуст или не найден.")
        return

    # 2) Готовим папки
    DUMP_DIR.mkdir(parents=True, exist_ok=True)

    # 3) Создаём HTTP-сессию
    session = requests.Session()
    session.headers.update({
        "User-Agent": "SimpleUniCrawler/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    index_lines = []
    ok_count = 0
    fail_count = 0

    # 4) Идём по ссылкам и скачиваем
    for i, url in enumerate(urls, start=1):
        filename = make_filename(i)
        filepath = DUMP_DIR / filename

        ok, content, info = download_page(session, url)

        if ok:
            # Сохраняем HTML как есть (не чистим разметку)
            filepath.write_bytes(content)
            index_lines.append(f"{filename}\t{url}")
            ok_count += 1
            print(f"{i}/{len(urls)} OK")
        else:
            # В случае ошибки файл можно не создавать, но строку в index добавим с пометкой FAIL
            index_lines.append(f"{filename}\t{url}\tFAIL {info}")
            fail_count += 1
            print(f"[{i}/{len(urls)}] FAIL {url} ({info})")

        # Пауза между запросами
        time.sleep(SLEEP_SECONDS)

    # 5) сохраняем index.txt
    INDEX_FILE.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print(f"\nУспешно: {ok_count}")
    print(f"Ошибок:  {fail_count}")


if __name__ == "__main__":
    main()
