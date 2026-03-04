from __future__ import annotations

from pathlib import Path
from collections import defaultdict


# Пути к данным второго задания и результатам третьего задания
TASK3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK3_DIR.parent
LEMMAS_DIR = PROJECT_ROOT / "lemmas"
ROOT_INDEX_FILE = PROJECT_ROOT / "index.txt"
INVERTED_INDEX_FILE = TASK3_DIR / "inverted_index.txt"


def build_inverted_index(lemmas_dir: Path = LEMMAS_DIR) -> dict[str, set[str]]:
    """
    Строим инвертированный индекс по файлам lemmas/*.txt.
    Для каждой леммы запоминаем список документов, где она встретилась.
    """
    inverted_index: dict[str, set[str]] = defaultdict(set)

    # Берём лемматизированные файлы из второго задания
    for lemma_file in sorted(lemmas_dir.glob("*.txt")):
        doc_id = lemma_file.stem

        for line in lemma_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            # В каждой строке первый элемент — это лемма
            lemma = line.split(maxsplit=1)[0]
            inverted_index[lemma].add(doc_id)

    return dict(inverted_index)


def save_inverted_index(
    inverted_index: dict[str, set[str]],
    index_path: Path = INVERTED_INDEX_FILE,
) -> None:
    """
    Сохраняем индекс в текстовый файл.
    Формат строки:
    <термин><TAB><doc_id 1><пробел><doc_id 2>...
    """
    lines = []

    # Для удобства чтения и проверки всё сортируем
    for term in sorted(inverted_index):
        doc_ids = sorted(inverted_index[term])
        lines.append(f"{term}\t{' '.join(doc_ids)}")

    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_inverted_index(index_path: Path = INVERTED_INDEX_FILE) -> dict[str, set[str]]:
    """
    Читаем готовый инвертированный индекс из файла.
    Формат строки:
    <термин><TAB><doc_id 1><пробел><doc_id 2>...
    """
    inverted_index: dict[str, set[str]] = {}

    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        term, _, docs_part = line.partition("\t")
        docs = set(filter(None, docs_part.split()))
        inverted_index[term] = docs

    return inverted_index


def main() -> None:
    # 1) Строим инвертированный индекс по данным второго задания
    inverted_index = build_inverted_index()

    # 2) Сохраняем его в отдельный файл для сдачи
    save_inverted_index(inverted_index)

    print(f"Saved inverted index to: {INVERTED_INDEX_FILE}")
    print(f"Terms: {len(inverted_index)}")


if __name__ == "__main__":
    main()
