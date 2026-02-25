from pathlib import Path
import re
from collections import defaultdict

import pymorphy3
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


DUMP_DIR = Path("dump")
TOKENS_DIR = Path("tokens")
LEMMAS_DIR = Path("lemmas")

TOKENS_ALL_FILE = Path("tokens_all.txt")
LEMMAS_ALL_FILE = Path("lemmas_all.txt")

# Токен = слово на кириллице, длина 2+
TOKEN_RE = re.compile(r"[а-яё]{2,}", re.IGNORECASE)


def html_to_text(html: str) -> str:
    """
    Достаём текст из HTML для корректной токенизации.
    """

    # Парсим HTML в дерево тегов
    soup = BeautifulSoup(html, "lxml")

    # Скрипты/стили не являются текстом статьи
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return soup.get_text(separator=" ")


def tokenize_ru(text: str, ru_stop: set[str]) -> list[str]:
    """
    Выделяем токены:
    - только кириллица
    - нижний регистр
    - исключаем стоп-слова: союзы/предлоги и т.п.
    """
    text = text.lower()
    tokens = []

    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok in ru_stop:
            continue
        tokens.append(tok)

    return tokens


def write_tokens_file(out_path: Path, tokens: list[str]) -> None:
    """Формат: <токен>\n"""
    out_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def write_lemmas_file(out_path: Path, lemma_map: dict[str, set[str]]) -> None:
    """Формат: <лемма> <токен1> <токен2> ... <токенN>\n"""
    lines = []
    for lemma in sorted(lemma_map.keys()):
        toks = sorted(lemma_map[lemma])
        lines.append(lemma + " " + " ".join(toks))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    # Стоп-слова русского языка (союзы/предлоги и т.п.)
    ru_stop = set(stopwords.words("russian"))

    # Лемматизатор pymorphy3 умеет разбирать русские слова и выдавать нормальную форму.
    morph = pymorphy3.MorphAnalyzer()

    # Папки для результатов
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    LEMMAS_DIR.mkdir(parents=True, exist_ok=True)

    # Берём все HTML-страницы из dump/
    html_files = sorted(DUMP_DIR.glob("*.html"))

    all_tokens_set: set[str] = set()  # все токены со всех страниц (уникальные)
    all_lemma_map: dict[str, set[str]] = defaultdict(set)   # лемма -> множество токенов со всех страниц

    for fp in html_files:
        html = fp.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(html)

        # Токенизируем, получаем список, затем делаем set,
        # чтобы убрать дубликаты токенов на этой странице.
        token_set = set(tokenize_ru(text, ru_stop))

        tokens_sorted = sorted(token_set)

        # Группировка по леммам для конкретной страницы.
        # lemma_map: лемма -> множество токенов, которые дали эту лемму.
        lemma_map: dict[str, set[str]] = defaultdict(set)
        for tok in tokens_sorted:
            lemma = morph.parse(tok)[0].normal_form
            lemma_map[lemma].add(tok)

        # dump/0001.html -> tokens/0001.txt, lemmas/0001.txt
        name = fp.stem
        write_tokens_file(TOKENS_DIR / f"{name}.txt", tokens_sorted)
        write_lemmas_file(LEMMAS_DIR / f"{name}.txt", lemma_map)

        # Добавляем в общие файлы
        all_tokens_set.update(token_set)
        for lemma, toks in lemma_map.items():
            all_lemma_map[lemma].update(toks)

        print(f"{fp.name}: tokens={len(tokens_sorted)}, lemmas={len(lemma_map)}")

    # Сохраняем общие файлы
    tokens_all_sorted = sorted(all_tokens_set)
    write_tokens_file(TOKENS_ALL_FILE, tokens_all_sorted)
    write_lemmas_file(LEMMAS_ALL_FILE, all_lemma_map)

    print(f"tokens_all: ({len(tokens_all_sorted)} tokens)")
    print(f"lemmas_all: ({len(all_lemma_map)} lemmas)")


if __name__ == "__main__":
    main()