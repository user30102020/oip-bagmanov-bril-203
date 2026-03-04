from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from inverted_index import (
    INVERTED_INDEX_FILE,
    LEMMAS_DIR,
    ROOT_INDEX_FILE,
    build_inverted_index,
    load_inverted_index,
    save_inverted_index,
)


# Разбиваем запрос на части: скобки, операторы AND/OR/NOT и слова.
TOKEN_RE = re.compile(r"\(|\)|AND|OR|NOT|[^\s()]+", re.IGNORECASE)
OPERATORS = {"AND", "OR", "NOT"}


class Lemmatizer:
    def __init__(self) -> None:
        # Загружаем готовые соответствия из lemmas/*.txt:
        # например, "агенты" -> "агент", "агента" -> "агент".
        self._term_to_lemma = self._load_term_to_lemma()

        try:
            import pymorphy3
        except ModuleNotFoundError:
            # Библиотека не обязательна: если её нет, работаем только по
            # словарю из ваших файлов lemmas/*.txt.
            self._morph = None
        else:
            self._morph = pymorphy3.MorphAnalyzer()

    def normalize(self, term: str) -> str:
        """
        Возвращает нормальную форму слова из запроса.
        Пример: "агентами" -> "агент".
        """
        term = term.lower()
        # Сначала проверяем в нашем словаре из lemmas/*.txt.
        if term in self._term_to_lemma:
            return self._term_to_lemma[term]
        # Если внешнего анализатора нет, возвращаем как есть.
        if self._morph is None:
            return term
        # Иначе пробуем лемматизировать через pymorphy3.
        return self._morph.parse(term)[0].normal_form

    @staticmethod
    def _load_term_to_lemma(lemmas_dir: Path = LEMMAS_DIR) -> dict[str, str]:
        """
        Читает все файлы в lemmas/ и делает словарь:
        "любая форма слова" -> "лемма".
        """
        term_to_lemma: dict[str, str] = {}

        for lemma_file in sorted(lemmas_dir.glob("*.txt")):
            for line in lemma_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                lemma = parts[0].lower()
                term_to_lemma[lemma] = lemma

                # Остальные слова в строке — формы этой же леммы.
                for token in parts[1:]:
                    term_to_lemma[token.lower()] = lemma

        return term_to_lemma


def load_doc_urls(index_path: Path = ROOT_INDEX_FILE) -> dict[str, str]:
    """
    Читаем index.txt и строим словарь:
    номер документа -> исходный URL
    """
    doc_urls: dict[str, str] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            continue

        filename = parts[0]
        url = parts[1]
        doc_id = Path(filename).stem
        doc_urls[doc_id] = url

    return doc_urls


def tokenize_query(query: str) -> list[str]:
    """
    Делим строку запроса на токены.
    Пример:
    "(агент AND токен) OR NOT дилемма"
    -> ["(", "агент", "AND", "токен", ")", "OR", "NOT", "дилемма"]
    """
    tokens = TOKEN_RE.findall(query)
    if not tokens:
        raise ValueError("Query is empty")
    return tokens


@dataclass
class Parser:
    tokens: list[str]
    all_docs: set[str]
    inverted_index: dict[str, set[str]]
    lemmatizer: Lemmatizer
    pos: int = 0

    def current(self) -> str | None:
        # Возвращает текущий токен (на который сейчас "смотрит" парсер).
        # Если токены закончились, возвращает None.
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def consume(self, expected: str | None = None) -> str:
        # "Съедает" текущий токен и сдвигает позицию на следующий.
        # Если передан expected, дополнительно проверяет,
        # что текущий токен именно такой (например "(" или "AND").
        token = self.current()
        if token is None:
            raise ValueError("Unexpected end of query")
        if expected is not None and token.upper() != expected:
            raise ValueError(f"Expected {expected}, got {token}")
        self.pos += 1
        return token

    def parse(self) -> set[str]:
        # Вход в парсер: считаем запрос целиком.
        result = self.parse_or()
        if self.current() is not None:
            raise ValueError(f"Unexpected token: {self.current()}")
        return result

    def parse_or(self) -> set[str]:
        # OR выполняется последним (самый низкий приоритет).
        result = self.parse_and()
        while self.current() and self.current().upper() == "OR":
            self.consume("OR")
            result |= self.parse_and()
        return result

    def parse_and(self) -> set[str]:
        # AND выполняется раньше, чем OR.
        result = self.parse_not()
        while self.current() and self.current().upper() == "AND":
            self.consume("AND")
            result &= self.parse_not()
        return result

    def parse_not(self) -> set[str]:
        # NOT: берём все документы и вычитаем выражение справа.
        if self.current() and self.current().upper() == "NOT":
            self.consume("NOT")
            return self.all_docs - self.parse_not()
        return self.parse_term()

    def parse_term(self) -> set[str]:
        token = self.current()
        if token is None:
            raise ValueError("Expected a term or '('")

        # Если встретили "(", сначала считаем выражение в скобках.
        if token == "(":
            self.consume("(")
            result = self.parse_or()
            self.consume(")")
            return result

        if token.upper() in OPERATORS:
            raise ValueError(f"Expected a term, got operator {token}")

        term = self.consume()
        # Нормализуем слово и берём список документов из индекса.
        normalized = self.lemmatizer.normalize(term)
        return set(self.inverted_index.get(normalized, set()))


def evaluate_query(
    query: str,
    inverted_index: dict[str, set[str]],
    all_docs: set[str],
    lemmatizer: Lemmatizer,
) -> list[str]:
    """
    Считаем булев запрос и возвращаем отсортированные id документов.
    """
    parser = Parser(
        tokens=tokenize_query(query),
        all_docs=all_docs,
        inverted_index=inverted_index,
        lemmatizer=lemmatizer,
    )
    return sorted(parser.parse())


def print_results(doc_ids: list[str], doc_urls: dict[str, str]) -> None:
    # Печатаем результат: id документа и ссылка.
    print(f"Documents found: {len(doc_ids)}")
    for doc_id in doc_ids:
        url = doc_urls.get(doc_id, "-")
        print(f"{doc_id}\t{url}")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python task3/boolean_search.py "(term1 AND term2) OR NOT term3"')
        raise SystemExit(1)

    # Берём запрос из командной строки.
    query = " ".join(sys.argv[1:])

    # Если индекс ещё не собран, строим его автоматически.
    if not INVERTED_INDEX_FILE.exists():
        inverted_index = build_inverted_index()
        save_inverted_index(inverted_index, INVERTED_INDEX_FILE)
    else:
        inverted_index = load_inverted_index(INVERTED_INDEX_FILE)

    # Готовим данные для поиска.
    doc_urls = load_doc_urls()
    all_docs = set(doc_urls)
    lemmatizer = Lemmatizer()

    # Выполняем запрос и печатаем результат.
    result = evaluate_query(query, inverted_index, all_docs, lemmatizer)
    print_results(result, doc_urls)


if __name__ == "__main__":
    main()
