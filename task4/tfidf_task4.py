from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path
import re

from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUMP_DIR = PROJECT_ROOT / "dump"
TOKENS_ALL_FILE = PROJECT_ROOT / "tokens_all.txt"
LEMMAS_ALL_FILE = PROJECT_ROOT / "lemmas_all.txt"

TASK4_DIR = Path(__file__).resolve().parent
TERMS_OUT_DIR = TASK4_DIR / "tfidf_terms"
LEMMAS_OUT_DIR = TASK4_DIR / "tfidf_lemmas"

# Токен: слово на кириллице, длина 2+
TOKEN_RE = re.compile(r"[а-яё]{2,}", re.IGNORECASE)


@dataclass
class DocStats:
    doc_id: str
    total_terms: int
    term_counts: Counter[str]
    lemma_counts: Counter[str]


def html_to_text(html: str) -> str:
    """
    Достаём текст из HTML и убираем script/style/noscript.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def load_terms_vocab(path: Path = TOKENS_ALL_FILE) -> set[str]:
    """
    Загружаем список терминов из задания 2.
    """
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_term_to_lemma(path: Path = LEMMAS_ALL_FILE) -> dict[str, str]:
    """
    Загружаем соответствия "токен -> лемма" из задания 2.
    """
    term_to_lemma: dict[str, str] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        lemma = parts[0].lower()
        term_to_lemma[lemma] = lemma
        for token in parts[1:]:
            term_to_lemma[token.lower()] = lemma

    return term_to_lemma


def tokenize_ru(text: str, terms_vocab: set[str]) -> list[str]:
    """
    Токенизируем текст и оставляем только термины из tokens_all.txt.
    """
    text = text.lower()
    tokens: list[str] = []

    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        if token in terms_vocab:
            tokens.append(token)

    return tokens


def collect_doc_stats(
    terms_vocab: set[str],
    term_to_lemma: dict[str, str],
) -> list[DocStats]:
    """
    Для каждого документа считаем:
    - число терминов;
    - частоты терминов;
    - частоты лемм.
    """
    docs: list[DocStats] = []

    for html_path in sorted(DUMP_DIR.glob("*.html")):
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(html)
        tokens = tokenize_ru(text, terms_vocab)

        term_counts = Counter(tokens)
        lemma_counts: Counter[str] = Counter()

        for term, count in term_counts.items():
            lemma = term_to_lemma.get(term, term)
            lemma_counts[lemma] += count

        docs.append(
            DocStats(
                doc_id=html_path.stem,
                total_terms=len(tokens),
                term_counts=term_counts,
                lemma_counts=lemma_counts,
            )
        )

    return docs


def compute_idf(df_map: dict[str, int], total_docs: int) -> dict[str, float]:
    """
    Считаем idf = ln(N / df), где:
    N  - общее число документов,
    df - число документов, где встретился термин/лемма.
    """
    idf_map: dict[str, float] = {}

    for item, df in df_map.items():
        idf_map[item] = log(total_docs / df)

    return idf_map


def build_df_maps(docs: list[DocStats]) -> tuple[dict[str, int], dict[str, int]]:
    """
    Считаем document frequency отдельно для терминов и лемм.
    """
    term_df: dict[str, int] = defaultdict(int)
    lemma_df: dict[str, int] = defaultdict(int)

    for doc in docs:
        for term in doc.term_counts.keys():
            term_df[term] += 1
        for lemma in doc.lemma_counts.keys():
            lemma_df[lemma] += 1

    return dict(term_df), dict(lemma_df)


def write_doc_tfidf(
    out_path: Path,
    counts: Counter[str],
    total_terms: int,
    idf_map: dict[str, float],
) -> None:
    """
    Формат строки:
    <термин или лемма><пробел><idf><пробел><tf-idf>
    """
    lines: list[str] = []

    if total_terms > 0:
        for item in sorted(counts.keys()):
            tf = counts[item] / total_terms
            idf = idf_map[item]
            tf_idf = tf * idf
            lines.append(f"{item} {idf:.8f} {tf_idf:.8f}")

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    # 1) Загружаем данные из задания 2
    terms_vocab = load_terms_vocab()
    term_to_lemma = load_term_to_lemma()

    # 2) Собираем статистику по документам
    docs = collect_doc_stats(terms_vocab, term_to_lemma)
    total_docs = len(docs)
    if total_docs == 0:
        raise RuntimeError("No HTML files found in dump/")

    # 3) Считаем df и idf
    term_df, lemma_df = build_df_maps(docs)
    term_idf = compute_idf(term_df, total_docs)
    lemma_idf = compute_idf(lemma_df, total_docs)

    # 4) Готовим папки для результатов
    TERMS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEMMAS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5) Для каждого документа пишем два файла:
    #    - tf-idf по терминам
    #    - tf-idf по леммам
    for doc in docs:
        write_doc_tfidf(
            out_path=TERMS_OUT_DIR / f"{doc.doc_id}.txt",
            counts=doc.term_counts,
            total_terms=doc.total_terms,
            idf_map=term_idf,
        )
        write_doc_tfidf(
            out_path=LEMMAS_OUT_DIR / f"{doc.doc_id}.txt",
            counts=doc.lemma_counts,
            total_terms=doc.total_terms,
            idf_map=lemma_idf,
        )

    print(f"Processed docs: {total_docs}")
    print(f"Terms output:  {TERMS_OUT_DIR}")
    print(f"Lemmas output: {LEMMAS_OUT_DIR}")


if __name__ == "__main__":
    main()
