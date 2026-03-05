from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKENS_DIR = PROJECT_ROOT / "tokens"
LEMMAS_DIR = PROJECT_ROOT / "lemmas"

TASK4_DIR = Path(__file__).resolve().parent
TERMS_OUT_DIR = TASK4_DIR / "tfidf_terms"
LEMMAS_OUT_DIR = TASK4_DIR / "tfidf_lemmas"


def load_tokens_for_doc(path: Path) -> list[str]:
    """
    Читаем tokens/<doc>.txt.
    Здесь один термин на строку.
    """
    tokens: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip().lower()
        if term:
            tokens.append(term)
    return tokens


def load_lemma_counts_for_doc(path: Path) -> Counter[str]:
    """
    Читаем lemmas/<doc>.txt.
    Формат строки:
    <лемма> <токен1> <токен2> ... <токенN>

    Для леммы считаем количество её форм в строке.
    """
    lemma_counts: Counter[str] = Counter()

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        lemma = parts[0].lower()
        forms = [t for t in parts[1:] if t.strip()]
        if forms:
            lemma_counts[lemma] += len(forms)

    return lemma_counts


def compute_idf(df_map: dict[str, int], total_docs: int) -> dict[str, float]:
    """
    IDF = ln(N / df), где:
    N  - общее число документов,
    df - в скольких документах встретился термин/лемма.
    """
    idf_map: dict[str, float] = {}
    for item, df in df_map.items():
        idf_map[item] = log(total_docs / df)
    return idf_map


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
    token_files = sorted(TOKENS_DIR.glob("*.txt"))
    if not token_files:
        raise RuntimeError("No files found in tokens/")

    # Данные по документам:
    # doc_terms[doc_id]  -> Counter терминов
    # doc_lemmas[doc_id] -> Counter лемм
    # doc_total_terms[doc_id] -> общее число терминов в документе
    doc_terms: dict[str, Counter[str]] = {}
    doc_lemmas: dict[str, Counter[str]] = {}
    doc_total_terms: dict[str, int] = {}

    # Document frequency:
    # в скольких документах встретился термин/лемма
    term_df: dict[str, int] = defaultdict(int)
    lemma_df: dict[str, int] = defaultdict(int)

    for token_file in token_files:
        doc_id = token_file.stem
        lemma_file = LEMMAS_DIR / f"{doc_id}.txt"

        tokens = load_tokens_for_doc(token_file)
        term_counts = Counter(tokens)
        total_terms = len(tokens)

        if lemma_file.exists():
            lemma_counts = load_lemma_counts_for_doc(lemma_file)
        else:
            lemma_counts = Counter()

        doc_terms[doc_id] = term_counts
        doc_lemmas[doc_id] = lemma_counts
        doc_total_terms[doc_id] = total_terms

        # df считаем сразу в этом же цикле (без отдельной функции)
        for term in term_counts.keys():
            term_df[term] += 1
        for lemma in lemma_counts.keys():
            lemma_df[lemma] += 1

    total_docs = len(doc_terms)
    term_idf = compute_idf(dict(term_df), total_docs)
    lemma_idf = compute_idf(dict(lemma_df), total_docs)

    TERMS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEMMAS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for doc_id in sorted(doc_terms.keys()):
        write_doc_tfidf(
            out_path=TERMS_OUT_DIR / f"{doc_id}.txt",
            counts=doc_terms[doc_id],
            total_terms=doc_total_terms[doc_id],
            idf_map=term_idf,
        )
        write_doc_tfidf(
            out_path=LEMMAS_OUT_DIR / f"{doc_id}.txt",
            counts=doc_lemmas[doc_id],
            total_terms=doc_total_terms[doc_id],
            idf_map=lemma_idf,
        )

    print(f"Processed docs: {total_docs}")
    print(f"Terms output:  {TERMS_OUT_DIR}")
    print(f"Lemmas output: {LEMMAS_OUT_DIR}")


if __name__ == "__main__":
    main()
