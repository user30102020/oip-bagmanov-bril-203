from __future__ import annotations

import re
import sys
from collections import Counter
from math import sqrt
from pathlib import Path

from vector_index import (
    PROJECT_ROOT,
    VECTOR_INDEX_FILE,
    build_vector_index,
    load_vector_index,
    save_vector_index,
)


ROOT_INDEX_FILE = PROJECT_ROOT / "index.txt"
LEMMAS_DIR = PROJECT_ROOT / "lemmas"
TOKEN_RE = re.compile(r"[а-яё]{2,}", re.IGNORECASE)


def load_term_to_lemma(lemmas_dir: Path = LEMMAS_DIR) -> dict[str, str]:
    """
    Строим словарь:
    форма слова -> лемма

    Берём эти данные из готовых файлов задания 2.
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

            for token in parts[1:]:
                term_to_lemma[token.lower()] = lemma

    return term_to_lemma


def build_morph():
    """
    pymorphy3 не обязателен.
    Если библиотека есть, используем её как запасной вариант.
    """
    try:
        import pymorphy3
    except ModuleNotFoundError:
        return None
    return pymorphy3.MorphAnalyzer()


def normalize_term(term: str, term_to_lemma: dict[str, str], morph) -> str:
    """
    Приводим слово запроса к лемме.
    """
    term = term.lower()

    # Сначала пробуем найти слово в готовом словаре из задания 2.
    if term in term_to_lemma:
        return term_to_lemma[term]

    # Если pymorphy3 нет, оставляем слово как есть.
    if morph is None:
        return term

    return morph.parse(term)[0].normal_form


def load_doc_urls(index_path: Path = ROOT_INDEX_FILE) -> dict[str, str]:
    """
    Читаем index.txt и строим словарь:
    doc_id -> URL
    """
    doc_urls: dict[str, str] = {}

    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            continue

        doc_id = Path(parts[0]).stem
        doc_urls[doc_id] = parts[1]

    return doc_urls


def tokenize_query(query: str, term_to_lemma: dict[str, str], morph) -> list[str]:
    """
    Выделяем слова из запроса и приводим их к леммам.
    """
    lemmas: list[str] = []

    for match in TOKEN_RE.finditer(query.lower()):
        term = match.group(0)
        lemmas.append(normalize_term(term, term_to_lemma, morph))

    return lemmas


def build_query_vector(
    query: str,
    idf_map: dict[str, float],
    term_to_lemma: dict[str, str],
    morph,
) -> dict[str, float]:
    """
    Строим tf-idf вектор запроса.
    Используем тот же словарь idf, что и у документов.
    """
    lemmas = tokenize_query(query, term_to_lemma, morph)
    if not lemmas:
        return {}

    # Оставляем только те леммы, которые есть в общем индексе.
    counts = Counter(lemma for lemma in lemmas if lemma in idf_map)
    total_terms = sum(counts.values())
    if total_terms == 0:
        return {}

    query_vector: dict[str, float] = {}
    for lemma, count in counts.items():
        tf = count / total_terms
        query_vector[lemma] = tf * idf_map[lemma]

    return query_vector


def vector_norm(vector: dict[str, float]) -> float:
    """
    Считаем длину вектора.
    """
    return sqrt(sum(value * value for value in vector.values()))


def cosine_similarity(
    query_vector: dict[str, float],
    doc_vector: dict[str, float],
    query_norm: float,
    doc_norm: float,
) -> float:
    """
    Считаем cosine similarity между запросом и документом.
    """
    if query_norm == 0 or doc_norm == 0:
        return 0.0

    # Скалярное произведение считаем только по словам запроса.
    dot_product = 0.0
    for lemma, weight in query_vector.items():
        dot_product += weight * doc_vector.get(lemma, 0.0)

    return dot_product / (query_norm * doc_norm)


def unpack_index_data(index_data: dict[str, object]) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float]]:
    """
    После чтения json приводим числа обратно к float.
    """
    idf_map = {key: float(value) for key, value in dict(index_data["idf"]).items()}
    doc_vectors = {
        doc_id: {lemma: float(weight) for lemma, weight in vector.items()}
        for doc_id, vector in dict(index_data["doc_vectors"]).items()
    }
    doc_norms = {doc_id: float(norm) for doc_id, norm in dict(index_data["doc_norms"]).items()}
    return idf_map, doc_vectors, doc_norms


def search(
    query: str,
    index_data: dict[str, object],
    term_to_lemma: dict[str, str],
    morph,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Возвращаем top_k самых близких документов.
    """
    idf_map, doc_vectors, doc_norms = unpack_index_data(index_data)

    query_vector = build_query_vector(query, idf_map, term_to_lemma, morph)
    query_norm = vector_norm(query_vector)

    scores: list[tuple[str, float]] = []
    for doc_id, doc_vector in doc_vectors.items():
        score = cosine_similarity(query_vector, doc_vector, query_norm, doc_norms[doc_id])
        if score > 0:
            scores.append((doc_id, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def print_results(results: list[tuple[str, float]], doc_urls: dict[str, str]) -> None:
    """
    Печатаем:
    doc_id score url
    """
    print(f"Documents found: {len(results)}")
    for doc_id, score in results:
        print(f"{doc_id}\t{score:.8f}\t{doc_urls.get(doc_id, '-')}")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python task5/vector_search.py "ваш запрос" [top_k]')
        raise SystemExit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # Если индекс ещё не собран, строим его автоматически.
    if not VECTOR_INDEX_FILE.exists():
        index_data = build_vector_index()
        save_vector_index(index_data)
    else:
        index_data = load_vector_index()

    term_to_lemma = load_term_to_lemma()
    morph = build_morph()
    doc_urls = load_doc_urls()

    results = search(query, index_data, term_to_lemma, morph, top_k=top_k)
    print_results(results, doc_urls)


if __name__ == "__main__":
    main()
