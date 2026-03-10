from __future__ import annotations

import json
from math import sqrt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TFIDF_LEMMAS_DIR = PROJECT_ROOT / "task4" / "tfidf_lemmas"
VECTOR_INDEX_FILE = Path(__file__).resolve().parent / "vector_index.json"


def vector_norm(vector: dict[str, float]) -> float:
    """
    Считаем длину вектора.
    Она нужна для cosine similarity.
    """
    return sqrt(sum(value * value for value in vector.values()))


def build_vector_index(tfidf_dir: Path = TFIDF_LEMMAS_DIR) -> dict[str, object]:
    """
    Собираем общий векторный индекс по файлам task4/tfidf_lemmas/*.txt.

    Что храним:
    - idf по леммам;
    - вектор каждого документа;
    - длину каждого документа.
    """
    doc_vectors: dict[str, dict[str, float]] = {}
    doc_norms: dict[str, float] = {}
    idf_map: dict[str, float] = {}

    for path in sorted(tfidf_dir.glob("*.txt")):
        doc_id = path.stem
        doc_vector: dict[str, float] = {}

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            # Формат строки:
            # <лемма> <idf> <tf-idf>
            lemma, idf, tf_idf = line.split()
            doc_vector[lemma] = float(tf_idf)

            # Для каждой леммы достаточно один раз сохранить её idf.
            if lemma not in idf_map:
                idf_map[lemma] = float(idf)

        doc_vectors[doc_id] = doc_vector
        doc_norms[doc_id] = vector_norm(doc_vector)

    return {
        "idf": idf_map,
        "doc_vectors": doc_vectors,
        "doc_norms": doc_norms,
    }


def save_vector_index(index_data: dict[str, object], out_path: Path = VECTOR_INDEX_FILE) -> None:
    """
    Сохраняем индекс в json.
    """
    out_path.write_text(
        json.dumps(index_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_vector_index(index_path: Path = VECTOR_INDEX_FILE) -> dict[str, object]:
    """
    Читаем готовый индекс из json.
    """
    return json.loads(index_path.read_text(encoding="utf-8"))


def main() -> None:
    index_data = build_vector_index()
    save_vector_index(index_data)

    print(f"Saved vector index to: {VECTOR_INDEX_FILE}")
    print(f"Documents: {len(index_data['doc_vectors'])}")
    print(f"Lemmas: {len(index_data['idf'])}")


if __name__ == "__main__":
    main()
