from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request

from vector_search import (
    VECTOR_INDEX_FILE,
    build_morph,
    load_doc_urls,
    load_term_to_lemma,
    load_vector_index,
    save_vector_index,
    search,
)
from vector_index import build_vector_index


# Путь к папке, где лежит этот файл: task5/
BASE_DIR = Path(__file__).resolve().parent

# Говорим Flask, что шаблоны лежат в task5/templates
app = Flask(__name__, template_folder=str(BASE_DIR))

# Загружаем индекс и служебные данные один раз при старте приложения,
# чтобы не делать это заново на каждый запрос пользователя.
if not VECTOR_INDEX_FILE.exists():
    index_data = build_vector_index()
    save_vector_index(index_data)
else:
    index_data = load_vector_index()

term_to_lemma = load_term_to_lemma()
morph = build_morph()
doc_urls = load_doc_urls()


@app.route("/", methods=["GET"])
def home():
    query = request.args.get("q", "").strip()
    results_for_template = []

    if query:
        # top_k=10 -> выводим только топ-10 результатов
        results = search(
            query=query,
            index_data=index_data,
            term_to_lemma=term_to_lemma,
            morph=morph,
            top_k=10,
        )

        # Подготавливаем данные для HTML-шаблона
        for doc_id, score in results:
            results_for_template.append(
                {
                    "doc_id": doc_id,
                    "score": score,
                    "url": doc_urls.get(doc_id, "-"),
                }
            )

    return render_template(
        "search.html",
        query=query,
        results=results_for_template,
    )


if __name__ == "__main__":
    app.run(debug=True)