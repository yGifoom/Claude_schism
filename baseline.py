import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def build_index(documents: list[str]) -> dict:
    tf = []
    df = Counter()

    for doc in documents:
        tokens = tokenize(doc)
        counts = Counter(tokens)
        total = len(tokens) or 1
        tf.append({term: count / total for term, count in counts.items()})
        df.update(counts.keys())

    n = len(documents)
    idf = {term: math.log(n / freq) for term, freq in df.items()}

    tfidf = [
        {term: tf_val * idf[term] for term, tf_val in doc_tf.items()}
        for doc_tf in tf
    ]

    return {"tfidf": tfidf, "idf": idf, "n": n}


def search(index: dict, query: str, k: int = 10) -> list[int]:
    idf = index["idf"]
    tfidf = index["tfidf"]

    tokens = tokenize(query)
    counts = Counter(tokens)
    total = len(tokens) or 1
    query_vec = {term: (count / total) * idf.get(term, 0) for term, count in counts.items()}

    def dot(doc_vec: dict) -> float:
        return sum(query_vec.get(term, 0) * val for term, val in doc_vec.items())

    scores = [(i, dot(doc_vec)) for i, doc_vec in enumerate(tfidf)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, score in scores[:k] if score > 0]
