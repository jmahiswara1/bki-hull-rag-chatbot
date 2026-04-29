import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from llm import get_embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db_v2"
COLLECTION_NAME = "bki_hull_2026_v2"
TOP_K = 5
DEFAULT_CANDIDATE_K = 20
DEFAULT_FINAL_K = 8

STOPWORDS = {
    "adalah",
    "agar",
    "akan",
    "atau",
    "dalam",
    "dan",
    "dengan",
    "di",
    "ini",
    "itu",
    "jika",
    "kami",
    "kapal",
    "ke",
    "pada",
    "saat",
    "saya",
    "sebagai",
    "untuk",
    "yang",
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def load_vector_store() -> Chroma:
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise RuntimeError(
            f"Vector store not found at {CHROMA_DIR}. Run `python src/ingest.py` first."
        )

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def get_retriever(k: int = TOP_K):
    return load_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+(?:[.,][0-9]+)?", text.lower())
    return {token for token in tokens if len(token) >= 2 and token not in STOPWORDS}


def extract_numeric_terms(text: str) -> set[str]:
    terms = set()

    for match in re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:mm|cm|m|kn|n|%|orang|persons?|sec|s)?\b", text.lower()):
        normalized = re.sub(r"\s+", "", match)
        if normalized:
            terms.add(normalized)

    for match in re.findall(r"\bes\d+\b", text.lower()):
        terms.add(match)

    return terms


def extract_phrases(text: str) -> set[str]:
    lowered = text.lower()
    phrases = set()

    for phrase in re.findall(r"[a-zA-Z][a-zA-Z0-9-]*(?:\s+[a-zA-Z][a-zA-Z0-9-]*){1,3}", lowered):
        words = [word for word in phrase.split() if word not in STOPWORDS]
        if len(words) >= 2:
            phrases.add(" ".join(words))

    return phrases


def copy_with_scores(document: Document, relevance_score: float, rerank_score: float) -> Document:
    metadata = dict(document.metadata)
    metadata["relevance_score"] = round(relevance_score, 4)
    metadata["rerank_score"] = round(rerank_score, 4)
    return Document(page_content=document.page_content, metadata=metadata)


def retrieve_candidates(query: str, candidate_k: int = DEFAULT_CANDIDATE_K) -> list[tuple[Document, float]]:
    vector_store = load_vector_store()

    try:
        candidates = vector_store.similarity_search_with_relevance_scores(query, k=candidate_k)
    except Exception:
        raw_documents = vector_store.similarity_search(query, k=candidate_k)
        candidates = [(document, 0.0) for document in raw_documents]

    return candidates


def rerank_documents(
    query: str,
    candidates: list[tuple[Document, float]],
    final_k: int = DEFAULT_FINAL_K,
) -> list[Document]:
    query_tokens = tokenize(query)
    query_numbers = extract_numeric_terms(query)
    query_phrases = extract_phrases(query)
    ranked: list[tuple[float, float, Document]] = []

    for document, relevance_score in candidates:
        content = document.page_content.lower()
        content_tokens = tokenize(content)
        content_numbers = extract_numeric_terms(content)
        overlap = len(query_tokens & content_tokens)
        overlap_ratio = overlap / max(len(query_tokens), 1)
        number_matches = len(query_numbers & content_numbers)
        phrase_matches = sum(1 for phrase in query_phrases if phrase in content)
        heading_bonus = 0.08 if any(word in content for word in ("table", "section", "sec.", "chapter")) else 0.0
        numeric_bonus = 0.18 * number_matches
        phrase_bonus = 0.06 * phrase_matches
        lexical_bonus = 0.45 * overlap_ratio
        rerank_score = float(relevance_score) + lexical_bonus + numeric_bonus + phrase_bonus + heading_bonus
        ranked.append((rerank_score, float(relevance_score), document))

    ranked.sort(key=lambda item: item[0], reverse=True)

    selected: list[Document] = []
    seen_chunks: set[tuple[str, str, str]] = set()
    page_counts: dict[tuple[str, str], int] = {}
    max_chunks_per_page = 2

    for rerank_score, relevance_score, document in ranked:
        source = str(document.metadata.get("source", ""))
        page = str(document.metadata.get("page", ""))
        chunk_index = str(document.metadata.get("chunk_index", ""))
        chunk_key = (source, page, chunk_index)
        page_key = (source, page)

        if chunk_key in seen_chunks:
            continue

        if page_counts.get(page_key, 0) >= max_chunks_per_page:
            continue

        seen_chunks.add(chunk_key)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        selected.append(copy_with_scores(document, relevance_score, rerank_score))

        if len(selected) >= final_k:
            break

    return selected


def retrieve_context(
    query: str,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_FINAL_K,
    min_score: float | None = None,
) -> list[Document]:
    candidates = retrieve_candidates(query, candidate_k)
    documents = rerank_documents(query, candidates, final_k)

    if min_score is None:
        return documents

    filtered = [
        document
        for document in documents
        if float(document.metadata.get("rerank_score", 0.0)) >= min_score
    ]

    return filtered
