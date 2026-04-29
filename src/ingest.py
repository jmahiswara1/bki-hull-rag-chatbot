import re
import sys
from pathlib import Path

import fitz
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import get_embeddings, ollama_setup_hint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "bki-rules-hull-2026.pdf"
CHROMA_DIR = PROJECT_ROOT / "chroma_db_v2"
COLLECTION_NAME = "bki_hull_2026_v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 350


def extract_page_text(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")

    if not blocks:
        return page.get_text("text").strip()

    sorted_blocks = sorted(blocks, key=lambda block: (round(block[1], 1), round(block[0], 1)))
    text_blocks = [str(block[4]).strip() for block in sorted_blocks if str(block[4]).strip()]
    return "\n".join(text_blocks).strip()


def extract_pdf_text(pdf_path: Path = PDF_PATH) -> list[Document]:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            "Place the BKI Rules PDF at data/bki-rules-hull-2026.pdf."
        )

    documents: list[Document] = []

    with fitz.open(str(pdf_path)) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            text = extract_page_text(page)

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_index + 1},
                    )
                )

    if not documents:
        raise RuntimeError(f"No text could be extracted from PDF: {pdf_path}")

    return documents


def is_table_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 12:
        return False

    numeric_lines = sum(1 for line in lines if re.search(r"\d", line))
    short_dense_lines = sum(1 for line in lines if len(line) <= 90 and len(line.split()) >= 3)
    formula_lines = sum(1 for line in lines if any(symbol in line for symbol in ("=", "≤", ">=", "<", ">")))

    return numeric_lines >= 5 and (short_dense_lines >= 8 or formula_lines >= 2)


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[Document] = []
    chunk_index = 1

    for document in documents:
        page_chunks = splitter.split_documents([document])

        if is_table_like(document.page_content):
            fallback_metadata = dict(document.metadata)
            fallback_metadata.update(
                {
                    "chunk_index": chunk_index,
                    "page_chunk_index": 0,
                    "chunk_type": "page_fallback",
                    "chunk_length": len(document.page_content),
                }
            )
            chunks.append(
                Document(page_content=document.page_content, metadata=fallback_metadata)
            )
            chunk_index += 1

        for page_chunk_index, chunk in enumerate(page_chunks, start=1):
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["page_chunk_index"] = page_chunk_index
            chunk.metadata["chunk_type"] = "chunk"
            chunk.metadata["chunk_length"] = len(chunk.page_content)
            chunks.append(chunk)
            chunk_index += 1

    if not chunks:
        raise RuntimeError("No chunks were created from the extracted PDF text.")

    return chunks


def build_vector_store(chunks: list[Document]) -> Chroma:
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        raise FileExistsError(
            f"Vector store already exists at {CHROMA_DIR}. "
            "Move or delete it before rebuilding the index."
        )

    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
            collection_name=COLLECTION_NAME,
        )

        return vector_store
    except Exception as exc:
        raise RuntimeError(
            "Failed to build the Chroma vector store.\n"
            + ollama_setup_hint()
            + f"\nOriginal error: {exc}"
        ) from exc


def main() -> int:
    try:
        documents = extract_pdf_text(PDF_PATH)
        chunks = split_documents(documents)
        page_fallbacks = sum(
            1 for chunk in chunks if chunk.metadata.get("chunk_type") == "page_fallback"
        )
        build_vector_store(chunks)
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded PDF: {PDF_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Extracted pages with text: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Page fallback chunks: {page_fallbacks}")
    print(f"Persisted vector store: {CHROMA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Collection: {COLLECTION_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
