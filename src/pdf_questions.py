import re
from pathlib import Path

import fitz

QUESTION_LABEL_PATTERN = re.compile(r"^pertanyaan(?:\s+\d+)?\s*[:：-]\s*(.*)$", re.IGNORECASE)
NUMBERED_ITEM_PATTERN = re.compile(r"^\d+\s*[.)-]\s*(.+)$")
QUESTION_WORDS = (
    "apa",
    "apakah",
    "bagaimana",
    "berapa",
    "kenapa",
    "mengapa",
    "kapan",
    "dimana",
    "which",
    "what",
    "when",
    "where",
    "why",
    "how",
)


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def clean_question(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([?.!,;:])", r"\1", cleaned)


def is_question_like(text: str) -> bool:
    normalized = text.lower().strip()
    return normalized.endswith("?") or normalized.startswith(QUESTION_WORDS)


def extract_pdf_lines(file_path: Path) -> list[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    lines: list[str] = []

    with fitz.open(str(file_path)) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            page_text = page.get_text("text")
            lines.extend(
                normalized
                for line in page_text.splitlines()
                if (normalized := normalize_line(line))
            )

    if not lines:
        raise ValueError(
            "No text could be extracted from the PDF. The file may contain scanned "
            "images and requires OCR or conversion to an editable format."
        )

    return lines


def extract_questions_from_lines(lines: list[str]) -> list[str]:
    questions: list[str] = []
    current_context: list[str] = []
    current_question: list[str] = []

    def finish_question() -> None:
        if not current_question:
            return

        question = clean_question(" ".join(current_question))
        if question:
            questions.append(question)

        current_question.clear()

    for line in lines:
        question_match = QUESTION_LABEL_PATTERN.match(line)
        if question_match:
            finish_question()
            current_question.extend(current_context)
            question_text = question_match.group(1).strip()
            if question_text:
                current_question.append(question_text)
            continue

        numbered_match = NUMBERED_ITEM_PATTERN.match(line)
        if numbered_match:
            finish_question()
            numbered_text = numbered_match.group(1).strip()

            if is_question_like(numbered_text):
                current_context = []
                current_question.append(numbered_text)
            else:
                current_context = [numbered_text]

            continue

        if current_question:
            current_question.append(line)
            continue

        if is_question_like(line):
            current_question.extend(current_context)
            current_question.append(line)

    finish_question()
    return questions


def load_questions_from_pdf(file_path: Path) -> list[str]:
    lines = extract_pdf_lines(file_path)
    questions = extract_questions_from_lines(lines)

    if not questions:
        raise ValueError(
            "No questions were detected in the PDF text. The file may contain image "
            "tables/questions that require OCR, or it should be converted to JSON, CSV, "
            "XLSX, or an editable DOCX table first."
        )

    return questions
