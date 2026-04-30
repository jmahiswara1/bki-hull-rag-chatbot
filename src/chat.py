import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from langchain_core.documents import Document

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.markup import escape
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
except ImportError:
    Console = None
    Markdown = None
    escape = None
    Panel = None
    Rule = None
    Table = None

from llm import FALLBACK_LLM_MODEL, get_llm_with_fallback
from pdf_questions import load_questions_from_pdf
from retriever import is_numeric_or_rule_query, retrieve_candidates, retrieve_context, rerank_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RICH_AVAILABLE = Console is not None
console = Console() if RICH_AVAILABLE else None
error_console = Console(stderr=True) if RICH_AVAILABLE else None
NORMAL_MODE = "normal"
FAST_MODE = "fast"
LLAMA_MODE = "llama"
LLAMA_MODEL = "llama3.2:3b"

MODE_CONFIGS = {
    NORMAL_MODE: {
        "candidate_k": 24,
        "final_k": 8,
        "history_turns": 10,
        "description": "qwen2.5:7b fallback, deeper retrieval, full history",
    },
    FAST_MODE: {
        "candidate_k": 16,
        "final_k": 5,
        "history_turns": 3,
        "description": "qwen2.5:3b, reranked retrieval, short history, grounded answers",
    },
    LLAMA_MODE: {
        "candidate_k": 36,
        "final_k": 5,
        "history_turns": 2,
        "min_score": 0.18,
        "description": "llama3.2:3b, stricter retrieval, compact grounded answers",
    },
}

FOLLOW_UP_MARKERS = (
    "tersebut",
    "di atas",
    "sebelumnya",
    "nilai itu",
    "ketebalannya",
    "lebarnya",
    "tingginya",
    "diameternya",
    "berapa lagi",
    "bagaimana dengan",
    "that",
    "those",
    "the above",
    "previous",
    "what about",
    "how about",
)

INDONESIAN_MARKERS = (
    " apa ",
    " itu ",
    " yang ",
    " dan ",
    " atau ",
    " berapa ",
    " bagaimana ",
    " jelaskan ",
    " ketebalan ",
    " kapal ",
)

INDONESIAN_RULE_TERM_EXPANSIONS = (
    (("tebal", "ketebalan"), ("thickness", "plate thickness")),
    (("senta", "penegar", "stiffener"), ("stiffener", "longitudinal stiffener")),
    (("jarak", "spasi"), ("spacing", "unsupported span")),
    (("modulus",), ("section modulus",)),
    (("pelat sisi",), ("side shell plating", "side plate")),
    (("alas",), ("bottom", "bottom plating")),
    (("double bottom", "dasar ganda"), ("double bottom",)),
    (("strut",), ("strut", "support")),
    (("palka", "bukaan palka"), ("hatch", "hatch opening")),
    (("bulk carrier", "kapal curah"), ("bulk carrier", "dry cargo in bulk")),
    (("haluan",), ("fore part", "fore end", "bow")),
    (("persyaratan", "harus", "perlu"), ("requirement", "shall")),
    (("minimum", "maksimum"), ("minimum", "maximum")),
    (("rumus",), ("formula",)),
    (("tabel",), ("table",)),
    (("konstruksi memanjang",), ("longitudinal framing", "longitudinal construction")),
)

GENERIC_NON_ANSWER_PATTERNS = (
    "secara umum",
    "tergantung pada beberapa faktor",
    "perlu mempertimbangkan",
    "tidak dapat ditentukan",
    "tidak dapat menjawab",
    "informasi umum",
    "bervariasi tergantung",
    "mengacu pada aturan",
    "based on several factors",
    "depends on several factors",
    "cannot determine",
    "general information",
)

CONTEXT_MATCH_STOPWORDS = {
    "adalah",
    "agar",
    "akan",
    "atau",
    "dalam",
    "dan",
    "dengan",
    "jika",
    "kapal",
    "pada",
    "untuk",
    "yang",
    "and",
    "are",
    "for",
    "the",
    "with",
}


def rich_escape(value: str) -> str:
    if escape is None:
        return value

    return escape(value)


def mode_badge(mode: str) -> str:
    if not RICH_AVAILABLE:
        return f"[{mode}]"

    styles = {
        NORMAL_MODE: "bold cyan",
        FAST_MODE: "bold green",
        LLAMA_MODE: "bold magenta",
    }
    style = styles.get(mode, "bold cyan")
    return f"[{style}]{rich_escape(f'[{mode}]')}[/{style}]"


def print_status(message: str, style: str = "cyan") -> None:
    if console is None:
        print(message)
        return

    console.print(message, style=style)


def print_error(message: str) -> None:
    if error_console is None or Panel is None:
        print(message, file=sys.stderr)
        return

    error_console.print(Panel(message, title="Error", border_style="red"))


def print_panel(title: str, body: str, style: str = "cyan") -> None:
    if console is None or Panel is None:
        print(title)
        print(body)
        return

    console.print(Panel(body, title=title, border_style=style))


def print_rule(title: str) -> None:
    if console is None or Rule is None:
        print(f"\n=== {title} ===")
        return

    console.print(Rule(title, style="dim"))


def prompt_user(mode: str) -> str:
    prompt = f"\n{mode_badge(mode)} [bold]You[/bold] > " if RICH_AVAILABLE else f"\n[{mode}] You > "

    if console is None:
        return input(prompt).strip()

    return console.input(prompt).strip()


def parse_command(user_input: str) -> tuple[str, str]:
    command, _, args = user_input.partition(" ")
    return command.lower(), args.strip()


def print_welcome() -> None:
    body = "\n".join(
        [
            "Ask questions in Bahasa Indonesia or English.",
            "Type /help for commands, /quit or /exit to leave.",
            "Modes: normal, fast, llama.",
        ]
    )
    print_panel("BKI Hull Rules Chatbot CLI", body, "cyan")


def clear_terminal() -> None:
    if console is not None:
        console.clear()
        return

    print("\033[2J\033[H", end="", flush=True)


def print_help() -> None:
    if console is None or Table is None:
        print("Available commands:")
        print("  /help                    Show this help message")
        print("  /clear                   Clear screen and conversation history")
        print("  /fast                    Use qwen2.5:3b with grounded concise answers")
        print("  /llama                   Use llama3.2:3b with grounded concise answers")
        print("  /normal                  Use qwen2.5:7b fallback with deeper retrieval")
        print("  /debug-retrieve <q>      Show retrieved chunks and scores for a question")
        print("  /import-json [path]      Ask independent questions from JSON, default: data/questions.json")
        print("  /import-pdf [path]       Ask independent questions from PDF, default: data/testing.pdf")
        print("  /quit                    Exit the chatbot")
        print("  /exit                    Exit the chatbot")
        return

    table = Table(title="Available commands", show_header=True, header_style="bold cyan")
    table.add_column("Group", style="dim", no_wrap=True)
    table.add_column("Command", style="bold")
    table.add_column("Description")
    table.add_row("Conversation", "/help", "Show this help message")
    table.add_row("Conversation", "/clear", "Clear screen and conversation history")
    table.add_row("Modes", "/fast", "Use qwen2.5:3b with grounded concise answers")
    table.add_row("Modes", "/llama", "Use llama3.2:3b with grounded concise answers")
    table.add_row("Modes", "/normal", "Use qwen2.5:7b fallback with deeper retrieval")
    table.add_row("Retrieval", "/debug-retrieve <q>", "Show retrieved chunks and scores for a question")
    table.add_row("Import", rich_escape("/import-json [path]"), "Ask independent questions from JSON, default: data/questions.json")
    table.add_row("Import", rich_escape("/import-pdf [path]"), "Ask independent questions from PDF, default: data/testing.pdf")
    table.add_row("Exit", "/quit", "Exit the chatbot")
    table.add_row("Exit", "/exit", "Exit the chatbot")
    console.print(table)


@contextmanager
def show_loading() -> Generator[None, None, None]:
    if console is None or not sys.stdout.isatty():
        print("...", flush=True)
        yield
        return

    with console.status("Retrieving context and asking model...", spinner="dots"):
        yield


def get_mode_config(mode: str) -> dict[str, Any]:
    return MODE_CONFIGS.get(mode, MODE_CONFIGS[NORMAL_MODE])


def get_history_limit(mode: str) -> int:
    return int(get_mode_config(mode)["history_turns"])


def get_cached_llm(mode: str, llm_cache: dict[str, Any]) -> Any:
    if mode in llm_cache:
        return llm_cache[mode]

    if mode == FAST_MODE:
        llm_cache[mode] = get_llm_with_fallback(
            preferred_model=FALLBACK_LLM_MODEL,
            fallback_model=FALLBACK_LLM_MODEL,
        )
    elif mode == LLAMA_MODE:
        llm_cache[mode] = get_llm_with_fallback(
            preferred_model=LLAMA_MODEL,
            fallback_model=LLAMA_MODEL,
            num_ctx=4096,
            num_predict=350,
            top_p=0.2,
            top_k=20,
            repeat_penalty=1.05,
        )
    else:
        llm_cache[mode] = get_llm_with_fallback()

    return llm_cache[mode]


def get_response_language(query: str) -> str:
    normalized_query = f" {query.lower()} "

    if any(marker in normalized_query for marker in INDONESIAN_MARKERS):
        return "Bahasa Indonesia"

    return "English"


def is_follow_up_question(query: str) -> bool:
    normalized_query = f" {query.lower()} "
    return any(marker in normalized_query for marker in FOLLOW_UP_MARKERS)


def expand_indonesian_rule_terms(query: str) -> str:
    lowered = query.lower()
    expanded_terms: list[str] = []
    seen: set[str] = set()

    for triggers, terms in INDONESIAN_RULE_TERM_EXPANSIONS:
        if not any(trigger in lowered for trigger in triggers):
            continue

        for term in terms:
            if term in seen:
                continue

            seen.add(term)
            expanded_terms.append(term)

    return ", ".join(expanded_terms)


def build_retrieval_query(
    query: str,
    history: list[tuple[str, str]],
    use_history: bool,
) -> str:
    if not use_history or not history or not is_follow_up_question(query):
        retrieval_query = query
    else:
        last_user_message = history[-1][0]
        retrieval_query = f"Previous user question: {last_user_message}\nCurrent user question: {query}"

    expanded_terms = expand_indonesian_rule_terms(retrieval_query)
    if not expanded_terms:
        return retrieval_query

    return f"{retrieval_query}\nRelated BKI English terms: {expanded_terms}"


def format_history(
    history: list[tuple[str, str]],
    max_turns: int,
    use_history: bool,
) -> str:
    if not use_history or not history:
        return "No previous conversation."

    lines = ["Previous conversation:"]
    for user_message, assistant_message in history[-max_turns:]:
        lines.append(f"User: {user_message}")
        lines.append(f"Assistant: {assistant_message}")

    return "\n".join(lines)


def format_context(documents: list[Document], query: str = "") -> str:
    if not documents:
        return "No relevant context was retrieved."

    chunks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown source")
        page = document.metadata.get("page", "unknown page")
        chunk_index = document.metadata.get("chunk_index", "unknown chunk")
        chunk_type = document.metadata.get("chunk_type", "unknown type")
        relevance_score = document.metadata.get("relevance_score", "n/a")
        rerank_score = document.metadata.get("rerank_score", "n/a")
        content = document.page_content.strip()
        matched_terms = matched_context_terms(query, content)
        chunks.append(
            f"[Source {index} | {source} | page {page} | chunk {chunk_index} | "
            f"type {chunk_type} | score {relevance_score} | rerank {rerank_score}]\n"
            f"Matched query terms: {matched_terms}\n"
            f"Text:\n{content}"
        )

    return "\n\n".join(chunks)


def collect_sources(documents: list[Document]) -> list[tuple[int, str, str, str]]:
    seen: set[tuple[str, str]] = set()
    sources: list[tuple[int, str, str, str]] = []

    for document in documents:
        source = str(document.metadata.get("source", "unknown source"))
        page = str(document.metadata.get("page", "unknown page"))
        rerank_score = str(document.metadata.get("rerank_score", "n/a"))
        key = (source, page)

        if key in seen:
            continue

        seen.add(key)

        try:
            page_number = int(page)
        except ValueError:
            page_number = sys.maxsize

        sources.append((page_number, source, page, rerank_score))

    return sorted(sources)


def format_sources(documents: list[Document]) -> str:
    sources = collect_sources(documents)

    if not sources:
        return ""

    lines = [
        f"- {source}, page {page}, rerank {rerank_score}"
        for _, source, page, rerank_score in sources
    ]
    return "Sources:\n" + "\n".join(lines)


def print_sources(documents: list[Document]) -> None:
    sources = collect_sources(documents)

    if not sources:
        return

    if console is None or Table is None:
        print(f"\n{format_sources(documents)}")
        return

    table = Table(title="Sources", show_header=True, header_style="bold cyan")
    table.add_column("Source")
    table.add_column("Page", justify="right")
    table.add_column("Rerank", justify="right")

    for _, source, page, rerank_score in sources:
        table.add_row(source, page, rerank_score)

    console.print(table)


def resolve_input_path(path_text: str, default_path: str) -> Path:
    if not path_text:
        path_text = default_path

    path = Path(path_text.strip().strip('"').strip("'")).expanduser()

    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return PROJECT_ROOT / path


def resolve_json_path(path_text: str) -> Path:
    return resolve_input_path(path_text, "data/questions.json")


def resolve_pdf_path(path_text: str) -> Path:
    return resolve_input_path(path_text, "data/testing.pdf")


def load_questions_from_json(file_path: Path) -> list[str]:
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of question objects or strings.")

    questions: list[str] = []

    for index, item in enumerate(data, start=1):
        if isinstance(item, dict):
            question = item.get("question")
        else:
            question = item

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Item #{index} must contain a non-empty question string.")

        questions.append(question.strip())

    return questions


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+(?:[.,][0-9]+)?", text.lower()))


def matched_context_terms(query: str, content: str) -> str:
    if not query:
        return "n/a"

    query_tokens = {
        token
        for token in tokenize(query)
        if len(token) >= 3 and token not in CONTEXT_MATCH_STOPWORDS
    }
    content_tokens = tokenize(content)
    matches = sorted(query_tokens & content_tokens)

    if not matches:
        return "none"

    return ", ".join(matches[:12])


def context_quality_note(query: str, documents: list[Document]) -> str:
    if not documents:
        return "Context quality: weak. No relevant context was retrieved."

    query_tokens = {token for token in tokenize(query) if len(token) >= 3}
    context_tokens = set()

    for document in documents:
        context_tokens.update(tokenize(document.page_content))

    overlap = len(query_tokens & context_tokens)
    top_relevance = max(float(document.metadata.get("relevance_score", 0.0)) for document in documents)
    top_rerank = max(float(document.metadata.get("rerank_score", 0.0)) for document in documents)

    if overlap < 2 and top_relevance < 0.25:
        return (
            "Context quality: weak. The retrieved chunks may not contain enough evidence. "
            "If the answer is not explicit, say it is not available in the retrieved context."
        )

    if is_numeric_or_rule_query(query):
        context_text = "\n".join(document.page_content.lower() for document in documents)
        has_rule_evidence = bool(
            re.search(r"\d|=|≤|>=|<|>|\b(table|formula|minimum|maximum|shall|requirement)\b", context_text)
        )
        if not has_rule_evidence:
            return (
                "Context quality: weak for a numeric or rule question. The retrieved chunks may mention the topic "
                "but not the requested value, formula, table, or condition. If the answer is not explicit, say it is not available."
            )

    return f"Context quality: usable. Top relevance score: {top_relevance:.4f}. Top rerank score: {top_rerank:.4f}."


def build_prompt(
    query: str,
    documents: list[Document],
    history: list[tuple[str, str]],
    mode: str,
    use_history: bool,
) -> str:
    response_language = get_response_language(query)
    if mode == LLAMA_MODE:
        answer_style = (
            "Answer in 1 short paragraph or up to 3 bullets. First identify the single retrieved source chunk that directly answers the question. "
            "Use other chunks only for required conditions or exceptions. For numeric or rule questions, state the exact requirement, value, formula, "
            "and applicable condition found in the context. For Indonesian questions, answer in Bahasa Indonesia but preserve BKI technical terms, "
            "formulas, units, table names, and symbols exactly as shown. If the retrieved chunks do not explicitly contain the requested value, "
            "formula, or condition, say it is not available in the retrieved BKI Hull Rules context."
        )
    elif mode == FAST_MODE:
        answer_style = "Answer clearly in 1-2 short paragraphs. This mode prioritizes speed, but the answer must still be complete enough to be useful."
    else:
        answer_style = "Answer clearly with enough technical context to justify the result."

    return f"""
You are a technical assistant for BKI Rules for Hull 2026.
Required answer language: {response_language}.
{answer_style}

Grounding rules that apply to every mode:
- Use only the retrieved context as evidence.
- If a requested value, formula, ratio, percentage, time, force, height, diameter, or rule condition is not explicit in the retrieved context, do not estimate, infer, or fabricate it.
- If context is incomplete or weak, say that the information is not available in the retrieved BKI Hull Rules context.
- Do not cite sections, pages, tables, formulas, or values that are not present in the retrieved context.
- For numeric, table, formula, or requirement questions, prefer exact wording and conditions from the retrieved context over paraphrase.
- If retrieved chunks appear to describe different cases and the question does not specify which case applies, state the ambiguity instead of choosing one.
- Conversation history may only clarify follow-up references, not provide factual evidence.

Answer format:
- Return only the final answer text.
- Do not use labels such as Jawaban, Answer, Bukti, Evidence, Catatan, or Notes.
- Do not quote long evidence blocks.
- If the answer is numeric, include the formula or short reasoning only when it is clearly supported by the retrieved context.

{context_quality_note(query, documents)}

Retrieved context:
{format_context(documents, query)}

Conversation history:
{format_history(history, get_history_limit(mode), use_history)}

User question:
{query}

Assistant answer:
""".strip()


def is_generic_non_answer(answer: str) -> bool:
    normalized_answer = answer.lower()
    return any(pattern in normalized_answer for pattern in GENERIC_NON_ANSWER_PATTERNS)


def build_strict_extraction_prompt(
    query: str,
    documents: list[Document],
    history: list[tuple[str, str]],
    mode: str,
    use_history: bool,
) -> str:
    response_language = get_response_language(query)
    return f"""
You are a technical assistant for BKI Rules for Hull 2026.
Required answer language: {response_language}.
The previous response was too generic. Extract the exact answer from the retrieved context.

Rules:
- Use only the retrieved context as evidence.
- Select the one source chunk that directly answers the question.
- Use other chunks only for conditions, exceptions, or scope.
- Preserve formulas, values, units, symbols, table names, and BKI technical terms exactly as shown.
- If the retrieved context does not explicitly contain the requested value, formula, or condition, say only that it is not available in the retrieved BKI Hull Rules context.
- Do not provide general ship-construction guidance.

{context_quality_note(query, documents)}

Retrieved context:
{format_context(documents, query)}

Conversation history:
{format_history(history, get_history_limit(mode), use_history)}

User question:
{query}

Assistant answer:
""".strip()


def generate_answer(
    query: str,
    history: list[tuple[str, str]],
    llm: Any,
    mode: str,
    use_history: bool,
) -> tuple[str, list[Document], dict[str, float]]:
    timings: dict[str, float] = {}
    config = get_mode_config(mode)
    retrieval_query = build_retrieval_query(query, history, use_history)

    retrieval_started_at = time.perf_counter()
    documents = retrieve_context(
        retrieval_query,
        candidate_k=int(config["candidate_k"]),
        final_k=int(config["final_k"]),
        min_score=config.get("min_score"),
    )
    timings["retrieval"] = time.perf_counter() - retrieval_started_at

    prompt_started_at = time.perf_counter()
    prompt = build_prompt(query, documents, history, mode, use_history)
    timings["prompt"] = time.perf_counter() - prompt_started_at

    generation_started_at = time.perf_counter()
    response = llm.invoke(prompt)
    timings["generation"] = time.perf_counter() - generation_started_at
    answer = getattr(response, "content", str(response)).strip()

    if mode == LLAMA_MODE and is_numeric_or_rule_query(query) and is_generic_non_answer(answer):
        retry_prompt = build_strict_extraction_prompt(query, documents, history, mode, use_history)
        retry_started_at = time.perf_counter()
        retry_response = llm.invoke(retry_prompt)
        retry_seconds = time.perf_counter() - retry_started_at
        retry_answer = getattr(retry_response, "content", str(retry_response)).strip()
        timings["generation"] += retry_seconds
        timings["retry_generation"] = retry_seconds
        if retry_answer:
            answer = retry_answer

    return answer, documents, timings


def answer_user_question(
    user_input: str,
    history: list[tuple[str, str]],
    llm_cache: dict[str, Any],
    mode: str,
    use_history: bool = True,
    append_history: bool = True,
) -> list[tuple[str, str]]:
    started_at = time.perf_counter()

    try:
        with show_loading():
            llm_started_at = time.perf_counter()
            llm = get_cached_llm(mode, llm_cache)
            llm_init_seconds = time.perf_counter() - llm_started_at
            answer, documents, timings = generate_answer(user_input, history, llm, mode, use_history)
    except Exception as exc:
        elapsed_seconds = time.perf_counter() - started_at
        print_error(f"Error after {elapsed_seconds:.2f} seconds: {exc}")
        return history

    elapsed_seconds = time.perf_counter() - started_at

    if console is None or Panel is None:
        print("\nAssistant:")
        print(answer)
    else:
        renderable = Markdown(answer) if Markdown is not None else answer
        console.print(Panel(renderable, title="Assistant", border_style="cyan"))

    timings["llm_init"] = llm_init_seconds
    timings["total"] = elapsed_seconds
    print_status(
        "Timing: "
        f"llm init {timings['llm_init']:.2f}s | "
        f"retrieval {timings['retrieval']:.2f}s | "
        f"prompt {timings['prompt']:.2f}s | "
        f"generation {timings['generation']:.2f}s | "
        f"total {timings['total']:.2f}s",
        "dim",
    )
    print_sources(documents)

    if not append_history:
        return history

    history.append((user_input, answer))
    return history[-get_history_limit(mode):]


def ask_independent_questions(
    questions: list[str],
    history: list[tuple[str, str]],
    llm_cache: dict[str, Any],
    mode: str,
) -> None:
    for index, question in enumerate(questions, start=1):
        print_rule(f"Question {index}/{len(questions)}")
        print_status(f"You: {question}", "bold")
        answer_user_question(
            question,
            history,
            llm_cache,
            mode,
            use_history=False,
            append_history=False,
        )


def import_json_questions(
    path_text: str,
    history: list[tuple[str, str]],
    llm_cache: dict[str, Any],
    mode: str,
) -> list[tuple[str, str]]:
    try:
        file_path = resolve_json_path(path_text)
        questions = load_questions_from_json(file_path)
    except Exception as exc:
        print_error(f"Import JSON failed: {exc}")
        return history

    if not questions:
        print_status("JSON file contains no questions.", "yellow")
        return history

    print_status(f"Importing {len(questions)} independent questions from {file_path}", "cyan")
    ask_independent_questions(questions, history, llm_cache, mode)
    print_status(f"\nFinished importing {len(questions)} questions. Interactive history was not changed.", "green")
    return history


def import_pdf_questions(
    path_text: str,
    history: list[tuple[str, str]],
    llm_cache: dict[str, Any],
    mode: str,
) -> list[tuple[str, str]]:
    try:
        file_path = resolve_pdf_path(path_text)
        questions = load_questions_from_pdf(file_path)
    except Exception as exc:
        print_error(f"Import PDF failed: {exc}")
        return history

    if not questions:
        print_status("PDF file contains no detected questions.", "yellow")
        return history

    print_status(f"Importing {len(questions)} independent questions from {file_path}", "cyan")
    print_status("PDF testing is used only as a question source; answers still use the BKI vector store.", "dim")
    ask_independent_questions(questions, history, llm_cache, mode)
    print_status(f"\nFinished importing {len(questions)} questions. Interactive history was not changed.", "green")
    return history


def print_retrieval_debug(question: str, mode: str) -> None:
    if not question.strip():
        print_status("Usage: /debug-retrieve <question>", "yellow")
        return

    config = get_mode_config(mode)
    retrieval_query = build_retrieval_query(question, [], False)
    retrieval_started_at = time.perf_counter()
    candidates = retrieve_candidates(retrieval_query, candidate_k=int(config["candidate_k"]))
    documents = rerank_documents(retrieval_query, candidates, final_k=int(config["final_k"]))
    min_score = config.get("min_score")
    if min_score is not None:
        documents = [
            document
            for document in documents
            if float(document.metadata.get("rerank_score", 0.0)) >= min_score
        ]
    retrieval_seconds = time.perf_counter() - retrieval_started_at
    numeric_query = is_numeric_or_rule_query(question)

    if not documents:
        print_status("No retrieved chunks.", "yellow")
        return

    debug_body = "\n".join(
        [
            f"Mode: {mode}",
            f"Query: {question}",
            f"Retrieval query: {retrieval_query}",
            f"Numeric/rule query: {numeric_query}",
            f"Candidates: {len(candidates)}",
            f"Final chunks: {len(documents)}",
            f"Retrieval + rerank: {retrieval_seconds:.2f}s",
        ]
    )
    print_panel("Retrieval debug", debug_body, "magenta")

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown source")
        page = document.metadata.get("page", "unknown page")
        chunk_index = document.metadata.get("chunk_index", "unknown chunk")
        chunk_type = document.metadata.get("chunk_type", "unknown type")
        chunk_length = document.metadata.get("chunk_length", "unknown length")
        relevance_score = document.metadata.get("relevance_score", "n/a")
        rerank_score = document.metadata.get("rerank_score", "n/a")
        preview = re.sub(r"\s+", " ", document.page_content.strip())[:900]
        body = "\n".join(
            [
                f"source: {source}",
                f"page: {page}",
                f"chunk: {chunk_index}",
                f"chunk_type: {chunk_type}",
                f"chunk_length: {chunk_length}",
                f"score: {relevance_score}",
                f"rerank: {rerank_score}",
                "",
                preview,
            ]
        )
        print_panel(f"Chunk {index}", body, "dim")


def chat_loop() -> int:
    print_welcome()
    sys.stdout.flush()

    llm_cache: dict[str, Any] = {}
    mode = NORMAL_MODE
    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = prompt_user(mode)
        except (EOFError, KeyboardInterrupt):
            print_status("\nGoodbye.", "cyan")
            return 0

        if not user_input:
            continue

        command, args = parse_command(user_input)

        if not args and command in {"/quit", "/exit"}:
            print_status("Goodbye.", "cyan")
            return 0

        if not args and command == "/help":
            print_help()
            continue

        if not args and command == "/clear":
            history.clear()
            clear_terminal()
            print_welcome()
            print_status("Conversation history and screen cleared.", "green")
            continue

        if not args and command == "/fast":
            mode = FAST_MODE
            history = history[-get_history_limit(mode):]
            print_status(f"Fast mode enabled: {MODE_CONFIGS[FAST_MODE]['description']}.", "green")
            continue

        if not args and command == "/llama":
            mode = LLAMA_MODE
            history = history[-get_history_limit(mode):]
            print_status(f"Llama mode enabled: {MODE_CONFIGS[LLAMA_MODE]['description']}.", "magenta")
            continue

        if not args and command == "/normal":
            mode = NORMAL_MODE
            print_status(f"Normal mode enabled: {MODE_CONFIGS[NORMAL_MODE]['description']}.", "cyan")
            continue

        if command == "/debug-retrieve":
            try:
                print_retrieval_debug(args, mode)
            except Exception as exc:
                print_error(f"Debug retrieval failed: {exc}")
            continue

        if command == "/import-json":
            history = import_json_questions(args, history, llm_cache, mode)
            continue

        if command == "/import-pdf":
            history = import_pdf_questions(args, history, llm_cache, mode)
            continue

        history = answer_user_question(user_input, history, llm_cache, mode)


if __name__ == "__main__":
    raise SystemExit(chat_loop())
