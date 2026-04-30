from langchain_ollama import ChatOllama, OllamaEmbeddings

DEFAULT_LLM_MODEL = "qwen2.5:7b"
FALLBACK_LLM_MODEL = "qwen2.5:3b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_TEMPERATURE = 0.0


def ollama_setup_hint() -> str:
    return (
        "Ensure Ollama is running and the required models are available:\n"
        "  ollama pull qwen2.5:7b\n"
        "  ollama pull qwen2.5:3b\n"
        "  ollama pull nomic-embed-text"
    )


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model_name)


def get_llm(
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    **model_kwargs,
) -> ChatOllama:
    return ChatOllama(model=model_name, temperature=temperature, **model_kwargs)


def get_llm_with_fallback(
    preferred_model: str = DEFAULT_LLM_MODEL,
    fallback_model: str = FALLBACK_LLM_MODEL,
    **model_kwargs,
) -> ChatOllama:
    errors: list[str] = []
    model_names = [preferred_model]

    if fallback_model != preferred_model:
        model_names.append(fallback_model)

    for model_name in model_names:
        llm = get_llm(model_name=model_name, **model_kwargs)
        try:
            llm.invoke("Reply with OK only.")
            return llm
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    raise RuntimeError(
        "Unable to initialize an Ollama chat model.\n"
        + ollama_setup_hint()
        + "\nErrors:\n  "
        + "\n  ".join(errors)
    )
