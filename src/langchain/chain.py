"""
LangChain RAG Chain execution module.
Coordinates context retrieval, prompt formatting, and LLM answer generation.
"""
import logging
import uuid
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.langchain.indexing import POLICY_DOC_ID, POLICY_FILENAME
from src.langchain.llm import get_llm
from src.langchain.prompts import get_general_prompt_template, get_rag_prompt_template
from src.langchain.retriever import retrieve_relevant_chunks

logger = logging.getLogger("src.langchain.chain")


class RagResult(tuple):
    """
    Subclass of tuple (answer, docs) for backwards compatibility
    with callers expecting `answer, docs = await generate_rag_answer(...)`,
    while exposing the `show_pdf: bool` property for frontend UI integration.
    """
    show_pdf: bool

    def __new__(cls, answer: str, docs: list[Document], show_pdf: bool = False):
        instance = super().__new__(cls, (answer, docs))
        instance.show_pdf = show_pdf
        return instance


def format_context(documents: Sequence[Document]) -> str:
    """Formats retrieved semantic Document chunks into structured context blocks for the LLM."""
    if not documents:
        return "No specific policy documents retrieved."

    formatted_chunks = []
    for idx, doc in enumerate(documents):
        filename = doc.metadata.get("filename", POLICY_FILENAME)
        page = doc.metadata.get("page", 1)
        score = doc.metadata.get("score", "N/A")
        formatted_chunks.append(
            f"--- [Document: {filename} | Page: {page} | Similarity: {score}] ---\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_chunks)


def format_chat_history(chat_history: Sequence[Any] | None) -> str:
    """Formats recent conversation turns into a clear dialogue history for the LLM."""
    if not chat_history:
        return "No prior conversation."

    formatted = []
    for msg in chat_history[-6:]:
        role = (
            getattr(msg, "role", None)
            or getattr(msg, "type", None)
            or (msg.get("role") if isinstance(msg, dict) else "")
            or "user"
        )
        content = (
            getattr(msg, "content", "")
            if not isinstance(msg, dict)
            else msg.get("content", "")
        )
        formatted.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted)


def should_suggest_handbook(answer: str) -> bool:
    """Checks whether the generated answer suggests checking the full handbook PDF."""
    ans_lower = answer.lower()
    return any(p in ans_lower for p in [
        "couldn't find",
        "could not find",
        "not specified",
        "not mentioned",
        "not documented",
        "outside our documented",
        "people operations",
        "policy handbook",
    ])


async def generate_rag_answer(
    question: str,
    document_id: str | uuid.UUID | None = None,
    top_k: int | None = None,
    llm: BaseChatModel | None = None,
    allow_fallback: bool = True,
    chat_history: Sequence[Any] | None = None,
) -> RagResult:
    """
    Executes a pure, LLM-driven RAG pipeline:
    1. Retrieves relevant policy chunks from vector store using dense embeddings + contextual search.
    2. Builds prompt with retrieved context, conversation history, and user question.
    3. Invokes the LLM to synthesize an accurate, grounded, and concise answer.
    4. Routes out-of-scope or ungrounded questions through the general LLM fallback chain.
    """
    target_doc_id = document_id or POLICY_DOC_ID
    effective_top_k = top_k if top_k is not None else settings.top_k
    output_parser = StrOutputParser()

    # 1. Retrieve semantically relevant policy chunks from vector store
    try:
        chunks = await retrieve_relevant_chunks(
            document_id=target_doc_id,
            query=question,
            top_k=effective_top_k,
            chat_history=chat_history,
        )
    except Exception as e:
        logger.error("Error during chunk retrieval: %s", e, exc_info=True)
        chunks = []

    logger.info(
        "Semantic Search for '%s' retrieved %d chunks (scores: %s)",
        question,
        len(chunks),
        [doc.metadata.get("score") for doc in chunks],
    )

    # 2. Acquire active LLM instance
    active_llm = llm or get_llm()
    if active_llm is None:
        msg = (
            "⚠️ **No LLM is configured.** Please set a valid `LLM_API_KEY` (e.g. Groq, OpenAI, xAI Grok, or local Ollama) "
            "in your backend `.env` file to enable AI answers."
        )
        return RagResult(msg, chunks, show_pdf=True)

    # 3. If relevant chunks were retrieved, invoke the Grounded RAG chain
    if chunks:
        context_str = format_context(chunks)
        has_history = bool(chat_history)
        history_str = format_chat_history(chat_history)

        prompt_template = get_rag_prompt_template(has_history=has_history)
        chain = prompt_template | active_llm | output_parser

        payload: dict[str, Any] = {
            "context": context_str,
            "question": question,
        }
        if has_history:
            payload["chat_history"] = history_str

        try:
            logger.info("Invoking LLM for grounded RAG answer...")
            answer = await chain.ainvoke(payload)
            clean_answer = str(answer).strip()
            show_pdf = should_suggest_handbook(clean_answer)
            return RagResult(clean_answer, chunks, show_pdf=show_pdf)
        except Exception as e:
            logger.error("LLM RAG invocation failed: %s", e)
            error_msg = str(e)
            if "403" in error_msg or "credits" in error_msg.lower():
                return RagResult(
                    "⚠️ **LLM Quota Exceeded**: The configured LLM provider returned a permission/credit error. "
                    "Please check your API key credits or switch to a free provider like Groq in your `.env`.",
                    chunks,
                    show_pdf=True,
                )
            return RagResult(
                f"⚠️ Error generating answer from LLM: {error_msg}. Please check your backend logs.",
                chunks,
                show_pdf=True,
            )

    # 4. If no relevant chunks found and fallback is enabled, use general fallback prompt
    if allow_fallback:
        try:
            logger.info("No matching policy chunks found. Invoking fallback LLM...")
            fallback_prompt = get_general_prompt_template()
            fallback_chain = fallback_prompt | active_llm | output_parser
            answer = await fallback_chain.ainvoke({"question": question})
            return RagResult(str(answer).strip(), [], show_pdf=True)
        except Exception as e:
            logger.error("Fallback LLM invocation failed: %s", e)
            return RagResult(
                "I couldn't find information regarding that in the official company policy documents. "
                "Please consult People Operations or refer to the full policy handbook.",
                [],
                show_pdf=True,
            )

    return RagResult(
        "No matching policy information was found in the handbook.",
        [],
        show_pdf=True,
    )
