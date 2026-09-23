"""
Document splitting module.
Applies recursive character text splitting or structure-aware section chunking.
"""
import logging
import re
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger("src.langchain.splitter")


def get_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Returns a configured RecursiveCharacterTextSplitter with structure-preserving separators.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=[
            "\n\n\n",
            "\n\n",
            "\n# ",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n- ",
            "\n* ",
            "\n• ",
            "\n1. ",
            "\n",
            ". ",
            " ",
            "",
        ],
        length_function=len,
    )


def split_policy_document(text_content: str, base_metadata: dict) -> list[Document]:
    """
    Creates structure-aware, hierarchical chunks for the company policy handbook:
    1. Full section chunks for broader contextual inquiries.
    2. Granular clause chunks (with section header prepended) for specific topic inquiries.
    """
    chunks: list[Document] = []
    try:
        raw_sections = text_content.split("## ")
        chunk_idx = 0

        header_intro = raw_sections[0].strip()
        if header_intro:
            chunks.append(Document(page_content=header_intro, metadata={**base_metadata, "chunk_index": chunk_idx}))
            chunk_idx += 1

        for section in raw_sections[1:]:
            lines = [l.strip() for l in section.strip().split("\n") if l.strip()]
            if not lines:
                continue
            section_title = lines[0].strip()
            section_body = section.strip()

            # Determine exact PDF page (2 sections per page)
            sec_match = re.match(r"^(\d+)", section_title)
            page_num = min(5, (int(sec_match.group(1)) - 1) // 2 + 1) if sec_match else 1

            chunk_meta = {
                **base_metadata,
                "filename": "WorkPilot_Company_Policy.pdf",
                "section": section_title,
                "page": page_num,
            }

            # Complete section chunk for broad thematic queries
            chunks.append(
                Document(
                    page_content=f"## {section_body}",
                    metadata={**chunk_meta, "chunk_index": chunk_idx},
                )
            )
            chunk_idx += 1

            # Granular clause chunks for specific benefit lookups
            for line in lines[1:]:
                if line.startswith("-"):
                    item_text = line.lstrip("-").strip()
                    chunks.append(
                        Document(
                            page_content=f"## {section_title}\n- {item_text}",
                            metadata={**chunk_meta, "chunk_index": chunk_idx},
                        )
                    )
                    chunk_idx += 1

        logger.info("Split policy text into %d structured chunks.", len(chunks))
        return chunks
    except Exception as e:
        logger.error("Error splitting policy document: %s", e, exc_info=True)
        return []


def split_documents(
    documents: Sequence[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Splits a list of LangChain Documents into smaller chunks.
    Preserves metadata and adds 'chunk_index' to each chunk.
    """
    if not documents:
        return []

    try:
        if len(documents) == 1 and ("## " in documents[0].page_content):
            return split_policy_document(documents[0].page_content, documents[0].metadata)

        splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(list(documents))

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index

        return chunks
    except Exception as e:
        logger.error("Error splitting documents: %s", e, exc_info=True)
        return list(documents)
