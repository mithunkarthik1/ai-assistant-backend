import re
from typing import Sequence
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def get_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Return a configured RecursiveCharacterTextSplitter instance with structure-preserving separators."""
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
    2. Granular rule/clause chunks (with section header prepended) for specific topic inquiries
       (e.g., gym, yoga classes, sports subscriptions, dental, vision, internet speed).
    """
    chunks: list[Document] = []
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
        if sec_match:
            sec_num = int(sec_match.group(1))
            page_num = min(5, (sec_num - 1) // 2 + 1)
        else:
            page_num = 1

        chunk_meta = {
            **base_metadata,
            "filename": "WorkPilot_Company_Policy.pdf",
            "section": section_title,
            "page": page_num,
        }

        # 1. Complete section chunk for broad thematic queries
        chunks.append(Document(
            page_content=f"## {section_body}",
            metadata={**chunk_meta, "chunk_index": chunk_idx}
        ))
        chunk_idx += 1

        # 2. Granular clause chunks for specific benefit / rule lookups
        for line in lines[1:]:
            if line.startswith("-"):
                item_text = line.lstrip("-").strip()
                chunks.append(Document(
                    page_content=f"## {section_title}\n- {item_text}",
                    metadata={**chunk_meta, "chunk_index": chunk_idx}
                ))
                chunk_idx += 1

    return chunks


def split_documents(
    documents: Sequence[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split a list of LangChain Documents into smaller chunks.
    Uses structure-aware granular chunking for company policy markdown,
    preserving metadata and adding 'chunk_index' to each chunk.
    """
    if not documents:
        return []

    # If this is the company policy handbook containing markdown sections
    if len(documents) == 1 and ("## " in documents[0].page_content):
        return split_policy_document(documents[0].page_content, documents[0].metadata)

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(list(documents))

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks
