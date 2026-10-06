import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import get_db
from src.rag.schema import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    DocumentDetailResponse,
    DocumentInfoResponse,
    DocumentUploadResponse,
)
from src.rag.service import (
    ChatService,
    ExtractionError,
    POLICY_DOC_ID,
    POLICY_PAGES,
    delete_document,
    generate_company_policy_pdf,
    get_document_detail,
    get_documents,
    get_policy_file_path,
    process_document_upload,
)

logger = logging.getLogger("src.rag.api")
router = APIRouter(prefix="/chat", tags=["chat"])
documents_router = APIRouter(prefix="/documents", tags=["documents"])
rag_router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about company policies",
)
@router.post(
    "/",
    response_model=ChatResponse,
    include_in_schema=False,
)
async def chat_with_policy(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Submits a user question to the RAG pipeline and returns the grounded LLM answer
    along with source document citations.
    """
    logger.info("Incoming chat request: '%s' (session_id=%s)", request.message, request.session_id)
    try:
        service = ChatService(db)
        response = await service.chat(request)
        logger.info(
            "Chat response generated successfully (sources=%d, show_pdf=%s)",
            len(response.sources),
            response.show_pdf,
        )
        return response
    except Exception as e:
        logger.error("Unhandled exception in chat_with_policy endpoint: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat query: {str(e)}",
        ) from e


@rag_router.post(
    "/query",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute RAG query with Qdrant vector retrieval and LLM generation",
)
async def query_rag_endpoint(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Submits user question to RAG pipeline via Qdrant similarity search and LLM synthesis.
    """
    return await chat_with_policy(request, db)


@router.get(
    "/history",
    response_model=list[ChatMessageResponse],
    summary="Get recent conversation history",
)
@router.get(
    "/{document_id}/history",
    response_model=list[ChatMessageResponse],
    summary="Get recent conversation history (legacy compatibility)",
    include_in_schema=False,
)
async def get_chat_history(
    document_id: str | None = None,
    session_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[ChatMessageResponse]:
    """
    Retrieves chronological message history for the active conversation.
    """
    try:
        service = ChatService(db)
        doc_uuid = None
        if document_id:
            try:
                doc_uuid = uuid.UUID(document_id)
            except Exception:
                pass
        history = await service.get_history(document_id=doc_uuid, session_id=session_id)
        logger.debug("Retrieved %d history items", len(history))
        return history
    except Exception as e:
        logger.error("Failed to fetch chat history: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history.",
        ) from e


@router.get(
    "/pdf",
    summary="Download or view the official WorkPilot Company Policy Handbook (PDF)",
)
async def get_policy_pdf() -> FileResponse:
    """
    Serves the official company policy PDF document for inline browser viewing.
    """
    try:
        pdf_path = get_policy_file_path()
        if not pdf_path.exists():
            logger.info("PDF handbook not found on disk; generating fresh copy...")
            generate_company_policy_pdf(pdf_path)

        if not pdf_path.exists():
            logger.error("Policy PDF could not be found or generated at: %s", pdf_path)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Policy PDF document not found.",
            )

        fname = pdf_path.name
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename=fname,
            content_disposition_type="inline",
            headers={
                "Content-Disposition": f'inline; filename="{fname}"',
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error serving policy PDF: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while accessing the policy PDF.",
        ) from e


@router.get(
    "/policy-pages",
    summary="Get structured company policy pages for interactive viewer and search",
)
async def get_policy_pages() -> dict[str, Any]:
    """
    Returns structured page-by-page JSON policy data for client-side handbook viewer.
    """
    return {"pages": POLICY_PAGES}


# ============================================================
# DOCUMENT MANAGEMENT & INCREMENTAL UPLOAD ROUTERS
# ============================================================

@documents_router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload PDF, DOCX, or TXT for incremental RAG indexing",
)
@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload PDF, DOCX, or TXT for incremental RAG indexing (chat alias)",
    include_in_schema=False,
)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    """
    Uploads a document (PDF, DOCX, TXT), performs structure-aware hybrid chunking,
    computes deterministic chunk hashes, and incrementally embeds/indexes only modified chunks.
    """
    filename = file.filename or "uploaded_document.pdf"
    logger.info("Received document upload request: '%s' (content_type=%s)", filename, file.content_type)

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{filename}' is empty.",
            )

        upload_result = await process_document_upload(
            file_bytes=file_bytes,
            file_name=filename,
            db=db,
        )
        logger.info(
            "Document '%s' processed successfully: added=%d, updated=%d, skipped=%d, deleted=%d",
            filename,
            upload_result.chunks_added,
            upload_result.chunks_updated,
            upload_result.chunks_skipped,
            upload_result.chunks_deleted,
        )
        return upload_result
    except ExtractionError as e:
        logger.warning("Extraction error for '%s': %s", filename, e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process document upload for '%s': %s", filename, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while indexing document '{filename}': {str(e)}",
        ) from e


@documents_router.get(
    "",
    response_model=list[DocumentInfoResponse],
    summary="List all registered knowledge base documents",
)
async def list_documents(
    db: AsyncSession = Depends(get_db),
) -> list[DocumentInfoResponse]:
    """
    Returns list of all documents registered in the system along with their current status and chunk counts.
    """
    try:
        docs = await get_documents(db)
        return [
            DocumentInfoResponse(
                document_id=d.document_id,
                file_name=d.file_name,
                file_hash=d.file_hash,
                file_type=d.file_type,
                status=d.status,
                chunk_count=d.chunk_count,
                is_default=(d.document_id == POLICY_DOC_ID),
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            for d in docs
        ]
    except Exception as e:
        logger.error("Failed to list documents: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document registry.",
        ) from e


@documents_router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details and indexed chunk structure",
)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    """
    Retrieves full details for a document including its hybrid chunk hierarchy and content hashes.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document UUID format.") from e

    try:
        doc, chunks = await get_document_detail(doc_uuid, db)
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

        chunk_data = [
            {
                "chunk_id": c.chunk_id,
                "section": c.section,
                "topic": c.topic,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "content_hash": c.content_hash,
                "content_preview": c.content[:160] + "..." if len(c.content) > 160 else c.content,
            }
            for c in chunks
        ]
        doc_info = DocumentInfoResponse(
            document_id=doc.document_id,
            file_name=doc.file_name,
            file_hash=doc.file_hash,
            file_type=doc.file_type,
            status=doc.status,
            chunk_count=doc.chunk_count,
            is_default=(doc.document_id == POLICY_DOC_ID),
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )
        return DocumentDetailResponse(document=doc_info, chunks=chunk_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document detail '%s': %s", document_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document details.",
        ) from e


@documents_router.delete(
    "/{document_id}",
    summary="Delete a document and purge its vectors from index",
)
async def delete_doc(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Deletes a document from registry and purges all corresponding vectors from vector store.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document UUID format.") from e

    if doc_uuid == POLICY_DOC_ID:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The default company policy document is protected and cannot be deleted.",
        )

    try:
        deleted = await delete_document(doc_uuid, db)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")
        return {"deleted": True, "document_id": document_id, "message": f"Document '{document_id}' and all vectors deleted."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document '%s': %s", document_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document.",
        ) from e

