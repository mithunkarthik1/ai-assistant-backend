"""
FastAPI route handlers for chat interactions and company policy resources.
Includes comprehensive logging, request timing, and robust error handling.
"""
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db
from src.rag.schema import ChatMessageResponse, ChatRequest, ChatResponse
from src.rag.service import ChatService, POLICY_PAGES, generate_company_policy_pdf

logger = logging.getLogger("src.rag.api")
router = APIRouter(prefix="/chat", tags=["chat"])


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
    db: AsyncSession = Depends(get_db),
) -> list[ChatMessageResponse]:
    """
    Retrieves chronological message history for the active conversation.
    """
    try:
        service = ChatService(db)
        history = await service.get_history()
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
        pdf_path = Path(__file__).resolve().parent.parent.parent / "data" / "WorkPilot_Company_Policy.pdf"
        if not pdf_path.exists():
            logger.info("PDF handbook not found on disk; generating fresh copy...")
            generate_company_policy_pdf(pdf_path)

        if not pdf_path.exists():
            logger.error("Policy PDF could not be found or generated at: %s", pdf_path)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Policy PDF document not found.",
            )

        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename="WorkPilot_Company_Policy.pdf",
            content_disposition_type="inline",
            headers={
                "Content-Disposition": 'inline; filename="WorkPilot_Company_Policy.pdf"',
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
