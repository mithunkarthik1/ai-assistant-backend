from pathlib import Path
from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat_bot.schema import ChatRequest, ChatResponse
from app.chat_bot.service import ChatService
from app.database import get_db
from app.rag.pdf_generator import POLICY_PAGES, generate_company_policy_pdf

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
    service = ChatService(db)
    return await service.chat(request)


@router.get(
    "/history",
    summary="Get recent conversation history",
)
@router.get(
    "/{document_id}/history",
    summary="Get recent conversation history (legacy compatibility)",
    include_in_schema=False,
)
async def get_chat_history(
    document_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    service = ChatService(db)
    return await service.get_history()


@router.get(
    "/pdf",
    summary="Download or view the official WorkPilot Company Policy Handbook (PDF)",
)
async def get_policy_pdf():
    pdf_path = Path(__file__).resolve().parent.parent.parent / "data" / "WorkPilot_Company_Policy.pdf"
    if not pdf_path.exists():
        generate_company_policy_pdf(pdf_path)

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Policy PDF document not found.")

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


@router.get(
    "/policy-pages",
    summary="Get structured company policy pages for interactive viewer and search",
)
async def get_policy_pages():
    return {"pages": POLICY_PAGES}
