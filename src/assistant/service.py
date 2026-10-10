"""Unified orchestration between the existing RAG service and agent service."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.agents.schema import AgentRequest
from src.agents.service import AgentService, ConversationStore, create_agent_service
from src.assistant.router import classify_request
from src.assistant.schema import AssistantRequest, AssistantResponse
from src.core.config import AgentSettings
from src.rag.schema import ChatRequest

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("src.assistant.service")


class UnifiedAssistantService:
    """Routes a single conversation to policy RAG, the project agent, or direct LLM."""

    def __init__(
        self,
        agent_settings: AgentSettings,
        store: ConversationStore | None = None,
        agent_service: AgentService | None = None,
        rag_service_factory: Callable[[AsyncSession], Any] | None = None,
    ) -> None:
        self.agent_settings = agent_settings
        self.store = store or ConversationStore(max_messages=agent_settings.memory_max_messages)
        self._agent_service = agent_service
        self.rag_service_factory = rag_service_factory
        self._is_mock_rag = rag_service_factory is not None

    def _get_agent_service(self) -> AgentService:
        if self._agent_service is None:
            self._agent_service = create_agent_service(self.agent_settings, store=self.store)
        return self._agent_service

    def _get_rag_factory(self) -> Callable[[AsyncSession], Any]:
        if self.rag_service_factory is not None:
            return self.rag_service_factory
        from src.rag.service import ChatService
        return ChatService

    async def chat(
        self,
        request: AssistantRequest,
        db: AsyncSession,
    ) -> AssistantResponse:
        session_id = request.session_id or str(uuid.uuid4())
        session = self.store.get(session_id)
        rag_factory = self._get_rag_factory()

        # Pre-populate in-memory session history from database if store is fresh or restarted
        if not session.messages:
            try:
                from src.rag.service import ChatService
                db_msgs = await ChatService(db).get_history(session_id=session_id, limit=20)
                if db_msgs:
                    for dm in db_msgs:
                        session.messages.append({"role": dm.role, "content": dm.content})
            except Exception as e:
                logger.debug("Could not pre-load session history from DB: %s", e)

        # Conversational greeting check: dynamically synthesize greeting via LLM with zero document citations
        from src.rag.service import is_greeting
        if is_greeting(request.message):
            agent_response = await self._get_agent_service().chat(
                AgentRequest(
                    message=request.message,
                    session_id=session_id,
                    project_id=request.project_id or session.current_project_id,
                    history=session.messages,
                    preferred_route="direct",
                )
            )
            self.store.append(session_id, request.message, agent_response.answer, session.current_project_id)
            try:
                chat_svc = rag_factory(db)
                await chat_svc.add_message(role="user", content=request.message, session_id=session_id)
                await chat_svc.add_message(role="assistant", content=agent_response.answer, session_id=session_id)
            except Exception:
                pass
            return AssistantResponse(
                answer=agent_response.answer,
                route="direct",
                session_id=session_id,
                sources=[],
                show_pdf=False,
            )

        # Check if the user is answering a prior document disambiguation prompt (e.g. 'both', 'both pdf', 'all', or doc selection)
        from src.rag.service import resolve_clarification_from_history, is_comparison_query
        resolved_message, is_both_selected = resolve_clarification_from_history(request.message, session.messages)

        # Check if the user query specifically targets any registered document in the knowledge base
        target_doc_id = None
        try:
            from src.rag.service import detect_target_document_from_query
            target_doc_id = await detect_target_document_from_query(request.message, db)
        except Exception:
            pass

        if is_both_selected:
            logger.info("Unified assistant request session_id=%s selected BOTH documents for '%s'", session_id, resolved_message)
            response = await rag_factory(db).chat(
                ChatRequest(
                    message=resolved_message,
                    session_id=session_id,
                    history=session.messages,
                )
            )
            self.store.append(session_id, request.message, response.answer, session.current_project_id)
            return AssistantResponse(
                answer=response.answer,
                route="policy",
                session_id=session_id,
                sources=response.sources,
                show_pdf=response.show_pdf,
            )

        if target_doc_id:
            actual_message = resolved_message if resolved_message != request.message else request.message
            logger.info(
                "Unified assistant request session_id=%s targets document %s for message: '%s'",
                session_id,
                target_doc_id,
                actual_message,
            )
            response = await rag_factory(db).chat(
                ChatRequest(
                    message=actual_message,
                    session_id=session_id,
                    document_id=str(target_doc_id),
                    history=session.messages,
                )
            )
            self.store.append(session_id, request.message, response.answer, session.current_project_id)
            return AssistantResponse(
                answer=response.answer,
                route="policy",
                session_id=session_id,
                sources=response.sources,
                show_pdf=response.show_pdf,
            )

        try:
            from src.rag.service import get_documents
            docs = await get_documents(db)
            known_doc_names = [d.file_name for d in docs if d.file_name]
        except Exception:
            known_doc_names = []

        decision = None
        if not self._is_mock_rag:
            try:
                from src.assistant.router import classify_request_with_model
                from src.rag.service import get_llm
                llm = get_llm()
                if llm:
                    decision = await classify_request_with_model(
                        message=request.message,
                        current_project_id=session.current_project_id,
                        explicit_project_id=request.project_id,
                        known_documents=known_doc_names,
                        llm=llm,
                        history=session.messages,
                    )
            except Exception as e:
                logger.warning("Model-based classification failed: %s; falling back to deterministic", e)

        if not decision:
            decision = classify_request(
                request.message,
                current_project_id=session.current_project_id,
                explicit_project_id=request.project_id,
                known_documents=known_doc_names,
                history=session.messages,
            )

        logger.info(
            "Unified assistant request session_id=%s route=%s message=%s",
            session_id,
            decision.route,
            request.message,
        )

        if decision.route == "clarification":
            answer = "Which project ID should I use for that request?"
            self.store.append(session_id, request.message, answer, session.current_project_id)
            try:
                chat_svc = rag_factory(db)
                await chat_svc.add_message(role="user", content=request.message, session_id=session_id)
                await chat_svc.add_message(role="assistant", content=answer, session_id=session_id)
            except Exception:
                pass
            return AssistantResponse(
                answer=answer,
                route="clarification",
                session_id=session_id,
            )

        # Check for matching chunks across the knowledge base
        matching_chunks = []
        if not self._is_mock_rag and not is_greeting(request.message):
            try:
                from src.rag.service import retrieve_relevant_chunks
                matching_chunks = await retrieve_relevant_chunks(
                    query=request.message,
                    chat_history=session.messages,
                )
            except Exception as e:
                logger.warning("Knowledge base retrieval check skipped: %s", e)

        # Multi-document disambiguation check:
        # Only ask when genuinely competing documents are found within margin and threshold,
        # AND user has not selected 'both' and query is not a comparison query
        if matching_chunks and not is_both_selected and not is_comparison_query(request.message):
            from src.rag.service import get_competing_documents, format_clarification_message
            competing = get_competing_documents(matching_chunks, request.message)
            if competing:
                clarification_answer = format_clarification_message(competing)
                self.store.append(session_id, request.message, clarification_answer, session.current_project_id)
                try:
                    chat_svc = rag_factory(db)
                    await chat_svc.add_message(role="user", content=request.message, session_id=session_id)
                    await chat_svc.add_message(role="assistant", content=clarification_answer, session_id=session_id)
                except Exception:
                    pass
                return AssistantResponse(
                    answer=clarification_answer,
                    route="clarification",
                    session_id=session_id,
                    sources=[],
                    show_pdf=False,
                )

        if decision.route == "policy":
            response = await rag_factory(db).chat(
                ChatRequest(
                    message=request.message,
                    session_id=session_id,
                    history=session.messages,
                )
            )
            self.store.append(session_id, request.message, response.answer, session.current_project_id)
            return AssistantResponse(
                answer=response.answer,
                route="policy",
                session_id=session_id,
                sources=response.sources,
                show_pdf=response.show_pdf,
            )

        top_chunk_score = 0.0
        if matching_chunks:
            top_chunk_score = max(float(getattr(c, "metadata", {}).get("score", 0.0)) for c in matching_chunks)
        from src.assistant.router import _GENERAL_KNOWLEDGE_TERMS
        import re
        has_general_coding = any(
            re.search(r"\b" + re.escape(t) + r"\b", request.message.lower()) for t in _GENERAL_KNOWLEDGE_TERMS
        )
        is_general_def = bool(
            re.match(r"^what\s+(?:is|are|does\s+\w+\s+mean)\s+(?:an?\s+)?([a-zA-Z0-9_\s]+?)\??$", request.message.lower().strip())
        )
        if decision.route == "direct" and matching_chunks and top_chunk_score >= 0.72 and not has_general_coding and not is_general_def:
            response = await rag_factory(db).chat(
                ChatRequest(
                    message=request.message,
                    session_id=session_id,
                    history=session.messages,
                )
            )
            self.store.append(session_id, request.message, response.answer, session.current_project_id)
            return AssistantResponse(
                answer=response.answer,
                route="policy",
                session_id=session_id,
                sources=response.sources,
                show_pdf=response.show_pdf,
            )

        agent_response = await self._get_agent_service().chat(
            AgentRequest(
                message=request.message,
                session_id=session_id,
                project_id=request.project_id or decision.project_id,
                history=session.messages,
                preferred_route="project_api" if decision.route == "project" else "direct",
            )
        )
        self.store.append(session_id, request.message, agent_response.answer, session.current_project_id)
        try:
            chat_svc = rag_factory(db)
            await chat_svc.add_message(role="user", content=request.message, session_id=session_id)
            await chat_svc.add_message(role="assistant", content=agent_response.answer, session_id=session_id)
        except Exception as e:
            logger.warning("Failed to record agent response in DB: %s", e)
        return AssistantResponse(
            answer=agent_response.answer,
            route="project" if agent_response.route == "project_api" else "direct",
            session_id=agent_response.session_id,
            selected_tool=agent_response.selected_tool,
            operation=agent_response.operation,
            project_id=agent_response.project_id,
            tool_error=agent_response.tool_error,
        )
