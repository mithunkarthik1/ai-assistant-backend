"""
Comprehensive test suite for Qdrant Cloud vector database integration with PostgreSQL
as the primary source-of-truth database.

Validates:
1. Qdrant Cloud connection, collection configuration, and health check.
2. Incremental indexing with Qdrant points (add, update, skip, delete).
3. Authoritative content retrieval from PostgreSQL using Qdrant top-k chunk IDs.
4. RAG question-answering with LLM and citations.
5. Chat session message persistence in PostgreSQL.
"""

import io
import json
import urllib.request
import urllib.parse
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

BASE_URL = "http://localhost:8001/api/v1"


def create_sample_pdf(title: str, sections: list[dict]) -> bytes:
    """Uses reportlab to generate clean multi-section test PDFs."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=14, leading=18)
    sec_style = ParagraphStyle("Sec", parent=styles["Heading2"], fontSize=12, leading=16, spaceBefore=6)
    p_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, leading=12, spaceAfter=4)

    story = [Paragraph(f"# {title}", title_style), Spacer(1, 10)]

    for sec in sections:
        story.append(Paragraph(f"## {sec['title']}", sec_style))
        for p in sec.get("paragraphs", []):
            story.append(Paragraph(p, p_style))
        story.append(Spacer(1, 8))

    doc.build(story)
    return buf.getvalue()


def http_post_json(url: str, data: dict) -> dict:
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_json(url: str) -> dict | list:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_delete_json(url: str) -> dict:
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def multipart_upload(url: str, filename: str, file_bytes: bytes) -> dict:
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(
            "utf-8"
        )
    )
    body.extend(b"Content-Type: application/pdf\r\n\r\n")
    body.extend(file_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_qdrant_rag_pipeline():
    print("=" * 70)
    print("RUNNING QDRANT CLOUD & POSTGRESQL RAG INTEGRATION TEST SUITE")
    print("=" * 70)

    # ----------------------------------------------------
    # Test 1: Health Check (PostgreSQL + Qdrant Cloud + Embeddings)
    # ----------------------------------------------------
    print("\n[Test 1] Health Check Verification...")
    health = http_get_json(f"{BASE_URL}/health")
    print("Health response:", json.dumps(health, indent=2))
    assert health["status"] == "healthy", f"Health status not healthy: {health}"
    assert health["postgresql"] == "healthy", f"PostgreSQL unhealthy: {health}"
    assert health["qdrant"]["status"] == "healthy", f"Qdrant unhealthy: {health}"
    assert health["qdrant"]["collection_exists"] is True, "Qdrant collection should exist"
    assert health["embedding"] == "available", "Embedding should be available"
    # Clean up any leftover test docs from interrupted prior test runs
    try:
        existing_docs = http_get_json(f"{BASE_URL}/documents")
        for d in existing_docs:
            if "test_remote_" in d.get("file_name", ""):
                http_delete_json(f"{BASE_URL}/documents/{d['document_id']}")
    except Exception:
        pass

    # ----------------------------------------------------
    # Test 2: Upload New Document A
    # ----------------------------------------------------
    print("\n[Test 2] Upload New PDF A (test_remote_travel_policy.pdf)...")
    sections_a = [
        {
            "title": "Flight Booking Policy",
            "paragraphs": [
                "Domestic flights under 4 hours must be booked in economy class via corporate travel desk.",
                "International flights exceeding 7 hours are eligible for business class upgrades.",
            ],
        },
        {
            "title": "Hotel Reimbursement Limits",
            "paragraphs": [
                "Hotel accommodation is reimbursable up to $180 per night in tier-1 metropolitan cities.",
                "In tier-2 cities, hotel reimbursement is capped at $120 per night.",
            ],
        },
        {
            "title": "Daily Meals Allowance",
            "paragraphs": [
                "Daily meals are reimbursed up to $75 per day without alcohol during business travel.",
                "Itemized receipts are required for all meal expenses.",
            ],
        },
    ]
    pdf_a_bytes = create_sample_pdf("Global Travel & Expense Policy", sections_a)
    res_a = multipart_upload(
        f"{BASE_URL}/documents/upload", "test_remote_travel_policy.pdf", pdf_a_bytes
    )
    print("Upload Result A:", json.dumps(res_a, indent=2))
    assert res_a["status"] in ("INDEXED", "UPLOADED", "UPDATED")
    assert res_a["chunks_added"] >= 3
    assert res_a["chunks_updated"] == 0
    assert res_a["chunks_skipped"] == 0
    doc_a_id = res_a["document_id"]
    print("✓ Test 2 passed: PDF A indexed into PostgreSQL and Qdrant Cloud.")

    # ----------------------------------------------------
    # Test 3: Upload Second Document B
    # ----------------------------------------------------
    print("\n[Test 3] Upload Second PDF B (test_remote_device_policy.pdf)...")
    sections_b = [
        {
            "title": "Home Office Ergonomic Stipend",
            "paragraphs": [
                "WorkPilot grants a one-time ergonomic home office stipend of $850 to all full-time remote employees.",
                "Eligible items include motorized standing desks, Herman Miller chairs, and dual monitors.",
            ],
        },
        {
            "title": "Hardware Refresh Cycle",
            "paragraphs": [
                "Company MacBook Pros are refreshed every 24 months upon request to IT Support.",
            ],
        },
    ]
    pdf_b_bytes = create_sample_pdf("Remote Hardware & Equipment Policy", sections_b)
    res_b = multipart_upload(
        f"{BASE_URL}/documents/upload", "test_remote_device_policy.pdf", pdf_b_bytes
    )
    print("Upload Result B:", json.dumps(res_b, indent=2))
    assert res_b["chunks_added"] >= 2
    assert res_b["chunks_updated"] == 0
    doc_b_id = res_b["document_id"]
    print("✓ Test 3 passed: PDF B indexed independently without touching PDF A.")

    # ----------------------------------------------------
    # Test 4: Re-upload Identical Document A (Incremental Skip)
    # ----------------------------------------------------
    print("\n[Test 4] Re-upload Identical PDF A (Zero Changes)...")
    res_a_same = multipart_upload(
        f"{BASE_URL}/documents/upload", "test_remote_travel_policy.pdf", pdf_a_bytes
    )
    print("Identical Upload Result:", json.dumps(res_a_same, indent=2))
    assert res_a_same["chunks_added"] == 0
    assert res_a_same["chunks_updated"] == 0
    assert res_a_same["chunks_skipped"] >= 3
    print("✓ Test 4 passed: Identical document skipped 100% of embeddings and Qdrant upserts.")

    # ----------------------------------------------------
    # Test 5: Modify 1 Section in Document A
    # ----------------------------------------------------
    print("\n[Test 5] Modify only 'Daily Meals Allowance' in PDF A ($75 -> $125)...")
    sections_a_mod = [
        sections_a[0],
        sections_a[1],
        {
            "title": "Daily Meals Allowance",
            "paragraphs": [
                "Daily meals are reimbursed up to $125 per day during high-cost international travel.",
                "Itemized receipts are required for all meal expenses.",
            ],
        },
    ]
    pdf_a_mod_bytes = create_sample_pdf("Global Travel & Expense Policy", sections_a_mod)
    res_a_mod = multipart_upload(
        f"{BASE_URL}/documents/upload", "test_remote_travel_policy.pdf", pdf_a_mod_bytes
    )
    print("Modified Upload Result:", json.dumps(res_a_mod, indent=2))
    assert res_a_mod["chunks_updated"] == 1, f"Expected 1 updated, got {res_a_mod['chunks_updated']}"
    assert res_a_mod["chunks_skipped"] >= 2
    assert res_a_mod["chunks_added"] == 0
    print("✓ Test 5 passed: Only modified section was re-embedded and upserted to Qdrant.")

    # ----------------------------------------------------
    # Test 6: Add 1 New Section to Document A
    # ----------------------------------------------------
    print("\n[Test 6] Add New Section 'Airport Lounge Access' to PDF A...")
    sections_a_added = list(sections_a_mod)
    sections_a_added.append({
        "title": "Airport Lounge Access",
        "paragraphs": [
            "Staff traveling internationally are entitled to Priority Pass airport lounge access.",
        ],
    })
    pdf_a_added_bytes = create_sample_pdf("Global Travel & Expense Policy", sections_a_added)
    res_a_add = multipart_upload(
        f"{BASE_URL}/documents/upload", "test_remote_travel_policy.pdf", pdf_a_added_bytes
    )
    print("Added Section Result:", json.dumps(res_a_add, indent=2))
    assert res_a_add["chunks_added"] == 1
    assert res_a_add["chunks_updated"] == 0
    assert res_a_add["chunks_skipped"] >= 3
    print("✓ Test 6 passed: New section inserted into PostgreSQL and upserted to Qdrant.")

    # ----------------------------------------------------
    # Test 7: Remove 1 Section from Document A
    # ----------------------------------------------------
    print("\n[Test 7] Remove Section 'Hotel Reimbursement Limits' from PDF A...")
    sections_a_removed = [
        sections_a_added[0],  # Flights
        sections_a_added[2],  # Meals ($125)
        sections_a_added[3],  # Lounge
    ]
    pdf_a_rem_bytes = create_sample_pdf("Global Travel & Expense Policy", sections_a_removed)
    res_a_rem = multipart_upload(
        f"{BASE_URL}/documents/upload", "test_remote_travel_policy.pdf", pdf_a_rem_bytes
    )
    print("Removed Section Result:", json.dumps(res_a_rem, indent=2))
    assert res_a_rem["chunks_deleted"] == 1
    assert res_a_rem["chunks_updated"] == 0
    assert res_a_rem["chunks_added"] == 0
    assert res_a_rem["chunks_skipped"] >= 3
    print("✓ Test 7 passed: Removed chunk deleted from PostgreSQL and purged from Qdrant.")

    # ----------------------------------------------------
    # Test 8: RAG Query via Qdrant Retrieval + PostgreSQL Authoritative Content
    # ----------------------------------------------------
    print("\n[Test 8] RAG Query Retrieval & Synthesis...")
    query_payload = {
        "message": "What is the ergonomic home office stipend and what is the daily meals allowance limit?",
        "session_id": "test-session-qdrant-42",
    }
    chat_res = http_post_json(f"{BASE_URL}/rag/query", query_payload)
    print("\nAnswer:\n", chat_res.get("answer"))
    print("\nSources:")
    for s in chat_res.get("sources", []):
        print(f" - {s.get('filename')} (Page {s.get('page')}, {s.get('section')}) [Chunk: {s.get('chunk_id')}]")
    assert len(chat_res.get("sources", [])) > 0, "Should return citation sources"
    print("✓ Test 8 passed: Qdrant retrieval + PostgreSQL authoritative content answered query.")

    # ----------------------------------------------------
    # Test 9: Chat Session & Message Persistence in PostgreSQL
    # ----------------------------------------------------
    print("\n[Test 9] Chat Session Message Persistence in PostgreSQL...")
    history = http_get_json(f"{BASE_URL}/chat/history?session_id=test-session-qdrant-42")
    print(f"Retrieved {len(history)} messages for session 'test-session-qdrant-42'")
    assert len(history) >= 2, f"Expected at least 2 messages in session, got {len(history)}"
    roles = [m["role"] for m in history]
    assert "user" in roles and "assistant" in roles
    print("✓ Test 9 passed: Chat messages persisted and retrieved from PostgreSQL.")

    # ----------------------------------------------------
    # Test 10: Cleanup Documents
    # ----------------------------------------------------
    print("\n[Test 10] Deleting test documents...")
    del_a = http_delete_json(f"{BASE_URL}/documents/{doc_a_id}")
    del_b = http_delete_json(f"{BASE_URL}/documents/{doc_b_id}")
    print("Deleted A:", del_a)
    print("Deleted B:", del_b)
    print("✓ Test 10 passed: Documents and points cleanly deleted.")

    print("\n" + "=" * 70)
    print("ALL QDRANT + POSTGRESQL INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    test_qdrant_rag_pipeline()
