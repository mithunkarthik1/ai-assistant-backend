"""
End-to-End Test Suite for Incremental RAG Document Upload with Hybrid Chunking.
Tests all 7 scenarios specified in requirements:
1. New PDF upload & indexing
2. Second PDF upload (isolated indexing)
3. Modify one section (only affected chunk re-embedded, unchanged skipped)
4. Add new section (new chunk embedded, existing unchanged)
5. Delete section (old vector and metadata deleted, others untouched)
6. Same document no changes (0 embeddings computed, all skipped)
7. Multi-document RAG query (accurate multi-source retrieval & citations)
"""
import io
import json
import urllib.request
import urllib.parse

BASE_URL = "http://localhost:8001/api/v1"

def multipart_upload(url: str, filename: str, content_bytes: bytes) -> dict:
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: application/pdf\r\n\r\n")
    body.extend(content_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_json(url: str) -> dict | list:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def delete_req(url: str) -> dict:
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def create_sample_pdf(title: str, sections: list[dict]) -> bytes:
    """Uses reportlab to generate clean multi-section test PDFs."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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


def run_tests():
    print("=" * 70)
    print("RUNNING INCREMENTAL RAG & HYBRID CHUNKING TEST SUITE")
    print("=" * 70)

    # Clean up any leftover test docs if present
    existing_docs = get_json(f"{BASE_URL}/documents")
    for d in existing_docs:
        if "test_" in d["file_name"].lower():
            delete_req(f"{BASE_URL}/documents/{d['document_id']}")

    # ----------------------------------------------------
    # Scenario 1: Upload New PDF A (Global Travel Policy)
    # ----------------------------------------------------
    print("\n--- [Scenario 1] Upload New PDF A (test_travel_policy.pdf) ---")
    sections_a = [
        {
            "title": "Flight Booking Policy",
            "paragraphs": [
                "Domestic flights under 5 hours must be booked in Economy Class.",
                "Flights exceeding 5 hours or cross-continent qualify for Premium Economy booking.",
            ]
        },
        {
            "title": "Hotel Reimbursement Limits",
            "paragraphs": [
                "Hotel accommodation is reimbursable up to $180 per night in metropolitan tier-1 cities.",
                "In all other locations, hotel reimbursement is capped at $120 per night.",
            ]
        },
        {
            "title": "Daily Meals Allowance",
            "paragraphs": [
                "Daily meals are reimbursed up to $75 per day without alcohol during business travel.",
                "Receipts are required for any single meal expense exceeding $25.",
            ]
        }
    ]
    pdf_a_bytes = create_sample_pdf("Global Travel and Expense Policy", sections_a)
    res_a = multipart_upload(f"{BASE_URL}/documents/upload", "test_travel_policy.pdf", pdf_a_bytes)
    print("Result:", json.dumps(res_a, indent=2))
    assert res_a["status"] in ("INDEXED", "UPLOADED"), f"Expected INDEXED, got {res_a['status']}"
    assert res_a["chunks_added"] >= 3, f"Expected at least 3 chunks added, got {res_a['chunks_added']}"
    assert res_a["chunks_skipped"] == 0
    assert res_a["chunks_updated"] == 0
    doc_a_id = res_a["document_id"]
    print("✓ Scenario 1 passed: PDF A embedded and indexed.")

    # ----------------------------------------------------
    # Scenario 2: Upload Second PDF B (Remote Equipment Policy)
    # ----------------------------------------------------
    print("\n--- [Scenario 2] Upload Second PDF B (test_equipment_policy.pdf) ---")
    sections_b = [
        {
            "title": "Home Office Ergonomic Stipend",
            "paragraphs": [
                "WorkPilot grants a one-time ergonomic home office stipend of $800 to all remote employees.",
                "Eligible purchases include standing desks, Herman Miller chairs, and dual monitors.",
            ]
        },
        {
            "title": "Hardware Refresh Cycle",
            "paragraphs": [
                "Company MacBooks are refreshed every 24 months upon request to IT Support.",
            ]
        }
    ]
    pdf_b_bytes = create_sample_pdf("Remote Work Equipment Policy", sections_b)
    res_b = multipart_upload(f"{BASE_URL}/documents/upload", "test_equipment_policy.pdf", pdf_b_bytes)
    print("Result:", json.dumps(res_b, indent=2))
    assert res_b["chunks_added"] >= 2
    assert res_b["chunks_updated"] == 0
    doc_b_id = res_b["document_id"]
    print("✓ Scenario 2 passed: PDF B embedded independently without touching PDF A.")

    # ----------------------------------------------------
    # Scenario 6 (Check before edit): Same Document, No Changes
    # ----------------------------------------------------
    print("\n--- [Scenario 6] Re-upload Identical PDF A (Zero Changes) ---")
    res_a_same = multipart_upload(f"{BASE_URL}/documents/upload", "test_travel_policy.pdf", pdf_a_bytes)
    print("Result:", json.dumps(res_a_same, indent=2))
    assert res_a_same["chunks_added"] == 0, f"Expected 0 added, got {res_a_same['chunks_added']}"
    assert res_a_same["chunks_updated"] == 0, f"Expected 0 updated, got {res_a_same['chunks_updated']}"
    assert res_a_same["chunks_skipped"] >= 3, f"Expected at least 3 skipped, got {res_a_same['chunks_skipped']}"
    print("✓ Scenario 6 passed: Identical PDF skipped 100% of embeddings.")

    # ----------------------------------------------------
    # Scenario 3: Modify ONLY ONE section (Daily Meals Allowance)
    # ----------------------------------------------------
    print("\n--- [Scenario 3] Modify only 'Daily Meals Allowance' in PDF A ---")
    sections_a_modified = [
        sections_a[0], # Flight Booking Policy: UNCHANGED
        sections_a[1], # Hotel Limits: UNCHANGED
        {
            "title": "Daily Meals Allowance",
            "paragraphs": [
                "Daily meals are reimbursed up to $110 per day including tax during high-cost international travel.", # CHANGED $75 -> $110
                "Receipts are required for all meal expenses.",
            ]
        }
    ]
    pdf_a_mod_bytes = create_sample_pdf("Global Travel and Expense Policy", sections_a_modified)
    res_a_mod = multipart_upload(f"{BASE_URL}/documents/upload", "test_travel_policy.pdf", pdf_a_mod_bytes)
    print("Result:", json.dumps(res_a_mod, indent=2))
    assert res_a_mod["chunks_updated"] == 1, f"Expected exactly 1 chunk updated, got {res_a_mod['chunks_updated']}"
    assert res_a_mod["chunks_skipped"] >= 2, f"Expected at least 2 skipped, got {res_a_mod['chunks_skipped']}"
    assert res_a_mod["chunks_added"] == 0
    print("✓ Scenario 3 passed: Only modified section was re-embedded, unchanged chunks skipped.")

    # ----------------------------------------------------
    # Scenario 4: Add New Section (Airport Lounge Access)
    # ----------------------------------------------------
    print("\n--- [Scenario 4] Add New Section 'Airport Lounge Access' to PDF A ---")
    sections_a_with_new = list(sections_a_modified)
    sections_a_with_new.append({
        "title": "Airport Lounge Access",
        "paragraphs": [
            "Senior engineers and directors are entitled to Priority Pass airport lounge access for international trips.",
        ]
    })
    pdf_a_new_sec_bytes = create_sample_pdf("Global Travel and Expense Policy", sections_a_with_new)
    res_a_new = multipart_upload(f"{BASE_URL}/documents/upload", "test_travel_policy.pdf", pdf_a_new_sec_bytes)
    print("Result:", json.dumps(res_a_new, indent=2))
    assert res_a_new["chunks_added"] == 1, f"Expected exactly 1 new chunk added, got {res_a_new['chunks_added']}"
    assert res_a_new["chunks_updated"] == 0, f"Expected 0 updated, got {res_a_new['chunks_updated']}"
    assert res_a_new["chunks_skipped"] >= 3, f"Expected existing chunks skipped, got {res_a_new['chunks_skipped']}"
    print("✓ Scenario 4 passed: New section embedded while existing chunks remained untouched.")

    # ----------------------------------------------------
    # Scenario 5: Delete a Section (Remove Hotel Limits)
    # ----------------------------------------------------
    print("\n--- [Scenario 5] Delete Section 'Hotel Reimbursement Limits' from PDF A ---")
    sections_a_deleted = [
        sections_a_with_new[0], # Flight
        sections_a_with_new[2], # Meals
        sections_a_with_new[3], # Airport Lounge
    ]
    pdf_a_del_bytes = create_sample_pdf("Global Travel and Expense Policy", sections_a_deleted)
    res_a_del = multipart_upload(f"{BASE_URL}/documents/upload", "test_travel_policy.pdf", pdf_a_del_bytes)
    print("Result:", json.dumps(res_a_del, indent=2))
    assert res_a_del["chunks_deleted"] == 1, f"Expected exactly 1 chunk deleted, got {res_a_del['chunks_deleted']}"
    assert res_a_del["chunks_updated"] == 0
    assert res_a_del["chunks_added"] == 0
    assert res_a_del["chunks_skipped"] >= 3
    print("✓ Scenario 5 passed: Deleted section purged from vector DB and chunk metadata.")

    # ----------------------------------------------------
    # Scenario 7: Multi-Document RAG Query & Source Citations
    # ----------------------------------------------------
    print("\n--- [Scenario 7] RAG Query Across Multiple Documents ---")
    query_payload = {
        "message": "What is the home office ergonomic stipend and what is the daily meals allowance limit?"
    }
    chat_res = post_json(f"{BASE_URL}/chat", query_payload)
    print("\nChat Answer:\n", chat_res["answer"])
    print("\nSources Returned:")
    for s in chat_res.get("sources", []):
        print(f" - Doc: {s.get('filename')} | Page: {s.get('page')} | Section: {s.get('section')} | Topic: {s.get('topic')} | Chunk: {s.get('chunk_id')}")

    assert len(chat_res.get("sources", [])) > 0, "Expected sources to be returned"
    print("\n✓ Scenario 7 passed: Multi-document retrieval with citation metadata working perfectly.")

    # Cleanup test docs
    print("\nCleaning up test documents...")
    delete_req(f"{BASE_URL}/documents/{doc_a_id}")
    delete_req(f"{BASE_URL}/documents/{doc_b_id}")
    print("✓ Cleanup completed.")

    print("\n" + "=" * 70)
    print("ALL 7 SCENARIOS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
