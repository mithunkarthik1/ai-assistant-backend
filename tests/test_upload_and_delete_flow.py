import asyncio
import io
import json
import urllib.request

def test_full_upload_qa_and_delete_flow():
    content = b"""Server Infrastructure and Disaster Recovery Guide 2026

Section: Core Infrastructure
The primary production database runs on CockroachDB v23.2 distributed across 3 regions.
The disaster recovery failover objective (RTO) is 15 minutes, and recovery point objective (RPO) is 30 seconds.
Emergency contacts for infrastructure failures are infra-oncall@workpilot.com or emergency hotline +1-800-555-0199.
"""

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b"Content-Disposition: form-data; name=\"file\"; filename=\"server_infra_guide.txt\"\r\n")
    body.write(b"Content-Type: text/plain\r\n\r\n")
    body.write(content)
    body.write(f"\r\n--{boundary}--\r\n".encode())

    # 1. Upload
    req = urllib.request.Request(
        "http://localhost:8001/api/v1/documents/upload",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as res:
        upload_data = json.loads(res.read().decode())
    
    print("\n[TEST] Upload status:", upload_data.get("status"))
    doc_id = upload_data["document_id"]
    assert doc_id is not None

    try:
        # 2. Check document list: default doc is protected, new doc is not
        req_list = urllib.request.urlopen("http://localhost:8001/api/v1/documents")
        docs = json.loads(req_list.read().decode())
        doc_map = {d["document_id"]: d for d in docs}
        default_docs = [d for d in docs if d.get("is_default") is True]
        assert len(default_docs) >= 1, "Expected at least one default document"
        default_doc_id = default_docs[0]["document_id"]
        assert doc_map[doc_id]["is_default"] is False

        # 3. Ask question about uploaded document
        payload = json.dumps({"message": "What is the disaster recovery failover objective RTO and RPO in the server guide?"}).encode()
        req_chat = urllib.request.Request(
            "http://localhost:8001/api/v1/assistant/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_chat) as chat_res:
            chat_data = json.loads(chat_res.read().decode())
        
        answer = chat_data.get("answer", "")
        sources = chat_data.get("sources", [])
        print("[TEST] Chat Answer:", answer)
        print("[TEST] Sources:", sources)
        
        assert "15 minutes" in answer or "15" in answer
        assert "30 seconds" in answer or "30" in answer
        assert len(sources) > 0
        assert any("server_infra_guide.txt" in s.get("filename", "") for s in sources)

    finally:
        # 4. Delete the uploaded document
        del_req = urllib.request.Request(f"http://localhost:8001/api/v1/documents/{doc_id}", method="DELETE")
        with urllib.request.urlopen(del_req) as del_res:
            del_data = json.loads(del_res.read().decode())
        print("[TEST] Delete Result:", del_data)
        assert del_data.get("deleted") is True

        # Verify default document deletion is blocked
        try:
            req_del_default = urllib.request.Request(f"http://localhost:8001/api/v1/documents/{default_doc_id}", method="DELETE")
            urllib.request.urlopen(req_del_default)
            assert False, "Default document deletion should have failed with 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400
            print("[TEST] Protected document correctly blocked with status 400")

    print("\n ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_full_upload_qa_and_delete_flow()
