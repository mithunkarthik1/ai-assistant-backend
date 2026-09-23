import asyncio
import re
from typing import NamedTuple
from langchain_core.messages import HumanMessage, AIMessage

from app.rag.chain import generate_rag_answer

class TestCase(NamedTuple):
    category: str
    query: str
    expected_keywords: list[str]
    expected_show_pdf: bool = False

TEST_CASES = [
    # 1. Working Hours & Core Timings
    TestCase("Hours", "what are working hours?", ["40", "10:00 AM", "4:00 PM"]),
    TestCase("Hours", "what are core collaboration hours?", ["10:00 AM", "4:00 PM"]),
    TestCase("Hours", "can I start work at 8 AM?", ["8:00 AM", "10:00 AM"]),
    TestCase("Hours", "can I start work at 7 AM?", ["8:00 AM", "not permitted"]),
    TestCase("Hours", "can I skip lunch and leave 1 hour early?", ["1-hour", "not permitted"]),

    # 2. Remote Work & Hybrid Guidelines
    TestCase("Remote", "how many days can I work from home?", ["3 days"]),
    TestCase("Remote", "can I work remotely 4 days a week without approval?", ["written approval", "Department Head"]),
    TestCase("Remote", "is there a setup stipend for home office?", ["$500", "stipend"]),
    TestCase("Remote", "what is the internet allowance?", ["$50", "monthly"]),
    TestCase("Remote", "how fast must the internet connection be?", ["50 Mbps"]),

    # 3. Leave Policy & PTO
    TestCase("Leave", "how many PTO days do we get?", ["18"]),
    TestCase("Leave", "how many unused vacation days carry over?", ["5"]),
    TestCase("Leave", "how much sick leave is provided?", ["12 days"]),
    TestCase("Leave", "do I need a doctor's certificate for 2 days sick leave?", ["3 consecutive days", "no doctor's note"]),
    TestCase("Leave", "what if I take 4 days of sick leave, do I need a medical note?", ["medical certificate", "3 consecutive days"]),
    TestCase("Leave", "what is the maternity leave duration?", ["26 weeks"]),
    TestCase("Leave", "what is paternity leave?", ["4 weeks"]),
    TestCase("Leave", "how many days for bereavement leave?", ["5"]),

    # 4. Travel & Expense Reimbursement
    TestCase("Travel", "what is the daily meal allowance on business travel?", ["$75"]),
    TestCase("Travel", "can I expense alcohol during business trips?", ["alcohol is strictly excluded"]),
    TestCase("Travel", "can I book premium economy for a 3-hour domestic flight?", ["Economy Class", "under 5 hours"]),
    TestCase("Travel", "can I fly premium economy for international flights?", ["Premium Economy", "over 5 hours"]),
    TestCase("Travel", "what is the hotel reimbursement limit for tier-1 cities?", ["$180"]),
    TestCase("Travel", "what is the deadline to submit expense claims?", ["30 days"]),
    TestCase("Travel", "can I submit expenses after 45 days?", ["30 days", "rejection"]),

    # 5. IT Equipment & Hardware
    TestCase("IT", "what laptop will I get as a software engineer?", ["MacBook Pro", "Dell XPS 15", "32GB"]),
    TestCase("IT", "what monitor and accessories are provided?", ["27-inch 4K monitor", "headset"]),
    TestCase("IT", "how often is laptop hardware refreshed?", ["3 years"]),
    TestCase("IT", "can I keep the MacBook when I resign?", ["returned to IT Support", "company property"]),

    # 6. Information Security & Passwords
    TestCase("Security", "what are the password complexity requirements?", ["12 characters"]),
    TestCase("Security", "is MFA mandatory?", ["mandatory", "Google Workspace"]),
    TestCase("Security", "how often must passwords be changed?", ["90 days"]),
    TestCase("Security", "can I reuse my old password?", ["5 passwords", "90 days"]),
    TestCase("Security", "after how many minutes of inactivity does the laptop auto-lock?", ["5 minutes"]),
    TestCase("Security", "is VPN required on public Wi-Fi?", ["WireGuard VPN"]),

    # 7. Employee Benefits & Insurance
    TestCase("Benefits", "what is the health insurance coverage limit?", ["$50,000", "inpatient"]),
    TestCase("Benefits", "are dental and vision covered?", ["$1,000"]),
    TestCase("Benefits", "is there a gym or sports reimbursement?", ["$60"]),
    TestCase("Benefits", "do we get free annual health checkups?", ["free annual health checkup vouchers"]),
    TestCase("Benefits", "how many free therapy sessions are included in EAP?", ["12"]),

    # 8. Code of Conduct & POSH
    TestCase("Conduct", "what is the POSH policy?", ["zero-tolerance", "ICC"]),
    TestCase("Conduct", "how do I report harassment?", ["ethics@workpilot.internal", "ICC"]),
    TestCase("Conduct", "what happens if someone retaliates against a whistleblower?", ["immediate termination"]),

    # 9. Performance Appraisal & Promotion
    TestCase("Appraisal", "when are performance reviews conducted?", ["April", "October"]),
    TestCase("Appraisal", "what is the rating scale for appraisals?", ["5-point"]),
    TestCase("Appraisal", "what are the 5 points in evaluation?", ["technical delivery", "ownership", "teamwork", "leadership"]),
    TestCase("Appraisal", "what are the core competencies or objectives for evaluation?", ["technical delivery", "ownership", "teamwork", "leadership"]),
    TestCase("Appraisal", "when are salary increments and promotions finalized?", ["November"]),

    # 10. Resignation & Notice Period
    TestCase("Notice", "what is the notice period for confirmed employees?", ["60 calendar days"]),
    TestCase("Notice", "what is the notice period during probation?", ["30 calendar days"]),
    TestCase("Notice", "can I buy out my notice period?", ["buyout", "mutual written consent"]),
    TestCase("Notice", "when is full and final settlement (F&F) paid?", ["45 days"]),
    TestCase("Notice", "60 days for what/", ["mandatory notice period", "resigning"]),
    TestCase("Notice", "what if i go before 60 days?", ["buyout", "mutual written consent"]),

    # 11. Typo / Colloquial / Slang Handling
    TestCase("Typos", "resioning procedure", ["notice period", "60 calendar days"]),
    TestCase("Typos", "employe ratng", ["5-point", "April", "October"]),
    TestCase("Typos", "coc policy", ["zero-tolerance", "harassment"]),
    TestCase("Typos", "wfh rules", ["3 days", "per week"]),
    TestCase("Typos", "fnf timeline", ["45 days", "Full & Final"]),

    # 12. Math & Reasoning / Leave Balance Calculation
    TestCase("Math", "if i take 4 days leave how many now left", ["14 days left"]),
    TestCase("Math", "If I take 5 days of vacation, how many PTO days remain?", ["13 PTO days left"]),
    TestCase("Math", "how many sick days left if i take 3?", ["9 days left"]),

    # 13. Compound & Comparison
    TestCase("Compound", "tell me about PTO and remote work", ["18 days", "3 days per week"]),
    TestCase("Compound", "difference between PTO and sick leave", ["18 days", "12 days"]),

    # 14. Catalog / Overview Queries
    TestCase("Catalog", "company policies", ["10 official company policies", "Working Hours", "Remote Work", "Notice Period"]),
    TestCase("Catalog", "company policy", ["10 official company policies"]),
    TestCase("Catalog", "what are company policies", ["10 official company policies"]),

    # 15. Conversational Turns
    TestCase("Chat", "hlo", ["WorkPilot employee assistant"]),
    TestCase("Chat", "hi dr", ["WorkPilot employee assistant"]),
    TestCase("Chat", "oi", ["WorkPilot employee assistant"]),
    TestCase("Chat", "hey", ["WorkPilot employee assistant"]),
    TestCase("Chat", "done", ["Glad to help"]),
    TestCase("Chat", "thnx", ["welcome"]),
    TestCase("Chat", "tq", ["welcome"]),
    TestCase("Chat", "who are you", ["WorkPilot's employee assistant"]),
    TestCase("Chat", "what is workpilot", ["WorkPilot is our enterprise organization"]),

    # 16. Out-of-Scope Queries (Referrals, show_pdf=True)
    TestCase("Scope", "cafeteria policy", ["Workplace Operations", "Office Administration"], expected_show_pdf=True),
    TestCase("Scope", "where can I park my car?", ["Building Management", "Workplace Facilities"], expected_show_pdf=True),
    TestCase("Scope", "what is the dress code?", ["People Operations"], expected_show_pdf=True),
    TestCase("Scope", "when is payday?", ["Finance and Payroll"], expected_show_pdf=True),
    TestCase("Scope", "how do I refer a candidate for a job?", ["Talent Acquisition"], expected_show_pdf=True),
    TestCase("Scope", "how to cook pasta?", ["culinary guides"], expected_show_pdf=True),
]

MULTI_TURN_SESSIONS = [
    (
        "Remote Work Session",
        [
            ("Can I work from home?", ["3 days per week"], False),
            ("Can I do 4 days?", ["requires written approval", "3 days per week"], False),
            ("What about internet allowance?", ["$50", "monthly"], False),
            ("How fast does it need to be?", ["50 Mbps"], False),
        ]
    ),
    (
        "Notice Period Session",
        [
            ("What is the standard notice period?", ["60 calendar days"], False),
            ("60 days for what/", ["mandatory notice period", "resigning"], False),
            ("what if i go before 60 days?", ["buyout", "mutual written consent"], False),
            ("when will i get my f&f?", ["45 days"], False),
        ]
    ),
    (
        "Leave & Calculation Session",
        [
            ("How many PTO days do we get?", ["18"], False),
            ("if i take 4 days leave means how many now left", ["14", "left"], False),
            ("and for sick leave?", ["8 days left"], False),
            ("do i need a doctor note for 4 days?", ["medical certificate is required", "3 consecutive days"], False),
        ]
    ),
    (
        "Travel Expenses Session",
        [
            ("What is the daily meal budget on business trips?", ["$75"], False),
            ("Can I buy beer with it?", ["strictly excluded"], False),
            ("How do I submit the receipts?", ["finance portal", "30 days"], False),
            ("What if I submit after 45 days?", ["rejection", "30 days"], False),
        ]
    ),
    (
        "Health Insurance & Claims Session",
        [
            ("what about the health insurance", ["$50,000", "spouse", "children"], False),
            ("how many can i claim", ["$50,000", "inpatient"], False),
            ("how much can i claim", ["$50,000", "annually"], False),
            ("coverable amount", ["$50,000"], False),
            ("coverable for", ["spouse", "dependent children"], False),
            ("is coverble for my family?", ["spouse", "dependent children"], False),
        ]
    ),
]

async def run_all_tests():
    print(f"Starting comprehensive scenario audit: {len(TEST_CASES)} standalone tests + {len(MULTI_TURN_SESSIONS)} multi-turn sessions...")
    passed = 0
    failed = 0
    failures = []

    # 1. Standalone test cases
    for idx, tc in enumerate(TEST_CASES, 1):
        try:
            res = await generate_rag_answer(tc.query)
            ans = res[0]
            show_pdf = getattr(res, "show_pdf", False)
            
            missing = [k for k in tc.expected_keywords if k.lower() not in ans.lower()]
            pdf_ok = (show_pdf == tc.expected_show_pdf)
            
            if missing or not pdf_ok:
                failed += 1
                err = f"[{tc.category}] Query: '{tc.query}'"
                if missing:
                    err += f"\n  Missing expected: {missing}"
                if not pdf_ok:
                    err += f"\n  show_pdf expected {tc.expected_show_pdf}, got {show_pdf}"
                err += f"\n  Actual answer: {ans}"
                failures.append(err)
                print(f"❌ Test {idx}/{len(TEST_CASES)} FAILED: {tc.query}")
            else:
                passed += 1
                print(f"✅ Test {idx}/{len(TEST_CASES)} PASSED: {tc.query}")
        except Exception as e:
            failed += 1
            failures.append(f"[{tc.category}] Query: '{tc.query}' -> EXCEPTION: {e}")
            print(f"❌ Test {idx}/{len(TEST_CASES)} ERROR: {tc.query} ({e})")

    # 2. Multi-turn sessions
    print("\n--- Multi-turn Sessions ---")
    session_passed = 0
    session_failed = 0
    for s_name, turns in MULTI_TURN_SESSIONS:
        history = []
        session_ok = True
        print(f"\nEvaluating {s_name}:")
        for turn_idx, (q, exp_kw, exp_pdf) in enumerate(turns, 1):
            try:
                res = await generate_rag_answer(q, chat_history=history)
                ans = res[0]
                show_pdf = getattr(res, "show_pdf", False)
                missing = [k for k in exp_kw if k.lower() not in ans.lower()]
                pdf_ok = (show_pdf == exp_pdf)
                if missing or not pdf_ok:
                    session_ok = False
                    err = f"[{s_name} Turn {turn_idx}] Query: '{q}'"
                    if missing:
                        err += f"\n  Missing expected: {missing}"
                    if not pdf_ok:
                        err += f"\n  show_pdf expected {exp_pdf}, got {show_pdf}"
                    err += f"\n  Actual answer: {ans}"
                    failures.append(err)
                    print(f"  ❌ Turn {turn_idx}: '{q}' FAILED (missing: {missing})")
                else:
                    print(f"  ✅ Turn {turn_idx}: '{q}' PASSED")
                history.append(HumanMessage(content=q))
                history.append(AIMessage(content=ans))
            except Exception as e:
                session_ok = False
                failures.append(f"[{s_name} Turn {turn_idx}] Query: '{q}' -> EXCEPTION: {e}")
                print(f"  ❌ Turn {turn_idx}: '{q}' ERROR: {e}")
        if session_ok:
            session_passed += 1
        else:
            session_failed += 1

    print("\n" + "="*60)
    print(f"AUDIT SUMMARY:")
    print(f"Standalone: {passed} passed, {failed} failed out of {len(TEST_CASES)}")
    print(f"Multi-turn: {session_passed} passed, {session_failed} failed out of {len(MULTI_TURN_SESSIONS)}")
    print("="*60)

    if failures:
        print("\nFAILURE DETAILS:")
        for f in failures:
            print("-"*40)
            print(f)
    else:
        print("🎉 ALL SCENARIO TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
