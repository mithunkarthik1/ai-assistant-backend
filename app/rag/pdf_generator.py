"""
Generates an official enterprise-grade WorkPilot Company Policy Handbook PDF
with structured 5-page layout, headers, page numbers, and corporate styling.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, HRFlowable


POLICY_PAGES = [
    {
        "page": 1,
        "sections": [
            {
                "num": "1",
                "title": "Working Hours and Core Timings",
                "intro": "Standard working hours are Monday through Friday, 9:00 AM to 6:00 PM local time.",
                "bullets": [
                    "The company maintains a 40-hour work week with a mandatory 1-hour lunch break daily.",
                    "Core collaboration hours are 10:00 AM to 4:00 PM, during which team members must be reachable on Slack and available for scheduled meetings.",
                    "Flexible working hours permit employees to adjust their start time between 8:00 AM and 10:00 AM upon manager approval."
                ]
            },
            {
                "num": "2",
                "title": "Remote Work and Hybrid Guidelines",
                "intro": "The company operates on a flexible hybrid work model allowing up to 3 days of remote work per week.",
                "bullets": [
                    "Full-time remote work requires prior written approval from the Department Head and People Operations.",
                    "Remote employees receive a one-time home-office setup stipend of $500 to purchase ergonomic furniture and desk equipment.",
                    "A monthly internet and utility allowance of $50 is provided to eligible remote employees.",
                    "Employees working remotely must maintain a dedicated quiet workspace and stable internet connection of at least 50 Mbps."
                ]
            }
        ]
    },
    {
        "page": 2,
        "sections": [
            {
                "num": "3",
                "title": "Leave Policy and Paid Time Off (PTO)",
                "intro": "The annual leave year runs from January 1 to December 31.",
                "bullets": [
                    "Paid Time Off (PTO): Full-time employees accrue 18 days of paid vacation per calendar year. Up to 5 unused PTO days can be carried forward to the following year.",
                    "Sick and Casual Leave: Employees are entitled to 12 days of paid sick and casual leave annually. Medical certificates are required for sick leave extending beyond 3 consecutive days.",
                    "Maternity Leave: Female employees are entitled to 26 weeks of fully paid maternity leave for up to two surviving children.",
                    "Paternity Leave: Male employees and non-birthing partners are entitled to 4 weeks of fully paid parental leave to be taken within the first 6 months of childbirth or adoption.",
                    "Bereavement Leave: 5 consecutive paid days off are provided in the event of the loss of an immediate family member."
                ]
            },
            {
                "num": "4",
                "title": "Travel and Expense Reimbursement Policy",
                "intro": "Business-related expenses incurred on behalf of the company are eligible for reimbursement.",
                "bullets": [
                    "Daily Meal Allowance: Capped at $75 per day without alcohol during official business travel.",
                    "Flight Booking Policy: Domestic flights under 5 hours must be booked in Economy Class; flights over 5 hours or international flights qualify for Premium Economy.",
                    "Hotel Accommodation Limit: Reimbursable up to $180 per night in tier-1 cities and $120 per night in other locations.",
                    "Expense Submission Window: Expense claims along with valid itemized tax receipts must be submitted via the finance portal within 30 days of incurring the expense.",
                    "Late Expense Claims: Claims submitted after 30 days are subject to rejection."
                ]
            }
        ]
    },
    {
        "page": 3,
        "sections": [
            {
                "num": "5",
                "title": "IT Equipment and Hardware Policy",
                "intro": "The company provides each full-time employee with enterprise-grade hardware to perform their duties.",
                "bullets": [
                    "Standard Engineering Laptops: Apple MacBook Pro 16-inch (M3/M4) or Dell XPS 15 laptop with 32GB RAM.",
                    "Hardware Accessories Provided: External 27-inch 4K monitor, wireless keyboard, mouse, and noise-canceling headset.",
                    "Hardware Refresh Cycle: Occurs every 3 years.",
                    "Asset Ownership and Return: All hardware remains company property and must be returned to IT Support upon termination or resignation."
                ]
            },
            {
                "num": "6",
                "title": "Information Security and Password Policy",
                "intro": "Information security is mandatory for all employees to safeguard client data and intellectual property.",
                "bullets": [
                    "Password Complexity Requirement: Passwords must be at least 12 characters in length, containing uppercase letters, lowercase letters, numbers, and at least one special symbol.",
                    "Multi-Factor Authentication (MFA): Strictly mandatory on all company accounts, Google Workspace, and GitHub.",
                    "Password Expiry and Rotation: Passwords must be updated every 90 days and cannot match the previous 5 passwords.",
                    "Auto-Lock Policy: Company devices must never be left unattended in public places and screens must auto-lock after 5 minutes of inactivity.",
                    "VPN Requirement: Connection to company networks from public Wi-Fi requires active connection through the official company WireGuard VPN."
                ]
            }
        ]
    },
    {
        "page": 4,
        "sections": [
            {
                "num": "7",
                "title": "Employee Benefits and Group Health Insurance",
                "intro": "The company offers comprehensive health and wellness coverage for full-time employees and their immediate families.",
                "bullets": [
                    "Group Health Insurance Coverage: Provides up to $50,000 annual inpatient hospitalization coverage covering employee, spouse, and up to two dependent children.",
                    "Annual Health Checkup: Free vouchers provided annually to all employees and spouses.",
                    "Dental and Vision Benefits: Covered up to $1,000 annually per employee.",
                    "Mental Health & Wellness: 12 free confidential therapy and counseling sessions per year via our Employee Assistance Program (EAP).",
                    "Gym and Fitness Reimbursement: Offers up to $60 per month towards gym memberships, yoga classes, or sports subscriptions."
                ]
            },
            {
                "num": "8",
                "title": "Code of Conduct and Anti-Harassment (POSH)",
                "intro": "WorkPilot enforces a zero-tolerance policy against any form of discrimination, harassment, or retaliation.",
                "bullets": [
                    "POSH Compliance: The company complies strictly with the Prevention of Sexual Harassment (POSH) framework.",
                    "Incident Reporting: Any observed or experienced harassment must be reported directly to the Internal Complaints Committee (ICC) or anonymously via ethics@workpilot.internal.",
                    "Anti-Retaliation Policy: Retaliation against anyone filing a complaint or participating in an investigation results in immediate termination."
                ]
            }
        ]
    },
    {
        "page": 5,
        "sections": [
            {
                "num": "9",
                "title": "Performance Appraisal and Promotion Policy",
                "intro": "Performance evaluations are conducted twice per year to foster professional growth.",
                "bullets": [
                    "Appraisal Cycles: Review cycles occur bi-annually in April (mid-year review) and October (annual performance & compensation appraisal).",
                    "Rating Framework: Performance is rated on a 5-point scale across technical delivery, ownership, teamwork, and leadership principles.",
                    "Salary Increments and Promotions: Finalized in November following the October appraisal cycle."
                ]
            },
            {
                "num": "10",
                "title": "Resignation and Notice Period Protocol",
                "intro": "The standard notice period for full-time confirmed employees is 60 calendar days.",
                "bullets": [
                    "Probationary employees have a notice period of 30 calendar days.",
                    "Buyout of notice period requires mutual written consent between the employee and department head.",
                    "Final settlement (Full & Final / F&F) including accrued salary, gratuity, and leave encashment is disbursed within 45 days of the last working day."
                ]
            }
        ]
    }
]


def generate_company_policy_pdf(output_path: Path | str | None = None) -> Path:
    """Generates a 5-page official WorkPilot Company Policy Handbook PDF."""
    if output_path is None:
        target_path = Path(__file__).resolve().parent.parent.parent / "data" / "WorkPilot_Company_Policy.pdf"
    else:
        target_path = Path(output_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(target_path),
        pagesize=letter,
        leftMargin=46,
        rightMargin=46,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    doc_header_style = ParagraphStyle(
        "DocHeader",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )

    doc_sub_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=8,
    )

    sec_title_style = ParagraphStyle(
        "SecTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=8,
        spaceAfter=4,
    )

    intro_style = ParagraphStyle(
        "IntroStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        leftIndent=14,
        spaceAfter=4,
    )

    page_footer_style = ParagraphStyle(
        "PageFooter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#94A3B8"),
        alignment=1, # Center
    )

    story = []

    for page_idx, page_data in enumerate(POLICY_PAGES):
        page_num = page_data["page"]

        # Header banner
        story.append(Paragraph("WORKPILOT ENTERPRISE COMPANY POLICY & EMPLOYEE HANDBOOK", doc_header_style))
        story.append(Paragraph("Official Human Resources & Operations Guidelines | Confidential & Proprietary", doc_sub_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=10))

        # Sections for this page
        for sec in page_data["sections"]:
            sec_heading = f"## {sec['num']}. {sec['title']}"
            story.append(Paragraph(sec_heading, sec_title_style))
            if sec.get("intro"):
                story.append(Paragraph(sec["intro"], intro_style))

            for bullet in sec.get("bullets", []):
                # Clean and format bold labels
                if ":" in bullet:
                    parts = bullet.split(":", 1)
                    bullet_text = f"• <b>{parts[0].strip()}:</b> {parts[1].strip()}"
                else:
                    bullet_text = f"• {bullet.strip()}"
                story.append(Paragraph(bullet_text, bullet_style))

            story.append(Spacer(1, 6))

        # Space and footer at bottom
        story.append(Spacer(1, 14))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceAfter=6))
        story.append(Paragraph(f"Page {page_num} of 5 — WorkPilot Policy Documentation", page_footer_style))

        if page_idx < len(POLICY_PAGES) - 1:
            story.append(PageBreak())

    doc.build(story)
    return target_path


if __name__ == "__main__":
    generated = generate_company_policy_pdf()
    print(f"Generated PDF at: {generated}")
