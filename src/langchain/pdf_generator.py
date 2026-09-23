"""
Generates an official enterprise-grade WorkPilot Company Policy Handbook PDF
with structured 5-page layout, headers, page numbers, and corporate styling.
"""
import logging
from pathlib import Path

logger = logging.getLogger("src.langchain.pdf_generator")


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
                    "Expense Submission Deadline: All expense reports and receipts must be submitted within 30 days of expense incurrence via the employee portal."
                ]
            }
        ]
    },
    {
        "page": 3,
        "sections": [
            {
                "num": "5",
                "title": "Health Insurance and Wellness Benefits",
                "intro": "Comprehensive group health insurance is provided to all full-time permanent employees effective from day one of employment.",
                "bullets": [
                    "Coverage: Medical, surgical, and hospitalization coverage up to $50,000 per policy year.",
                    "Dependents: Policy covers the employee, spouse, and up to two dependent children.",
                    "Dental and Vision: An annual benefit of $500 per covered member is provided for preventive dental checkups, cleaning, and corrective eyewear.",
                    "Mental Health Support: Up to 8 confidential sessions per year with licensed counselors through our Employee Assistance Program (EAP).",
                    "Wellness Stipend: $50 per month toward gym memberships, yoga classes, or fitness subscriptions."
                ]
            },
            {
                "num": "6",
                "title": "Code of Conduct and Anti-Harassment",
                "intro": "WorkPilot is committed to providing a safe, inclusive, and harassment-free workplace for everyone.",
                "bullets": [
                    "Zero Tolerance: Harassment, discrimination, or bullying based on race, gender, religion, sexual orientation, disability, or age will result in immediate disciplinary action up to termination.",
                    "Reporting: Incidents can be reported directly to People Operations, a designated HR partner, or anonymously via our confidential whistle-blower helpline.",
                    "Non-Retaliation: Retaliation against any employee reporting a violation in good faith is strictly prohibited."
                ]
            }
        ]
    },
    {
        "page": 4,
        "sections": [
            {
                "num": "7",
                "title": "Device Security, Data Protection, and Acceptable Use",
                "intro": "All company-provided laptops and equipment are monitored for security compliance.",
                "bullets": [
                    "Multi-Factor Authentication (MFA): Mandatory for all company accounts, SSO, VPN, and email access.",
                    "Password Policy: Minimum 12 characters with a mix of uppercase, lowercase, numbers, and symbols, rotated every 90 days.",
                    "Data Classification: Customer data and source code are classified as Confidential and must never be copied to personal devices, USB drives, or unapproved cloud storage.",
                    "Incident Reporting: Lost or stolen laptops must be reported to the IT Security Team within 2 hours of discovery for immediate remote wipe."
                ]
            },
            {
                "num": "8",
                "title": "Performance Reviews, Promotions, and Appraisals",
                "intro": "Performance appraisals follow a structured bi-annual review cycle in June and December.",
                "bullets": [
                    "Self-evaluation followed by 360-degree peer feedback and manager review.",
                    "Performance ratings range from 1 (Needs Improvement) to 5 (Exceeds Expectations).",
                    "Promotion Eligibility: Requires minimum 12 months in current role and sustained rating of 4 or above in the previous two evaluation cycles.",
                    "Annual Merit Increases: Effective annually on April 1 based on overall company performance and individual ratings."
                ]
            }
        ]
    },
    {
        "page": 5,
        "sections": [
            {
                "num": "9",
                "title": "Learning, Development, and Certifications",
                "intro": "Continuous learning and professional growth are core values at WorkPilot.",
                "bullets": [
                    "Annual Learning Budget: $1,200 per full-time employee per calendar year for courses, books, workshops, and conferences.",
                    "Professional Certifications: Examination fees for relevant technical or domain certifications are 100% reimbursed upon passing.",
                    "Study Leave: Up to 3 days of paid study leave per year for approved certification examinations."
                ]
            },
            {
                "num": "10",
                "title": "Separation, Resignation, and Exit Process",
                "intro": "Guidelines for a smooth offboarding process when an employee leaves the company.",
                "bullets": [
                    "Notice Period: Standard notice period is 30 days for individual contributors and 60 days for lead and managerial roles.",
                    "Notice Buyout: Permissible only with written approval from the Department Head and People Operations.",
                    "Asset Return: All company property including laptops, monitors, access cards, and company credit cards must be returned by the last working day.",
                    "Full and Final Settlement: Processed within 30 days of the last working day, including encashment of eligible unused PTO days."
                ]
            }
        ]
    }
]


def generate_company_policy_pdf(target_path: Path | str | None = None) -> Path:
    """
    Generates an official enterprise-grade WorkPilot Company Policy Handbook PDF
    with structured 5-page layout, headers, page numbers, and corporate styling.
    """
    if target_path is None:
        target_path = Path(__file__).resolve().parent.parent.parent / "data" / "WorkPilot_Company_Policy.pdf"
    else:
        target_path = Path(target_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        logger.warning("ReportLab is not installed; writing fallback text-based PDF placeholder")
        target_path.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\nxref\n0 3\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n108\n%%EOF\n")
        return target_path

    doc = SimpleDocTemplate(
        str(target_path),
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    doc_header_style = ParagraphStyle(
        "DocHeader",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=2,
    )

    doc_sub_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=6,
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
