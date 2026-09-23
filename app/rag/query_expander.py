"""
Stage 4: Query Expander
Responsible for acronym expansion, domain enrichment, and generating multi-angle search queries.
"""
import re
from dataclasses import dataclass, field


@dataclass
class ExpandedQueries:
    primary_query: str
    enriched_query: str | None = None
    search_terms: list[str] = field(default_factory=list)


class QueryExpander:
    """Expands queries with domain synonyms, acronym expansions, and multi-angle formulations."""

    ACRONYMS = {
        r"\bpto\b": "paid time off annual leave",
        r"\bposh\b": "prevention of sexual harassment internal complaints committee",
        r"\bf&f\b": "full and final settlement",
        r"\bfnf\b": "full and final settlement",
        r"\bmfa\b": "multi factor authentication 2fa",
        r"\bvpn\b": "virtual private network wireguard",
        r"\beap\b": "employee assistance program mental health therapy",
        r"\bwfh\b": "work from home remote work",
        r"\bicc\b": "internal complaints committee anti harassment",
    }

    DOMAIN_ENRICHMENTS = [
        (r"\b(meal|meals|food|dinner|lunch|per\s+diem)\b", "daily meal allowance $75 expense reimbursement travel finance portal"),
        (r"\b(flight|flights|airfare|ticket|tickets)\b", "domestic flight economy class 5 hours international premium economy travel booking"),
        (r"\b(hotel|hotels|lodging|accommodation)\b", "hotel accommodation tier-1 cities $180 $120 business travel"),
        (r"\b(insurance|hospital|hospitalization|mediclaim)\b", "group health insurance $50000 annual inpatient hospitalization spouse dependent children"),
        (r"\b(dental|vision|teeth|eye|eyes|glasses)\b", "dental and vision coverage $1000 annually per employee"),
        (r"\b(notice|resign|resignation|quitting)\b", "resignation notice period 60 days 30 days probation buyout 45 days settlement"),
        (r"\b(remote|hybrid|wfh)\b", "remote work 3 days per week internet utility $50 home office setup stipend $500"),
        (r"\b(laptop|macbook|dell|hardware)\b", "engineering laptop 16-inch macbook pro dell xps 15 32gb ram 3-year refresh cycle"),
        (r"\b(leave|leaves|vacation)\b", "paid vacation pto 18 days sick casual leave 12 days maternity 26 weeks paternity 4 weeks"),
        (r"\b(appraisal|rating|increment)\b", "performance appraisal 5-point rating scale bi-annual april october review november increment"),
    ]

    @classmethod
    def expand(cls, query: str) -> ExpandedQueries:
        q_lower = query.lower()

        # 1. Expand acronyms
        expanded_acronym_query = q_lower
        for pat, replacement in cls.ACRONYMS.items():
            if re.search(pat, expanded_acronym_query):
                expanded_acronym_query = re.sub(pat, replacement, expanded_acronym_query)

        # 2. Enrich with domain concepts
        enrichment_additions = []
        for pat, domain_concepts in cls.DOMAIN_ENRICHMENTS:
            if re.search(pat, q_lower):
                enrichment_additions.append(domain_concepts)

        enriched_query = None
        if enrichment_additions:
            enriched_query = f"{query} {' '.join(enrichment_additions)}"

        # 3. Compile search terms
        search_terms = list(set(re.findall(r"\b[a-zA-Z0-9_\-\$]{3,}\b", f"{query} {enriched_query or ''}")))

        return ExpandedQueries(
            primary_query=query,
            enriched_query=enriched_query,
            search_terms=search_terms,
        )
