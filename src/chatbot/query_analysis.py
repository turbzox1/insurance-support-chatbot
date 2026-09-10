"""Cheap decisions for clear cases; one combined LLM analysis for uncertain cases."""

import re

INSURANCE = re.compile(
    r"\b(insurance|insurer|insured|policy|policies|premium|premiums|claim|claims|"
    r"ombudsman|ombudsmen|irdai|grievance|cashless|deductible|copay|co[ -]pay|"
    r"bajaj|bagic|sum insured|reimbursement|waiting period|no claim bonus)\b",
    re.I,
)
LIVE = re.compile(
    r"\b(current|currently|latest|today|recent|news|updates?|announcements?)\b"
    r"|\bthis (week|month|year)\b",
    re.I,
)
FOLLOWUP = re.compile(
    r"\b(it|they|them|this|that|these|those|its)\b"
    r"|^(and|what about|how about|can i buy|is there)\b",
    re.I,
)
OUTSIDE = re.compile(
    r"\b(python|javascript|programming|recipe|football|cricket|weather|poem)\b", re.I
)


def quick_analysis(question, history):
    """Return None when semantic classification / reference resolution is needed."""
    insurance = bool(INSURANCE.search(question))
    if OUTSIDE.search(question) and not insurance:
        return {
            "domain": "NON_INSURANCE",
            "intent": "KNOWLEDGE",
            "question_type": "COMPLETE",
            "rewritten_question": question,
        }
    if LIVE.search(question) and insurance:
        return {
            "domain": "INSURANCE",
            "intent": "LIVE_INFORMATION",
            "question_type": "COMPLETE",
            "rewritten_question": question,
        }
    if FOLLOWUP.search(question):
        if history:
            return None
        # A demonstrative alone is insufficient to determine personal claim eligibility.
        if re.search(r"\b(it|them|this|that|these|those|they|its)\b", question, re.I):
            return {
                "domain": "INSURANCE" if insurance else "UNKNOWN",
                "intent": "KNOWLEDGE",
                "question_type": "AMBIGUOUS",
                "rewritten_question": question,
            }
    if insurance:
        return {
            "domain": "INSURANCE",
            "intent": "KNOWLEDGE",
            "question_type": "COMPLETE",
            "rewritten_question": question,
        }
    return None
