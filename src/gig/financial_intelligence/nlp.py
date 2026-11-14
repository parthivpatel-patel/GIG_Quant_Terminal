"""Original finite rules for cited financial mentions, never trained NLP or truth.

Only explicit supported units are parsed. All event outputs are text mentions;
negation/conditional flags indicate lexical cues, not syntactic understanding.
"""
from __future__ import annotations

import re
from decimal import Decimal, localcontext

IMPLEMENTATION = "gig.financial-mention-rules.v1"
NUMBER = r"[+-]?(?:[0-9][0-9A-Za-z_.,/⁄−–—+-]*|nan|inf(?:inity)?)"
MEASUREMENT = re.compile(r"(?<![\w.,/⁄−–—+-])(?P<number>"+NUMBER+r")\s*(?P<unit>percent\b|%|basis\s+points\b|bps\b)(?![\w,/+-])", re.I)
MONEY = re.compile(r"(?<![\w])(?P<currency>USD\s*|US\$\s*|\$)(?P<number>"+NUMBER+r")(?:\s*(?P<scale>thousand|million|billion)\b)?", re.I)
VALID_NUMBER = re.compile(r"[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?")
TENOR = re.compile(r"(?<![\w+−–—-])(?:BC_(?P<code>[0-9]{1,3})(?P<code_unit>MONTH|YEAR)|(?P<n>[0-9]{1,3})[- ](?P<unit>month|year)s?)\b", re.I)
ENTITY = re.compile(r"\b(?:U\.S\. Treasury|Treasury|Federal Reserve)\b|\b(?:NYSE|NASDAQ):\s*[A-Z]{1,5}\b")
EVENT = re.compile(r"\b(default(?:ed)?|downgrade(?:d)?|upgrade(?:d)?|issuance|bankrupt(?:cy)?|earnings|dividend|(?:rais(?:e|ed|es)|hik(?:e|ed|es)|cut|lower(?:ed|s)?)\s+(?:policy\s+|interest\s+)?rates?)\b", re.I)
METRIC = re.compile(r"\b(revenue|sales|profit|earnings|income|debt|cash|yield|margin|rate|rates)\b", re.I)


def _evidence(passage, start, end):
    surface = passage["text"][start:end]
    direct = passage["normalization"] == "EXACT_UTF8_LINES"
    source_start = passage["source_start"]+len(passage["text"][:start].encode()) if direct else passage["source_start"]
    source_end = source_start+len(surface.encode()) if direct else passage["source_end"]
    return {"document_id": passage["document_id"], "passage_id": passage["passage_id"],
            "raw_sha256": passage["raw_sha256"], "source_url": passage["source_url"],
            "source_start": source_start, "source_end": source_end,
            "source_quote": surface if direct else passage["source_excerpt"],
            "normalized_char_start": start, "normalized_char_end": end, "surface_text": surface,
            "evidence_kind": "EXACT_SOURCE_BYTES" if direct else "NORMALIZED_MENTION_WITH_RAW_PASSAGE_CITATION"}


def _decimal(text):
    # No Decimal construction until syntax and magnitude have been bounded.
    if len(text) > 48 or sum(c.isdigit() for c in text) > 32:
        return None, "DIGIT_BUDGET_EXCEEDED"
    if re.fullmatch(r"[+-]?(?:nan|inf(?:inity)?)", text, re.I):
        return None, "NONFINITE_UNSUPPORTED"
    if "e" in text.lower():
        return None, "EXPONENT_SYNTAX_UNSUPPORTED"
    if not VALID_NUMBER.fullmatch(text):
        return None, "AMBIGUOUS_OR_UNSUPPORTED_NUMBER_SYNTAX"
    return Decimal(text.replace(",", "")), None


def _treasury_evidence(passage, start, end, tenor, value):
    raw_text = passage["source_excerpt"]
    pattern = r"<d:"+re.escape(tenor)+r"\b[^>]*>([^<]+)</d:"+re.escape(tenor)+r">"
    match = re.search(pattern, raw_text)
    if match is None or Decimal(match[1]) != value:
        raise ValueError("Treasury normalized financial fact differs from raw XML")
    evidence = _evidence(passage, start, end)
    evidence.update(source_start=passage["source_start"]+len(raw_text[:match.start()].encode()),
                    source_end=passage["source_start"]+len(raw_text[:match.end()].encode()),
                    source_quote=match.group(), evidence_kind="EXACT_XML_ELEMENT_FOR_NORMALIZED_YIELD")
    return evidence


def extract(passages):
    if not isinstance(passages, list) or len(passages) > 20:
        raise ValueError("financial mention extraction requires at most20retrieved passages")
    facts, tenors, entities, events, unsupported = [], [], [], [], []
    for passage in passages:
        text = passage["text"]
        if not isinstance(text, str) or len(text) > 4096:
            raise ValueError("bounded financial mention passage required")
        tenor_matches = list(TENOR.finditer(text))
        for match in tenor_matches:
            number = match["code"] or match["n"]
            unit = (match["code_unit"] or match["unit"]).lower()
            tenors.append({"tenor_months": str(Decimal(number)*(12 if unit == "year" else 1)),
                           "unit": "MONTHS", "provenance": _evidence(passage, *match.span())})
        for match in ENTITY.finditer(text):
            entities.append({"mention": match.group(), "link_state": "UNKNOWN_NO_ISSUER_AUTHORITY",
                             "provenance": _evidence(passage, *match.span())})
        for match in EVENT.finditer(text):
            # Sentence approximation is explicit; decimal points are irrelevant to cue flags.
            left = max(text.rfind(".", 0, match.start()), text.rfind(";", 0, match.start()))+1
            right_candidates = [v for v in (text.find(".", match.end()), text.find(";", match.end())) if v >= 0]
            context = text[left:min(right_candidates) if right_candidates else len(text)]
            events.append({"mention": match.group(), "occurrence": "UNKNOWN",
                "negation_cue_present": bool(re.search(r"\b(?:no|not|never|without)\b", context, re.I)),
                "conditional_cue_present": bool(re.search(r"\b(?:if|may|might|could|would|should)\b", context, re.I)),
                "provenance": _evidence(passage, *match.span())})
        for regex in (MEASUREMENT, MONEY):
            for match in regex.finditer(text):
                number = match["number"]
                # A terminal sentence full stop after a currency amount is punctuation.
                number_end = match.end("number")
                if regex is MONEY and number.endswith(".") and (number_end == len(text) or text[number_end].isspace()):
                    number = number[:-1]
                value, reason = _decimal(number)
                evidence_start, evidence_end = match.span()
                prefix_fraction = re.search(r"[0-9][0-9.,]*\s*[/⁄]\s*$", text[:match.start()])
                suffix_fraction = re.match(r"\s*[/⁄]\s*[0-9][0-9.,]*", text[match.end():])
                if prefix_fraction or suffix_fraction:
                    reason = "AMBIGUOUS_OR_UNSUPPORTED_NUMBER_SYNTAX"
                    if prefix_fraction:
                        evidence_start = prefix_fraction.start()
                    if suffix_fraction:
                        evidence_end += suffix_fraction.end()
                if reason:
                    unsupported.append({"reason": reason, "provenance": _evidence(passage, evidence_start, evidence_end)})
                    continue
                currency_mention = match["currency"].strip() if regex is MONEY else None
                if regex is MONEY:
                    unit = "UNSPECIFIED_DOLLAR" if currency_mention == "$" else "USD"
                    currency_state = "UNKNOWN" if currency_mention == "$" else "EXPLICIT_SOURCE_SYNTAX"
                else:
                    unit = "PERCENT" if match["unit"].lower() in {"percent", "%"} else "BASIS_POINTS"
                    currency_state = "NOT_APPLICABLE"
                scale = match["scale"].lower() if regex is MONEY and match["scale"] else None
                multiplier = {None: 1, "thousand": 1000, "million": 1000000, "billion": 1000000000}[scale]
                with localcontext() as context:
                    context.prec = 64
                    scaled = value*multiplier
                prefix = text[max(0, match.start()-64):match.start()]
                metrics = list(METRIC.finditer(prefix))
                metric = metrics[-1].group().lower() if metrics else "UNKNOWN"
                prior_tenors = [t for t in tenor_matches if regex is not MONEY
                    and 0 <= match.start()-t.end() <= 32
                    and not re.search(r"[.;!?]", text[t.end():match.start()])]
                tenor = prior_tenors[-1] if prior_tenors else None
                evidence = _evidence(passage, *match.span())
                if passage["normalization"] == "GIG_XML_RECORD_RENDERING_NOT_PUBLISHER_PROSE":
                    if tenor is None or tenor["code"] is None or unit != "PERCENT":
                        raise ValueError("Treasury normalized fact requires exact tenor and percent unit")
                    evidence = _treasury_evidence(passage, match.start(), match.end(), tenor.group(), value)
                facts.append({"kind": "PARSED_FINANCIAL_MENTION_NOT_VERIFIED_FACT", "metric_mention": metric,
                    "decimal_value": str(scaled), "quoted_decimal": str(value), "unit": unit,
                    "currency_state": currency_state, "currency_mention": currency_mention,
                    "scale": scale, "scale_multiplier": str(multiplier),
                    "tenor_months": str(Decimal(tenor["code"] or tenor["n"])*(12 if (tenor["code_unit"] or tenor["unit"]).lower() == "year" else 1)) if tenor else None,
                    "observation_date": passage.get("observation_date"), "truth_state": "UNKNOWN",
                    "provenance": evidence})
    return {"implementation": IMPLEMENTATION, "method": "ORIGINAL_RULE_BASED_NOT_FINBERT_OR_TRAINED_NLP",
            "scope": "RETRIEVED_PASSAGES_ONLY", "state": "AVAILABLE" if passages else "NO_MATCH",
            "facts": facts, "tenor_mentions": tenors, "entity_mentions": entities,
            "event_mentions": events, "unsupported_mentions": unsupported,
            "sentiment_state": "UNAVAILABLE", "issuer_linking_state": "UNKNOWN",
            "limitations": ["Parsed mentions are not verified financial truth or issuer facts",
                "Events are mentions only; occurrence UNKNOWN even with negation/conditional cue flags",
                "Cue flags use sentence-local lexical rules, not reliable scope or causal understanding",
                "Only explicit percent/bps/dollar syntax and thousand/million/billion scales supported",
                "Bare$currency UNKNOWN/UNSPECIFIED_DOLLAR; only explicit USD/US$syntax identifies USD",
                "Currency syntax is a parsed source mention, not verified currency jurisdiction or accounting admission",
                "Exponent, ambiguous separators, nonfinite and oversized numerics unsupported",
                "Normalized Treasury mentions cite XML elements; other normalized mentions cite raw passage"]}
