"""Earnings-call transcript structure and cited document comparison.

Both read admitted immutable documents only and are extractive: speaker turns,
question-and-answer exchanges and forward-looking sentences are located by
fixed rules and quoted with citations; a comparison aligns the passages of two
documents and reports what was added, removed, reworded or renumbered.
Numerical and event mentions use the existing rule parser and stay cited.
Nothing is scored for sentiment or checked against the world.
"""
from __future__ import annotations

import re
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Any

from gig.financial_intelligence import nlp
from gig.financial_intelligence.research import IMPLEMENTATION, statements
from gig.financial_intelligence.store import FinancialCorpus, digest
from gig.timestamps import aware_timestamp

MAX_TURNS = 400
MAX_MENTIONS = 200
MAX_CHANGES = 400
REWORD_SIMILARITY = .5
LABEL = re.compile(r"^(?P<name>[A-Z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*){0,3})\s*"
                   r"(?:(?:[-–—,]\s*|\()(?P<role>[^:]{1,96}?)\)?)?\s*:\s*(?P<rest>.*)$", re.S)
SECTIONS = {"prepared remarks": "PREPARED_REMARKS", "presentation": "PREPARED_REMARKS",
            "opening remarks": "PREPARED_REMARKS", "question-and-answer session": "QUESTION_AND_ANSWER",
            "question and answer session": "QUESTION_AND_ANSWER", "questions and answers": "QUESTION_AND_ANSWER",
            "question-and-answer": "QUESTION_AND_ANSWER", "q&a": "QUESTION_AND_ANSWER",
            "q&a session": "QUESTION_AND_ANSWER"}
MANAGEMENT = re.compile(r"\b(chief|ceo|cfo|coo|cto|president|officer|chair(?:man|woman|person)?|founder|treasurer|"
                        r"controller|investor relations|head of|vp)\b", re.I)
FORWARD = re.compile(r"\b(expects?|expected|expecting|anticipat(?:e|es|ed|ing)|guidance|outlook|forecasts?|forecasting|"
                     r"targets?|targeting|plans? to|planning to|intends? to|will|going forward|next (?:quarter|year)|"
                     r"full[- ]year|projects?|projected|projecting)\b", re.I)
HEDGE = re.compile(r"\b(may|might|could|approximately|roughly|around|uncertain|uncertainty|subject to|"
                   r"depends? on|depending on|if)\b", re.I)
SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
NUMBER = re.compile(r"(?<![\w.])[-+(]?\$?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?%?\)?")
WORD = re.compile(r"[\w$%.,'’-]+")
LIMITS = ["A line that starts with one to four capitalized words and a colon is read as a speaker label",
          "Roles come only from the words in a speaker label; an unlabelled role stays UNKNOWN",
          "Forward-looking and hedging cues are fixed word lists; a match shows the words appear, not what they mean",
          "No language model generated or interpreted any text; document instructions are never executed"]


def _citation(passage: dict[str, Any]) -> dict[str, Any]:
    return {k: passage[k] for k in ("document_id", "passage_id", "source_start", "source_end", "raw_sha256", "source_url")}


def _role(name: str, role: str | None) -> str:
    described = f"{name} {role or ''}"
    if re.search(r"\b(operator|moderator)\b", described, re.I):
        return "OPERATOR"
    if role and re.search(r"\banalyst\b", role, re.I):
        return "ANALYST"
    if role and MANAGEMENT.search(role):
        return "MANAGEMENT"
    return "UNKNOWN"


def transcript(corpus: FinancialCorpus, document_id: str) -> dict[str, Any]:
    """Speaker turns, sections, question-and-answer exchanges and forward-looking sentences of one document."""
    document = corpus.document(document_id)
    speakers: dict[str, dict[str, Any]] = {}
    turns: list[dict[str, Any]] = []
    section, labelled, preamble = "UNLABELLED", False, 0
    mentions: list[dict[str, Any]] = []
    spoken: list[dict[str, Any]] = []
    for passage in document["passages"]:
        marker = SECTIONS.get(passage["text"].strip().strip(".:").strip().lower())
        if marker:
            section, labelled = marker, True
            continue
        label = LABEL.match(passage["text"])
        text = passage["text"]
        if label:
            name, role, text = " ".join(label["name"].split()), label["role"], label["rest"]
            known = speakers.setdefault(name, {"name": name, "role_text": None, "role": "UNKNOWN", "turns": 0, "words": 0})
            if role and known["role_text"] is None:
                known["role_text"], known["role"] = role.strip(), _role(name, role)
            elif known["role"] == "UNKNOWN":
                known["role"] = _role(name, None)
            known["turns"] += 1
            turns.append({"turn": len(turns), "speaker": name, "role": known["role"], "section": section,
                          "words": 0, "opening": text[:240], "citations": []})
        elif not turns:
            preamble += 1
            continue
        turn = turns[-1]
        words = len(WORD.findall(text))
        turn["words"] += words
        turn["citations"].append(_citation(passage))
        spoken.append(passage)
        speakers[turn["speaker"]]["words"] += words
        offset = len(passage["text"])-len(text)
        for sentence in SENTENCE.finditer(text):
            cue = FORWARD.search(sentence.group())
            if cue and len(mentions) < MAX_MENTIONS:
                quoted = sentence.group().strip()
                start = offset+sentence.start()+len(sentence.group())-len(sentence.group().lstrip())
                mentions.append({"turn": turn["turn"], "speaker": turn["speaker"], "cue": cue.group().lower(),
                                 "hedged": bool(HEDGE.search(quoted)), "has_number": bool(NUMBER.search(quoted)),
                                 "sentence": quoted[:400], "passage_text_start": start,
                                 "passage_text_end": start+len(quoted), "citation": _citation(passage),
                                 "occurrence_state": "STATED_IN_TEXT_NOT_VERIFIED"})
    # A role first stated on a later label applies to every turn of that speaker.
    for turn in turns:
        turn["role"] = speakers[turn["speaker"]]["role"]
    for mention in mentions:
        mention["role"] = turns[mention["turn"]]["role"]
    exchanges: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for turn in turns:
        if labelled and turn["section"] != "QUESTION_AND_ANSWER":
            continue
        if turn["role"] == "ANALYST":
            if current and current["analyst"] == turn["speaker"] and not current["answer_turns"]:
                current["question_turns"].append(turn["turn"])
            else:
                current = {"analyst": turn["speaker"], "question_turns": [turn["turn"]], "answer_turns": [],
                           "responders": []}
                exchanges.append(current)
        elif turn["role"] == "MANAGEMENT" and current:
            current["answer_turns"].append(turn["turn"])
            if turn["speaker"] not in current["responders"]:
                current["responders"].append(turn["speaker"])
        elif turn["role"] == "OPERATOR":
            current = None
    for exchange in exchanges:
        exchange["question_words"] = sum(turns[i]["words"] for i in exchange["question_turns"])
        exchange["answer_words"] = sum(turns[i]["words"] for i in exchange["answer_turns"])
        exchange["answered"] = bool(exchange["answer_turns"])
    total = sum(turn["words"] for turn in turns)
    by_role = {role: sum(turn["words"] for turn in turns if turn["role"] == role)
               for role in ("MANAGEMENT", "ANALYST", "OPERATOR", "UNKNOWN")}
    annotated = spoken[:20]
    parsed = nlp.extract(annotated) if annotated else {"implementation": nlp.IMPLEMENTATION, "facts": [], "event_mentions": [],
                                                       "entity_mentions": [], "unsupported_mentions": []}
    metadata = document["metadata"]
    roster = sorted(speakers.values(), key=lambda row: (-row["words"], row["name"]))
    body = {"implementation": IMPLEMENTATION, "operation": "transcript", "document_id": metadata["document_id"],
            "document_title": metadata["title"], "classification": metadata["classification"],
            "as_of": metadata["known_at"],
            "transcript_state": "AVAILABLE" if len(speakers) >= 2 else "NO_SPEAKER_TURNS_RECOGNIZED",
            "sections_labelled": labelled, "preamble_passages": preamble, "speakers": roster,
            "turn_count": len(turns), "turns": turns[:MAX_TURNS], "turns_truncated": len(turns) > MAX_TURNS,
            "words": total, "words_by_role": by_role,
            "management_word_share": f"{Decimal(by_role['MANAGEMENT'])/Decimal(total):.6f}" if total else None,
            "exchanges": exchanges, "questions_without_management_answer": sum(not e["answered"] for e in exchanges),
            "forward_looking": mentions, "forward_looking_truncated": len(mentions) == MAX_MENTIONS,
            "hedged_forward_looking": sum(m["hedged"] for m in mentions),
            "speaker_identity_state": "LABEL_TEXT_ONLY_NOT_VERIFIED", "sentiment_state": "NOT_SCORED",
            "financial_mentions": {"implementation": parsed["implementation"],
                                   "coverage": "ALL_SPEAKER_PASSAGES" if len(spoken) <= 20 else "FIRST_20_SPEAKER_PASSAGES",
                                   "speaker_passages": len(spoken), "passages_annotated": len(annotated),
                                   "facts": parsed["facts"], "event_mentions": parsed["event_mentions"],
                                   "entity_mentions": parsed["entity_mentions"],
                                   "unsupported_mentions": parsed["unsupported_mentions"],
                                   "truth_state": "UNKNOWN", "issuer_linking_state": "UNKNOWN"},
            "content_truth_state": "UNKNOWN", "generative_state": "UNAVAILABLE", "trading_approved": False,
            "limitations": LIMITS}
    return {**body, "record_id": digest(body)}


def _masked(text: str) -> str:
    return NUMBER.sub("#", text)


def _side(passage: dict[str, Any]) -> dict[str, Any]:
    return {"text": passage["text"][:600], "ordinal": passage["ordinal"], "citation": _citation(passage)}


def compare(corpus: FinancialCorpus, base_document_id: str, revised_document_id: str) -> dict[str, Any]:
    """Passage-level differences between two admitted documents, with statement value changes."""
    base, revised = corpus.document(base_document_id), corpus.document(revised_document_id)
    if base["metadata"]["document_id"] == revised["metadata"]["document_id"]:
        raise ValueError("comparison requires two different documents")
    before, after = base["passages"], revised["passages"]
    changes: list[dict[str, Any]] = []

    def removed(passage):
        changes.append({"kind": "REMOVED", "base": _side(passage), "revised": None})

    def added(passage):
        changes.append({"kind": "ADDED", "base": None, "revised": _side(passage)})

    unchanged = 0
    matcher = SequenceMatcher(None, [p["text"] for p in before], [p["text"] for p in after], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            unchanged += i2-i1
            continue
        paired = min(i2-i1, j2-j1) if tag == "replace" else 0
        for offset in range(paired):
            old, new = before[i1+offset], after[j1+offset]
            similarity = SequenceMatcher(None, WORD.findall(old["text"]), WORD.findall(new["text"]), autojunk=False).ratio()
            numeric = _masked(old["text"]) == _masked(new["text"])
            if not numeric and similarity < REWORD_SIMILARITY:
                removed(old)
                added(new)
                continue
            changes.append({"kind": "NUMERIC_ONLY" if numeric else "REWORDED", "base": _side(old), "revised": _side(new),
                            "similarity": f"{similarity:.6f}", "numbers_before": NUMBER.findall(old["text"]),
                            "numbers_after": NUMBER.findall(new["text"])})
        for passage in before[i1+paired:i2]:
            removed(passage)
        for passage in after[j1+paired:j2]:
            added(passage)
    first, second = statements(corpus, base_document_id), statements(corpus, revised_document_id)
    earlier = {(fact["item"], fact["period"]): fact for fact in first["facts"]}
    restated = []
    for fact in second["facts"]:
        prior = earlier.get((fact["item"], fact["period"]))
        if prior and prior["decimal_value"] != fact["decimal_value"]:
            restated.append({"item": fact["item"], "period": fact["period"], "base": prior["decimal_value"],
                             "revised": fact["decimal_value"],
                             "difference": str(Decimal(fact["decimal_value"])-Decimal(prior["decimal_value"])),
                             "base_citation": prior["citation"], "revised_citation": fact["citation"],
                             "kind": "DIFFERENCE_BETWEEN_PARSED_MENTIONS_NOT_A_VERIFIED_RESTATEMENT"})
    counts = {kind: sum(change["kind"] == kind for change in changes)
              for kind in ("ADDED", "REMOVED", "REWORDED", "NUMERIC_ONLY")}
    stamps = [base["metadata"]["known_at"], revised["metadata"]["known_at"]]
    body = {"implementation": IMPLEMENTATION, "operation": "compare",
            "base_document_id": base["metadata"]["document_id"], "revised_document_id": revised["metadata"]["document_id"],
            "base_title": base["metadata"]["title"], "revised_title": revised["metadata"]["title"],
            "as_of": max(stamps, key=aware_timestamp),
            "classification": "SYNTHETIC" if "SYNTHETIC" in (base["metadata"]["classification"],
                                                              revised["metadata"]["classification"]) else "HISTORICAL",
            "base_passages": len(before), "revised_passages": len(after), "unchanged_passages": unchanged,
            "counts": counts, "identical": not changes, "changes": changes[:MAX_CHANGES],
            "changes_truncated": len(changes) > MAX_CHANGES, "statement_value_changes": restated,
            "alignment": "ORDERED_EXACT_PASSAGE_MATCHING_THEN_POSITIONAL_PAIRING",
            "comparison_kind": "EXTRACTIVE_TEXT_DIFFERENCE_NOT_AN_INTERPRETATION",
            "content_truth_state": "UNKNOWN", "generative_state": "UNAVAILABLE", "trading_approved": False,
            "limitations": ["Passages are aligned in order by exact text, then paired by position inside each changed run",
                            "A moved passage is reported as removed and added",
                            "A pair below one half word similarity is reported as a removal and an addition",
                            "Statement value changes compare parsed mentions, not verified issuer restatements",
                            "No language model generated or interpreted any text"]}
    return {**body, "record_id": digest(body)}
