"""Offline, source-preserving financial news analysis and local FinBERT adapter.

Only caller-supplied, explicitly licensed articles are admitted. Company linking
is dictionary evidence, lexicon output is labelled LEXICON, and no weights or
articles are downloaded. This is research evidence, never a trading approval.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from gig.nlp.lexicon import NEGATIVE, POSITIVE, lexicon_score
from gig.timestamps import aware_timestamp

_MAX_ARTICLES = 1000
_MAX_TEXT = 100_000
_MAX_TOTAL_TEXT = 2_000_000
_EVENTS = {
    "EARNINGS": {"earnings", "eps", "revenue", "quarterly"},
    "CORPORATE_ACTION": {"dividend", "buyback", "split", "spinoff"},
    "MERGER_ACQUISITION": {"merger", "acquisition", "acquires", "takeover"},
    "LEGAL_REGULATORY": {"lawsuit", "fraud", "investigation", "regulator", "antitrust"},
    "GUIDANCE": {"guidance", "outlook", "forecast"},
    "CAPITAL_FINANCING": {"offering", "issuance", "refinancing", "bankruptcy"},
}
_TOKEN = re.compile(r"[a-z]+", re.IGNORECASE)


def _encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _time(value: Any) -> str:
    return aware_timestamp(value).isoformat()


def _text(value: Any, name: str, bound: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > bound or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be bounded text" + ("" if allow_empty else " and nonempty"))
    # Validate UTF-8 encodability without normalizing/replacing source characters.
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} contains invalid Unicode") from exc
    return value


@dataclass(frozen=True, slots=True)
class ArticleEvidence:
    payload: bytes
    evidence_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or hashlib.sha256(self.payload).hexdigest() != self.evidence_hash:
            raise ValueError("article evidence hash mismatch")

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.payload)


def admit_article(raw: Mapping[str, Any]) -> ArticleEvidence:
    """Admit exact original strings and explicit source/licensing/knowledge times.

    Required fields: source, article_id, url, license, title, body, published_at,
    available_at. Optional field symbols is provider-declared attribution.
    Acceptance records the caller's permission declaration, not legal validation.
    """
    required = {"source", "article_id", "url", "license", "title", "body", "published_at", "available_at"}
    if not isinstance(raw, Mapping) or not required.issubset(raw) or set(raw)-required-{"symbols"}:
        raise ValueError("article fields must follow the evidence schema")
    article = {name: _text(raw[name], name, bound, allow_empty=empty) for name,bound,empty in (
        ("source",512,False),("article_id",512,False),("url",4096,True),("license",4096,False),
        ("title",_MAX_TEXT,True),("body",_MAX_TEXT,True))}
    if not article["title"] and not article["body"]:
        raise ValueError("article must include original title or body")
    if len(article["title"])+len(article["body"]) > _MAX_TEXT:
        raise ValueError("combined article text exceeds declared bound")
    if article["url"] and not article["url"].lower().startswith(("https://", "http://")):
        raise ValueError("source URL must use HTTP(S) or be explicitly empty")
    article["published_at"] = _time(raw["published_at"])
    article["available_at"] = _time(raw["available_at"])
    if pd.Timestamp(article["available_at"]) < pd.Timestamp(article["published_at"]):
        raise ValueError("article cannot be available before its publication")
    symbols = raw.get("symbols", [])
    if not isinstance(symbols, list) or len(symbols) > 100 or any(not isinstance(symbol,str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9.:-]{0,31}",symbol) for symbol in symbols):
        raise ValueError("provider symbols must be a bounded list of explicit symbol strings")
    article["symbols"] = sorted(set(symbols))
    article["identity"] = _hash({"source":article["source"],"article_id":article["article_id"]})
    article["content_hash"] = _hash({"title":article["title"],"body":article["body"]})
    article["source_permission"] = "CALLER_DECLARED"
    article["schema_version"] = "1"
    payload = _encoded(article)
    return ArticleEvidence(payload,hashlib.sha256(payload).hexdigest())


def _links(text: str, aliases: Mapping[str,str]) -> list[dict[str, Any]]:
    if len(aliases) > 500:
        raise ValueError("entity alias bound exceeded")
    matches = []
    for alias,symbol in sorted(aliases.items()):
        if not isinstance(alias,str) or not alias.strip() or len(alias)>256 or not isinstance(symbol,str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9.:-]{0,31}",symbol):
            raise ValueError("entity aliases require bounded names and explicit symbols")
        for match in re.finditer(r"(?<!\w)"+re.escape(alias)+r"(?!\w)",text,re.IGNORECASE):
            if len(matches)>=20_000:
                raise ValueError("article entity match bound exceeded")
            matches.append({"symbol":symbol,"matched_text":match.group(),"start":match.start(),
                            "end":match.end(),"method":"CALLER_DICTIONARY"})
    return sorted(matches,key=lambda row:(row["start"],row["end"],row["symbol"]))


def analyze_articles(
    articles: Iterable[Mapping[str, Any] | ArticleEvidence], *, as_of: Any,
    aliases: Mapping[str,str] | None = None, finbert: LocalFinBERT | None = None,
) -> dict[str, Any]:
    """Analyze knowable articles only, preserving every admitted source string.

    Duplicate detection is exact text hashing. Syndicated text remains visible
    with attribution, but duplicate rows are excluded from aggregate sentiment.
    Event labels and entity links are explainable lexical/dictionary heuristics.
    No entity model, economic significance, or content authenticity is implied.
    """
    cutoff = pd.Timestamp(_time(as_of)); dictionary = dict(aliases or {})
    evidence: list[ArticleEvidence] = []
    total_text = 0
    for article in articles:
        if len(evidence) >= _MAX_ARTICLES:
            raise ValueError("article count bound exceeded")
        if isinstance(article,ArticleEvidence):
            decoded=article.as_dict()
            raw={key:decoded[key] for key in ("source","article_id","url","license","title","body","published_at","available_at","symbols")}
            checked=admit_article(raw)
            if checked.payload!=article.payload:
                raise ValueError("noncanonical article evidence")
            evidence.append(checked)
        else:
            evidence.append(admit_article(article))
        admitted = evidence[-1].as_dict()
        total_text += len(admitted["title"]) + len(admitted["body"])
        if total_text > _MAX_TOTAL_TEXT:
            raise ValueError("aggregate article text bound exceeded")
    evidence.sort(key=lambda item:(pd.Timestamp(item.as_dict()["available_at"]),item.evidence_hash))
    seen_content: dict[str,str] = {}; output=[]; excluded=0
    revisions: dict[tuple[str, str], ArticleEvidence] = {}
    replayed = 0
    for item in evidence:
        row = item.as_dict()
        revision = (row["identity"], row["available_at"])
        prior = revisions.get(revision)
        if prior is not None:
            if prior.payload != item.payload:
                raise ValueError("conflicting article revision at identical knowledge time")
            replayed += 1
        else:
            revisions[revision] = item
    latest: dict[str, ArticleEvidence] = {}
    for item in revisions.values():
        row = item.as_dict()
        if pd.Timestamp(row["available_at"]) <= cutoff:
            latest[row["identity"]] = item
    for item in revisions.values():
        article = item.as_dict()
        if pd.Timestamp(article["available_at"]) > cutoff:
            excluded += 1; continue
        text = article["title"]+"\n"+article["body"]
        tokens = list(_TOKEN.finditer(text)); token_set = {token.group().lower() for token in tokens}
        positive = [{"text":match.group(),"start":match.start(),"end":match.end()} for match in tokens if match.group().lower() in POSITIVE]
        negative = [{"text":match.group(),"start":match.start(),"end":match.end()} for match in tokens if match.group().lower() in NEGATIVE]
        entities = _links(text,dictionary)
        symbols = sorted(set(article["symbols"])|{link["symbol"] for link in entities})
        sentiment = {"method":"LEXICON","model_id":"gig.financial_lexicon.v1",
                     "score":lexicon_score(text),"positive_matches":positive,"negative_matches":negative,
                     "limitations":"Bag-of-words; ignores negation, context and sarcasm"}
        if finbert is not None:
            sentiment = finbert.score(text)
        event_evidence = [{"event":event,"matched_terms":sorted(terms&token_set)} for event,terms in _EVENTS.items() if terms&token_set]
        duplicate_of = seen_content.get(article["content_hash"])
        superseded_by = latest[article["identity"]].evidence_hash
        superseded_by = None if superseded_by == item.evidence_hash else superseded_by
        if superseded_by is None:
            seen_content.setdefault(article["content_hash"],item.evidence_hash)
        output.append({**article,"evidence_hash":item.evidence_hash,"duplicate_of":duplicate_of,
                       "superseded_by":superseded_by,
                       "sentiment":sentiment,"entities":entities,"linked_symbols":symbols,
                       "relevance": {"score":1. if symbols else 0.,"method":"EXPLICIT_ENTITY_MATCH"},
                       "events":event_evidence or [{"event":"UNCLASSIFIED","matched_terms":[]}],
                       "analysis_text_join":"title + newline + body; offsets are Python Unicode codepoints"})
    unique = [row for row in output if row["duplicate_of"] is None and row["superseded_by"] is None]
    return {"state":"AVAILABLE" if output else "UNAVAILABLE","as_of":cutoff.isoformat(),"articles":output,
            "future_articles_excluded":excluded,"unique_content_count":len(unique),
            "replayed_articles":replayed,
            "mean_sentiment":sum(row["sentiment"]["score"] for row in unique)/len(unique) if unique else None,
            "alias_mapping_hash":_hash(dictionary),"classification":"CALLER_SUPPLIED_NEWS",
            "trading_approved":False}


def sentiment_timeline(analysis: Mapping[str,Any], symbol: str | None = None) -> list[dict[str,Any]]:
    """Daily UTC knowledge-time sentiment; exact duplicate content counted once."""
    buckets: dict[str,list[dict[str,Any]]] = {}
    for article in analysis["articles"]:
        if (article["duplicate_of"] is not None or article.get("superseded_by") is not None
                or (symbol is not None and symbol not in article["linked_symbols"])):
            continue
        day = pd.Timestamp(article["available_at"]).date().isoformat()
        buckets.setdefault(day,[]).append(article)
    return [{"date":day,"symbol":symbol,"count":len(rows),
             "mean_sentiment":sum(row["sentiment"]["score"] for row in rows)/len(rows),
             "source_evidence_hashes":[row["evidence_hash"] for row in rows],
             "timestamp_semantics":"knowledge_time_UTC_day"} for day,rows in sorted(buckets.items())]


class NewsStore:
    """Immutable SQLite source evidence with idempotent crash/restart replay."""

    def __init__(self,path: str | Path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS articles (evidence_hash TEXT PRIMARY KEY, identity TEXT NOT NULL, available_at TEXT NOT NULL, payload BLOB NOT NULL, UNIQUE(identity,available_at))")

    def ingest(self,articles: Iterable[Mapping[str,Any]]) -> dict[str,int]:
        admitted=[]; total_text=0
        for article in articles:
            if len(admitted)>=_MAX_ARTICLES: raise ValueError("article count bound exceeded")
            admitted.append(admit_article(article))
            row=admitted[-1].as_dict();total_text+=len(row["title"])+len(row["body"])
            if total_text>_MAX_TOTAL_TEXT: raise ValueError("aggregate article text bound exceeded")
        inserted=replayed=0
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            for evidence in admitted:
                article=evidence.as_dict()
                prior=connection.execute("SELECT evidence_hash FROM articles WHERE identity=? AND available_at=?",(article["identity"],article["available_at"])).fetchone()
                if prior:
                    if prior[0]!=evidence.evidence_hash:
                        raise ValueError("conflicting article revision; preserve later knowledge timestamp")
                    replayed+=1
                else:
                    connection.execute("INSERT INTO articles VALUES (?,?,?,?)",(evidence.evidence_hash,article["identity"],article["available_at"],evidence.payload))
                    inserted+=1
        return {"inserted":inserted,"replayed":replayed}

    def snapshot(self,as_of: Any, *, aliases: Mapping[str,str] | None = None) -> dict[str,Any]:
        cutoff=pd.Timestamp(_time(as_of)); latest: dict[str,ArticleEvidence]={}
        with sqlite3.connect(self.path) as connection:
            for digest,identity,available,payload in connection.execute("SELECT evidence_hash,identity,available_at,payload FROM articles"):
                if pd.Timestamp(available)>cutoff: continue
                evidence=ArticleEvidence(bytes(payload),digest)
                decoded=evidence.as_dict()
                if decoded["identity"]!=identity or decoded["available_at"]!=available:
                    raise ValueError("article metadata does not match source evidence")
                prior=latest.get(identity)
                if prior is None or pd.Timestamp(prior.as_dict()["available_at"])<pd.Timestamp(available):
                    latest[identity]=evidence
                if len(latest)>_MAX_ARTICLES: raise ValueError("snapshot article bound exceeded; select narrower archive")
        return analyze_articles(latest.values(),as_of=cutoff,aliases=aliases)


def finbert_capability(directory: str | Path | None = None) -> dict[str,Any]:
    """Dependency/file inventory does not establish successful model loading."""
    def installed(name):
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError,ValueError):
            return False
    dependencies={"torch":installed("torch"),"transformers":installed("transformers")}
    supplied=directory is not None and Path(directory).is_dir()
    return {"state":"UNKNOWN" if supplied and all(dependencies.values()) else "UNAVAILABLE",
            "dependencies":dependencies,"local_directory_supplied":supplied,"load_verified":False,
            "downloads_enabled":False,"adapter_implemented":True,"trading_approved":False,
            "reason":"Use LocalFinBERT to validate hashes and load a caller-provided checkpoint locally"}


class LocalFinBERT:
    """CPU inference from hash-verified, caller-provided local safetensors only.

    ``finbert-provenance.json`` must declare model_id ``ProsusAI/finbert``, a
    nonempty revision, and SHA-256 digests for every local model/tokenizer file.
    The manifest's origin is caller-declared, not independently authenticated.
    Instantiation validates and loads weights locally; failure never substitutes
    lexicon output. Original text is kept outside the explicitly truncated model
    input. No pickle checkpoints or remote model IDs are accepted.
    """

    def __init__(self,directory: str | Path):
        root=Path(directory).resolve()
        manifest_path=root/"finbert-provenance.json"
        if not root.is_dir() or not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("local FinBERT directory and provenance manifest required")
        if manifest_path.stat().st_size > 131072:
            raise ValueError("FinBERT provenance manifest exceeds declared bound")
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_id")!="ProsusAI/finbert" or not isinstance(manifest.get("revision"),str) or not manifest["revision"]:
            raise ValueError("explicit FinBERT model identity and revision required")
        files=manifest.get("files")
        if not isinstance(files,dict) or len(files)>256 or not {"config.json","model.safetensors"}.issubset(files):
            raise ValueError("FinBERT config and safetensors hashes required")
        actual=set(); total=0
        for path in root.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("FinBERT checkpoint must contain flat regular files only")
            total+=path.stat().st_size
            if total>2*1024**3 or len(actual)>256:
                raise ValueError("FinBERT checkpoint exceeds declared size/file bounds")
            if path.name!="finbert-provenance.json": actual.add(path.name)
        if actual!=set(files):
            raise ValueError("every local model/tokenizer file must have a manifest digest")
        for name,digest in files.items():
            if not isinstance(name,str) or Path(name).name!=name or not isinstance(digest,str):
                raise ValueError("invalid local model manifest path/hash")
            path=root/name
            if path.is_symlink() or path.resolve().parent!=root or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError("local model file hash mismatch")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Local FinBERT requires torch and transformers; no substitution allowed") from exc
        self._torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(str(root),local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForSequenceClassification.from_pretrained(str(root),local_files_only=True,use_safetensors=True,trust_remote_code=False).to("cpu").eval()
        self.labels={int(key):str(value).lower() for key,value in self.model.config.id2label.items()}
        if (set(self.labels)!={0,1,2} or set(self.labels.values())!={"positive","negative","neutral"}
                or self.model.config.model_type!="bert"):
            raise ValueError("local checkpoint is not a supported three-label FinBERT BERT model")
        self.identity=_hash(manifest)
        self.revision=manifest["revision"]

    def score(self,text: str) -> dict[str,Any]:
        text=_text(text,"model input",_MAX_TEXT+1,allow_empty=True)
        original_count=len(self.tokenizer.encode(text,add_special_tokens=True,truncation=False))
        encoded=self.tokenizer(text,return_tensors="pt",truncation=True,max_length=512)
        with self._torch.inference_mode():
            probabilities=self._torch.softmax(self.model(**encoded).logits,dim=-1)[0].tolist()
        if len(probabilities)!=3 or any(not math.isfinite(value) for value in probabilities):
            raise ValueError("FinBERT produced invalid probabilities")
        by_label={self.labels[index]:float(value) for index,value in enumerate(probabilities)}
        return {"method":"FINBERT_LOCAL","model_id":"ProsusAI/finbert","revision":self.revision,
                "model_manifest_hash":self.identity,"origin_evidence":"CALLER_DECLARED_LOCAL",
                "probabilities":by_label,"score":by_label["positive"]-by_label["negative"],
                "model_input_truncated":original_count>512,"original_token_count":original_count,
                "max_model_tokens":512,"device":"cpu"}
