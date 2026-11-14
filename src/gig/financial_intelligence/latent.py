"""Original bounded corpus-local TF-IDF/SVD retrieval, never pretrained semantics.

Pure numbers and ISO dates are excluded by a fixed feature policy; no top-k
vocabulary or passage truncation is permitted. BM25 retains these distinctions.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from gig.financial_intelligence.retrieval import SYNONYMS, rank, rank_candidates, tokens
from gig.service.serialization import dumps

IMPLEMENTATION = "gig.corpus-local-lsa.v1"
HYBRID = "gig.bm25-lsa-rrf.v1"
FEATURE_POLICY = "ALPHABETIC_BEARING_TOKENS_ONLY_NO_PURE_NUMBERS_OR_DATES.v1"
MAX_PASSAGES, MAX_TERMS, MAX_DIMENSIONS = 256, 512, 16
RRF_K = 60
EPSILON = 1e-12
LIMITATIONS = [
    "CORPUS_LOCAL_LSA; not pretrained neural semantics, LLM or paraphrase understanding",
    "Pure-number/date distinctions excluded from LSA; hybrid BM25 preserves lexical distinctions",
    "Similarity reflects corpus co-occurrence, not verified financial relevance or confidence",
    "Homogeneous passages can tie; unseen vocabulary and zero projections cannot be understood",
]


def _identity(value):
    import hashlib
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def settings(mode, dimensions):
    if not isinstance(mode, str) or mode not in {"bm25", "latent", "hybrid"}:
        raise ValueError("explicit supported financial retrieval mode required")
    if type(dimensions) is not int or not 1 <= dimensions <= MAX_DIMENSIONS:
        raise ValueError("LSA dimensions require an exact integer from1to16")
    if mode == "bm25" and dimensions != MAX_DIMENSIONS:
        raise ValueError("LSA dimensions require latent or hybrid mode")


def _features(text):
    return Counter(t for t in tokens(text) if any("a" <= c <= "z" for c in t))


def _matrix(passages):
    if not isinstance(passages, list) or not 1 <= len(passages) <= MAX_PASSAGES:
        raise ValueError("LSA requires1to256visible passages; no silent truncation")
    counts = [_features(p["text"]) for p in passages]
    vocabulary = sorted(set().union(*(set(c) for c in counts)))
    if not vocabulary or len(vocabulary) > MAX_TERMS:
        raise ValueError("LSA requires1to512retained terms; no silent truncation")
    idf = [math.log((1+len(counts))/(1+sum(t in c for c in counts)))+1 for t in vocabulary]
    matrix = np.array([[(1+math.log(c[t]))*weight if c[t] else 0.0
                       for t, weight in zip(vocabulary, idf, strict=True)] for c in counts])
    norms = np.linalg.norm(matrix, axis=1)
    matrix = matrix/np.where(norms > 0, norms, 1)[:, None]
    corpus_identity = _identity([p["passage_id"] for p in passages])
    statistics_identity = _identity([dict(sorted(c.items())) for c in counts])
    return matrix, vocabulary, idf, corpus_identity, statistics_identity


def _unit_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1)
    return matrix/np.where(norms > EPSILON, norms, 1)[:, None]


def fit(passages, dimensions=MAX_DIMENSIONS):
    """Fit only supplied visible revisions. No assets, network, random seed or cache."""
    settings("latent", dimensions)
    matrix, vocabulary, idf, corpus, statistics = _matrix(passages)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    effective_rank = int(np.count_nonzero(singular > singular[0]*EPSILON))
    retained = min(dimensions, effective_rank)
    components = vt[:retained].copy()
    # Canonical sign improves auditability; replay uses saved arrays, not refits.
    for row in components:
        if row[int(np.argmax(np.abs(row)))] < 0:
            row *= -1
    vectors = _unit_rows(matrix @ components.T)
    return {"implementation": IMPLEMENTATION, "model_kind": "CORPUS_LOCAL_LSA",
            "numpy_version": np.__version__, "feature_policy": FEATURE_POLICY,
            "tf_idf_policy": "LOG_TF_SMOOTH_IDF_L2.v1", "requested_dimensions": dimensions,
            "dimensions": retained, "effective_rank": effective_rank,
            "passage_count": len(passages), "vocabulary_count": len(vocabulary),
            "corpus_identity": corpus, "statistics_identity": statistics,
            "training_scope": "LATEST_VISIBLE_REVISIONS_ONLY",
            "bounds": {"passages": MAX_PASSAGES, "retained_terms": MAX_TERMS, "dimensions": MAX_DIMENSIONS},
            "vocabulary": vocabulary, "idf": idf,
            "components": components.tolist(), "document_vectors": vectors.tolist(),
            "singular_values": singular[:retained].tolist(), "limitations": LIMITATIONS}


def validate(model, passages, dimensions):
    """Bind a saved fit to exact visible input, without refitting the SVD."""
    matrix, vocabulary, idf, corpus, statistics = _matrix(passages)
    if (model.get("implementation") != IMPLEMENTATION or model.get("feature_policy") != FEATURE_POLICY
            or model.get("requested_dimensions") != dimensions or model.get("vocabulary") != vocabulary
            or model.get("idf") != idf or model.get("corpus_identity") != corpus
            or model.get("statistics_identity") != statistics):
        raise ValueError("LSA saved model differs from visible corpus/statistics")
    size = model.get("dimensions")
    if type(size) is not int or not 1 <= size <= dimensions:
        raise ValueError("LSA saved dimension invalid")
    components = np.asarray(model["components"], dtype=float)
    vectors = np.asarray(model["document_vectors"], dtype=float)
    if (components.shape != (size, len(vocabulary)) or vectors.shape != (len(passages), size)
            or not np.isfinite(components).all() or not np.isfinite(vectors).all()
            or not np.allclose(components @ components.T, np.eye(size), atol=1e-10, rtol=0)
            or not np.allclose(_unit_rows(matrix @ components.T), vectors, atol=1e-10, rtol=0)):
        raise ValueError("LSA saved projection integrity failed")


def metadata(model):
    return {k: v for k, v in model.items() if k not in {"vocabulary", "idf", "components", "document_vectors", "singular_values"}}


def retrieve(passages, query, limit, *, mode, dimensions=MAX_DIMENSIONS, expand=False, model=None):
    settings(mode, dimensions)
    _, terms = rank([], query, limit, expand=expand)
    model = fit(passages, dimensions) if model is None else model
    validate(model, passages, dimensions)
    query_counts = _features(query)
    if expand:
        query_counts.update(SYNONYMS[t] for t in list(query_counts) if t in SYNONYMS)
    vocabulary = model["vocabulary"]
    vector = np.array([(1+math.log(query_counts[t]))*weight if query_counts[t] else 0.0
                       for t, weight in zip(vocabulary, model["idf"], strict=True)])
    projected = vector @ np.asarray(model["components"]).T
    norm = float(np.linalg.norm(projected))
    known = sorted(set(query_counts).intersection(vocabulary))
    state = "UNKNOWN_VOCABULARY" if not known else ("ZERO_PROJECTION" if norm <= EPSILON else "AVAILABLE")
    dense = []
    if norm > EPSILON:
        scores = np.asarray(model["document_vectors"]) @ (projected/norm)
        for passage, score in zip(passages, scores, strict=True):
            if score > EPSILON:
                dense.append({**passage, "score": min(1.0, float(score)), "score_kind": "CORPUS_LOCAL_LSA_COSINE_NOT_FINANCIAL_CONFIDENCE"})
        dense.sort(key=lambda p: (-p["score"], p["document_id"], p["ordinal"]))
    if mode == "latent":
        result = dense[:limit]
    else:
        lexical, _ = rank_candidates(passages, query, expand=expand)
        candidates = {}
        for component, rows in (("bm25", lexical), ("latent", dense)):
            for position, row in enumerate(rows, 1):
                item = candidates.setdefault(row["passage_id"], {**row, "score": 0.0,
                    "bm25_score": None, "latent_score": None, "bm25_rank": None, "latent_rank": None,
                    "score_kind": "RRF_RANK_FUSION_NOT_FINANCIAL_CONFIDENCE"})
                item[component+"_score"] = row["score"]
                item[component+"_rank"] = position
                item["score"] += 1/(RRF_K+position)
        result = sorted(candidates.values(), key=lambda p: (-p["score"], p["document_id"], p["ordinal"]))[:limit]
    evidence = {"model_id": _identity(model), "model": metadata(model), "latent_query_state": state,
                "known_query_terms": known, "unknown_query_terms": sorted(set(query_counts)-set(vocabulary)),
                "retrieval_state": "AVAILABLE" if result else "NO_MATCH"}
    if mode == "hybrid":
        evidence["fusion"] = {"implementation": HYBRID, "method": "RECIPROCAL_RANK_FUSION", "rank_constant": RRF_K,
                              "component_weights": {"bm25": 1, "latent": 1}}
    return result, terms, model, evidence
