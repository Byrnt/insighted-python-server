from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import math
import re
from collections import Counter

app = FastAPI(title="InSightEd Analysis Server")


class SemanticResponse(BaseModel):
    participantId: Optional[str] = ""
    organizationName: Optional[str] = ""
    focusConcept: Optional[str] = ""
    responseText: str
    timestamp: Optional[str] = None


class SemanticPayload(BaseModel):
    sessionId: str
    responseCount: int
    createdAt: Optional[str] = None
    responses: List[SemanticResponse]


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "InSightEd Analysis Server"
    }


def clean_text(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    tokens = text.split()

    stopwords = {
        "the", "a", "an", "and", "or", "but", "if", "then", "to", "of", "in",
        "on", "for", "with", "as", "is", "are", "was", "were", "be", "being",
        "been", "it", "this", "that", "these", "those", "i", "we", "you",
        "they", "he", "she", "them", "his", "her", "their", "our", "my",
        "me", "us", "do", "does", "did", "so", "because", "about", "from"
    }

    return [t for t in tokens if t not in stopwords and len(t) > 1]


def cosine_similarity(counter_a: Counter, counter_b: Counter) -> float:
    shared = set(counter_a.keys()) & set(counter_b.keys())
    dot = sum(counter_a[t] * counter_b[t] for t in shared)

    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def cosine_distance(counter_a: Counter, counter_b: Counter) -> float:
    return 1.0 - cosine_similarity(counter_a, counter_b)


def centroid_counter(counters: List[Counter]) -> Counter:
    centroid = Counter()

    if not counters:
        return centroid

    for c in counters:
        centroid.update(c)

    n = len(counters)

    for key in list(centroid.keys()):
        centroid[key] = centroid[key] / n

    return centroid


def analyze_text_set(texts: List[str]):
    token_lists = [clean_text(t) for t in texts]
    counters = [Counter(tokens) for tokens in token_lists]

    n = len(counters)

    pairwise_distances = []

    for i in range(n):
        for j in range(i + 1, n):
            pairwise_distances.append(
                cosine_distance(counters[i], counters[j])
            )

    offdiag_mean = (
        sum(pairwise_distances) / len(pairwise_distances)
        if pairwise_distances else None
    )

    centroid = centroid_counter(counters)

    centroid_distances = [
        cosine_distance(c, centroid)
        for c in counters
    ]

    centroid_tightness = (
        sum(centroid_distances) / len(centroid_distances)
        if centroid_distances else None
    )

    all_tokens = [token for tokens in token_lists for token in tokens]
    top_terms = Counter(all_tokens).most_common(15)

    return {
        "n": n,
        "offdiagMeanDistance": offdiag_mean,
        "centroidTightness": centroid_tightness,
        "topTerms": [
            {"term": term, "count": count}
            for term, count in top_terms
        ],
        "responseDiagnostics": [
            {
                "index": i,
                "tokenCount": len(token_lists[i]),
                "centroidDistance": centroid_distances[i]
                if i < len(centroid_distances) else None
            }
            for i in range(n)
        ]
    }


@app.post("/analyze-semantic")
def analyze_semantic(payload: SemanticPayload):
    texts = [
        r.responseText
        for r in payload.responses
        if r.responseText and r.responseText.strip()
    ]

    metrics = analyze_text_set(texts)

    return {
        "success": True,
        "analysisType": "semantic_convergence_basic",
        "sessionId": payload.sessionId,
        "responseCount": len(texts),
        "receivedAt": datetime.now().isoformat(),
        "summary": (
            f"Analyzed {len(texts)} responses for Session {payload.sessionId}. "
            f"Centroid tightness: {metrics['centroidTightness']:.3f}; "
            f"mean pairwise distance: {metrics['offdiagMeanDistance']:.3f}."
            if metrics["centroidTightness"] is not None
            and metrics["offdiagMeanDistance"] is not None
            else f"Analyzed {len(texts)} responses for Session {payload.sessionId}."
        ),
        "metrics": metrics
    }