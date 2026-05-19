from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from collections import Counter
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="InSightEd Analysis Server")

model = SentenceTransformer("all-MiniLM-L6-v2")


class ResponseItem(BaseModel):
    participantId: Optional[str] = ""
    organizationName: Optional[str] = ""
    focusConcept: Optional[str] = ""
    responseText: str
    timestamp: Optional[str] = None


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "to", "of", "in",
    "on", "for", "with", "as", "is", "are", "was", "were", "be", "being",
    "been", "it", "this", "that", "these", "those", "i", "we", "you",
    "they", "he", "she", "them", "his", "her", "their", "our", "my",
    "me", "us", "do", "does", "did", "so", "because", "about", "from",
    "means", "mean", "community", "commitment"
}


def tokenize(text: str):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return [
        t for t in text.split()
        if t not in STOPWORDS and len(t) > 2
    ]


def top_terms(texts, n=12):
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize(text))
    return [
        {"term": term, "count": count}
        for term, count in Counter(all_tokens).most_common(n)
    ]


def distinctive_terms(texts_a, texts_b, n=10):
    tokens_a = Counter()
    tokens_b = Counter()

    for text in texts_a:
        tokens_a.update(tokenize(text))

    for text in texts_b:
        tokens_b.update(tokenize(text))

    total_a = sum(tokens_a.values()) or 1
    total_b = sum(tokens_b.values()) or 1

    scores = []

    for term in set(tokens_a.keys()) | set(tokens_b.keys()):
        rate_a = tokens_a[term] / total_a
        rate_b = tokens_b[term] / total_b
        scores.append((term, rate_a - rate_b, tokens_a[term], tokens_b[term]))

    scores_a = sorted(scores, key=lambda x: x[1], reverse=True)[:n]
    scores_b = sorted(scores, key=lambda x: x[1])[:n]

    return {
        "moreCharacteristicOfA": [
            {
                "term": term,
                "relativeDifference": round(diff, 4),
                "countA": count_a,
                "countB": count_b
            }
            for term, diff, count_a, count_b in scores_a
            if diff > 0
        ],
        "moreCharacteristicOfB": [
            {
                "term": term,
                "relativeDifference": round(abs(diff), 4),
                "countA": count_a,
                "countB": count_b
            }
            for term, diff, count_a, count_b in scores_b
            if diff < 0
        ]
    }


def compute_metrics(responses):
    valid = [
        r for r in responses
        if r.responseText and r.responseText.strip()
    ]

    texts = [r.responseText for r in valid]

    if len(texts) == 0:
        return {
            "responseCount": 0,
            "centroidTightness": None,
            "meanPairwiseDistance": None,
            "topTerms": [],
            "outliers": []
        }

    if len(texts) == 1:
        return {
            "responseCount": 1,
            "centroidTightness": 0,
            "meanPairwiseDistance": 0,
            "topTerms": top_terms(texts),
            "outliers": []
        }

    embeddings = model.encode(texts)

    centroid = np.mean(embeddings, axis=0)

    centroid_distances = np.array([
        np.linalg.norm(vec - centroid)
        for vec in embeddings
    ])

    centroid_tightness = float(np.mean(centroid_distances))

    similarity_matrix = cosine_similarity(embeddings)

    pairwise_distances = []

    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            pairwise_distances.append(1 - similarity_matrix[i][j])

    mean_pairwise_distance = float(np.mean(pairwise_distances))

    outlier_indices = centroid_distances.argsort()[::-1][:3]

    outliers = []

    for idx in outlier_indices:
        outliers.append({
            "participantId": valid[int(idx)].participantId,
            "responseText": valid[int(idx)].responseText,
            "centroidDistance": round(float(centroid_distances[int(idx)]), 3)
        })

    return {
        "responseCount": len(texts),
        "centroidTightness": round(centroid_tightness, 3),
        "meanPairwiseDistance": round(mean_pairwise_distance, 3),
        "topTerms": top_terms(texts),
        "outliers": outliers
    }


def warnings_for(metrics, label):
    warnings = []

    n = metrics.get("responseCount", 0)

    if n < 3:
        warnings.append(
            f"{label} has fewer than 3 responses. Semantic metrics are unstable."
        )
    elif n < 8:
        warnings.append(
            f"{label} has a small sample size. Interpret comparison cautiously."
        )

    if metrics.get("meanPairwiseDistance") == 0 and n <= 1:
        warnings.append(
            f"{label} has only one usable response, so pairwise distance is not meaningful."
        )

    return warnings


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "InSightEd Analysis Server"
    }


@app.post("/analyze-semantic")
async def analyze_semantic(request: Request):
    payload = await request.json()
    analysis_type = payload.get("analysisType", "single_session")

    if analysis_type == "single_session":
        responses = [
            ResponseItem(**r)
            for r in payload.get("responses", [])
        ]

        metrics = compute_metrics(responses)

        return {
            "success": True,
            "analysisType": "single_session",
            "sessionId": payload.get("sessionId"),
            "responseCount": metrics["responseCount"],
            "summary": (
                f"Analyzed {metrics['responseCount']} responses "
                f"for Session {payload.get('sessionId')}. "
                f"Centroid tightness: {metrics['centroidTightness']}; "
                f"mean pairwise distance: {metrics['meanPairwiseDistance']}."
            ),
            "metrics": metrics,
            "interpretiveWarnings": warnings_for(metrics, "This session"),
            "receivedAt": datetime.now().isoformat()
        }

    if analysis_type == "session_comparison":
        set_a = payload.get("setA", {})
        set_b = payload.get("setB", {})

        responses_a = [
            ResponseItem(**r)
            for r in set_a.get("responses", [])
        ]

        responses_b = [
            ResponseItem(**r)
            for r in set_b.get("responses", [])
        ]

        texts_a = [r.responseText for r in responses_a if r.responseText]
        texts_b = [r.responseText for r in responses_b if r.responseText]

        metrics_a = compute_metrics(responses_a)
        metrics_b = compute_metrics(responses_b)

        coherence_change = round(
            metrics_b["centroidTightness"] - metrics_a["centroidTightness"],
            3
        )

        pairwise_change = round(
            metrics_b["meanPairwiseDistance"] - metrics_a["meanPairwiseDistance"],
            3
        )

        coherence_direction = (
            "more coherent"
            if coherence_change < 0
            else "less coherent"
            if coherence_change > 0
            else "no change"
        )

        homogeneity_direction = (
            "more homogeneous"
            if pairwise_change < 0
            else "less homogeneous"
            if pairwise_change > 0
            else "no change"
        )

        term_comparison = distinctive_terms(texts_a, texts_b)

        warnings = []
        warnings.extend(warnings_for(metrics_a, f"Session {set_a.get('sessionId')}"))
        warnings.extend(warnings_for(metrics_b, f"Session {set_b.get('sessionId')}"))

        return {
            "success": True,
            "analysisType": "session_comparison",
            "summary": (
                f"Compared Session {set_a.get('sessionId')} "
                f"to Session {set_b.get('sessionId')}. "
                f"Coherence shift: {coherence_change} "
                f"({coherence_direction}). "
                f"Homogeneity shift: {pairwise_change} "
                f"({homogeneity_direction})."
            ),
            "comparison": {
                "sessionA": {
                    "sessionId": set_a.get("sessionId"),
                    "metrics": metrics_a
                },
                "sessionB": {
                    "sessionId": set_b.get("sessionId"),
                    "metrics": metrics_b
                },
                "changes": {
                    "coherence": {
                        "rawChange": coherence_change,
                        "direction": coherence_direction,
                        "interpretation": "Lower centroid tightness means responses are closer to their semantic center."
                    },
                    "homogeneity": {
                        "rawChange": pairwise_change,
                        "direction": homogeneity_direction,
                        "interpretation": "Lower mean pairwise distance means responses are more similar to each other overall."
                    }
                },
                "distinctiveTerms": term_comparison,
                "interpretiveWarnings": warnings
            },
            "receivedAt": datetime.now().isoformat()
        }

    return {
        "success": False,
        "error": "Unknown analysisType",
        "receivedPayloadKeys": list(payload.keys())
    }