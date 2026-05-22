from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from collections import Counter
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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


def response_length_stats(texts):
    token_counts = [len(tokenize(t)) for t in texts]

    if not token_counts:
        return {
            "averageTokenCount": 0,
            "minTokenCount": 0,
            "maxTokenCount": 0
        }

    return {
        "averageTokenCount": round(float(np.mean(token_counts)), 2),
        "minTokenCount": int(min(token_counts)),
        "maxTokenCount": int(max(token_counts))
    }


def lexical_diversity(texts):
    tokens = []
    for text in texts:
        tokens.extend(tokenize(text))

    if not tokens:
        return 0

    return round(len(set(tokens)) / len(tokens), 3)


def safe_difference(value_b, value_a):
    if value_a is None or value_b is None:
        return None

    return round(value_b - value_a, 3)


def safe_round(value, digits=3):
    if value is None:
        return None

    return round(float(value), digits)


def vector_distance(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return None

    return float(np.linalg.norm(vec_a - vec_b))


def cosine_sim(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return None

    numerator = float(np.dot(vec_a, vec_b))
    denominator = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    if denominator == 0:
        return None

    return numerator / denominator


def pooled_dispersion(spread_a, spread_b):
    valid = [
        value for value in [spread_a, spread_b]
        if value is not None
    ]

    if not valid:
        return None

    return float(np.mean(valid))


def haldane_from_delta(delta, pooled_sd):
    if delta is None or pooled_sd is None or pooled_sd == 0:
        return None

    return float(delta / pooled_sd)


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


def co_occurrence_matrix(texts, top_n=12):
    terms = [item["term"] for item in top_terms(texts, top_n)]

    matrix = {
        term_a: {term_b: 0 for term_b in terms}
        for term_a in terms
    }

    for text in texts:
        tokens = set(tokenize(text))

        for term_a in terms:
            if term_a not in tokens:
                continue

            for term_b in terms:
                if term_b in tokens:
                    matrix[term_a][term_b] += 1

    return {
        "terms": terms,
        "matrix": matrix,
        "explanation": (
            "Each cell shows how many responses contained both terms. "
            "The diagonal shows how often each term appeared in a response at all."
        )
    }


def discover_semantic_clusters(embeddings, texts, valid, k=5):
    if embeddings is None or len(texts) < 2:
        return []

    n_responses = len(texts)
    n_clusters = min(k, n_responses)

    if n_clusters < 2:
        return []

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    clusters = []

    for cluster_index in range(n_clusters):
        member_indices = [
            i for i, label in enumerate(labels)
            if label == cluster_index
        ]

        if not member_indices:
            continue

        cluster_texts = [
            texts[i] for i in member_indices
        ]

        cluster_embeddings = embeddings[member_indices]
        cluster_center = centers[cluster_index]

        distances = np.linalg.norm(
            cluster_embeddings - cluster_center,
            axis=1
        )

        nearest_order = np.argsort(distances)[:3]

        representative_responses = []

        for local_idx in nearest_order:
            original_idx = member_indices[int(local_idx)]
            response = valid[original_idx]

            representative_responses.append({
                "participantId": response.participantId,
                "responseText": response.responseText,
                "distanceToClusterCenter": round(float(distances[int(local_idx)]), 3)
            })

        clusters.append({
            "clusterId": int(cluster_index + 1),
            "responseCount": len(member_indices),
            "percentage": round((len(member_indices) / n_responses) * 100, 1),
            "topTerms": top_terms(cluster_texts, 10),
            "representativeResponses": representative_responses
        })

    clusters.sort(
        key=lambda c: c["responseCount"],
        reverse=True
    )

    return clusters


def cluster_status(percentage_a, percentage_b):
    change = percentage_b - percentage_a

    if percentage_a < 5 and percentage_b >= 15:
        return "emerging"

    if percentage_a >= 15 and percentage_b < 5:
        return "declining"

    if abs(change) < 10:
        return "stable"

    if change > 0:
        return "strengthening"

    return "weakening"


def discover_global_semantic_clusters(session_a, session_b, k=5, goal_embedding=None):
    embeddings_a = session_a["embeddings"]
    embeddings_b = session_b["embeddings"]

    if embeddings_a is None or embeddings_b is None:
        return []

    texts_a = session_a["texts"]
    texts_b = session_b["texts"]

    valid_a = session_a["valid"]
    valid_b = session_b["valid"]

    if len(texts_a) == 0 or len(texts_b) == 0:
        return []

    combined_embeddings = np.vstack([embeddings_a, embeddings_b])
    combined_texts = texts_a + texts_b

    combined_valid = []

    for response in valid_a:
        combined_valid.append({
            "session": "A",
            "participantId": response.participantId,
            "responseText": response.responseText
        })

    for response in valid_b:
        combined_valid.append({
            "session": "B",
            "participantId": response.participantId,
            "responseText": response.responseText
        })

    total_count = len(combined_texts)
    n_clusters = min(k, total_count)

    if n_clusters < 2:
        return []

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(combined_embeddings)
    centers = kmeans.cluster_centers_

    clusters = []

    for cluster_index in range(n_clusters):
        member_indices = [
            i for i, label in enumerate(labels)
            if label == cluster_index
        ]

        if not member_indices:
            continue

        member_indices_a = [
            i for i in member_indices
            if combined_valid[i]["session"] == "A"
        ]

        member_indices_b = [
            i for i in member_indices
            if combined_valid[i]["session"] == "B"
        ]

        response_count_a = len(member_indices_a)
        response_count_b = len(member_indices_b)

        percentage_a = round((response_count_a / len(texts_a)) * 100, 1)
        percentage_b = round((response_count_b / len(texts_b)) * 100, 1)
        percentage_change = round(percentage_b - percentage_a, 1)

        cluster_texts = [
            combined_texts[i] for i in member_indices
        ]

        cluster_embeddings = combined_embeddings[member_indices]
        cluster_center = centers[cluster_index]

        distances = np.linalg.norm(
            cluster_embeddings - cluster_center,
            axis=1
        )

        def representative_for_session(target_session, limit=3):
            candidates = []

            for local_idx, original_idx in enumerate(member_indices):
                item = combined_valid[original_idx]

                if item["session"] != target_session:
                    continue

                candidates.append({
                    "participantId": item["participantId"],
                    "responseText": item["responseText"],
                    "distanceToClusterCenter": round(float(distances[local_idx]), 3)
                })

            candidates.sort(
                key=lambda item: item["distanceToClusterCenter"]
            )

            return candidates[:limit]

        def centroid_for_session(session_indices):
            if not session_indices:
                return None

            local_embeddings = combined_embeddings[session_indices]
            return np.mean(local_embeddings, axis=0)

        def spread_around_centroid(session_indices, centroid):
            if not session_indices or centroid is None:
                return None

            local_embeddings = combined_embeddings[session_indices]
            local_distances = np.linalg.norm(
                local_embeddings - centroid,
                axis=1
            )

            return float(np.mean(local_distances))

        centroid_a = centroid_for_session(member_indices_a)
        centroid_b = centroid_for_session(member_indices_b)

        spread_a = spread_around_centroid(member_indices_a, centroid_a)
        spread_b = spread_around_centroid(member_indices_b, centroid_b)

        pooled_theme_sd = pooled_dispersion(spread_a, spread_b)

        theme_centroid_shift = vector_distance(centroid_a, centroid_b)
        semantic_shift_haldane = haldane_from_delta(
            theme_centroid_shift,
            pooled_theme_sd
        )

        goal_distance_a = vector_distance(centroid_a, goal_embedding)
        goal_distance_b = vector_distance(centroid_b, goal_embedding)

        goal_similarity_a = cosine_sim(centroid_a, goal_embedding)
        goal_similarity_b = cosine_sim(centroid_b, goal_embedding)

        goal_distance_change = safe_difference(goal_distance_b, goal_distance_a)

        goal_movement_delta = None
        if goal_distance_a is not None and goal_distance_b is not None:
            goal_movement_delta = goal_distance_a - goal_distance_b

        goal_movement_haldane = haldane_from_delta(
            goal_movement_delta,
            pooled_theme_sd
        )

        clusters.append({
            "clusterId": int(cluster_index + 1),
            "responseCountA": response_count_a,
            "percentageA": percentage_a,
            "responseCountB": response_count_b,
            "percentageB": percentage_b,
            "percentageChange": percentage_change,
            "status": cluster_status(percentage_a, percentage_b),

            "topTerms": top_terms(cluster_texts, 10),
            "representativeResponsesA": representative_for_session("A"),
            "representativeResponsesB": representative_for_session("B"),

            "spreadA": safe_round(spread_a),
            "spreadB": safe_round(spread_b),
            "spreadChange": safe_difference(spread_b, spread_a),
            "pooledThemeDispersion": safe_round(pooled_theme_sd),

            "themeCentroidShift": safe_round(theme_centroid_shift),
            "semanticShiftHaldane": safe_round(semantic_shift_haldane),

            "goalDistanceA": safe_round(goal_distance_a),
            "goalDistanceB": safe_round(goal_distance_b),
            "goalDistanceChange": goal_distance_change,
            "goalSimilarityA": safe_round(goal_similarity_a),
            "goalSimilarityB": safe_round(goal_similarity_b),
            "goalSimilarityChange": safe_difference(goal_similarity_b, goal_similarity_a),
            "goalMovementDelta": safe_round(goal_movement_delta),
            "goalMovementHaldane": safe_round(goal_movement_haldane),
            "goalMovementInterpretation": interpret_goal_movement_haldane(goal_movement_haldane)
        })

    clusters.sort(
        key=lambda c: abs(c["percentageChange"]),
        reverse=True
    )

    return clusters


def interpret_goal_movement_haldane(value):
    if value is None:
        return "Goal movement could not be calculated for this region."

    if value >= 1.0:
        return "Major movement toward the declared goal."
    if value >= 0.5:
        return "Moderate movement toward the declared goal."
    if value >= 0.2:
        return "Small movement toward the declared goal."
    if value > -0.2:
        return "Little or no movement relative to the declared goal."
    if value > -0.5:
        return "Small movement away from the declared goal."
    if value > -1.0:
        return "Moderate movement away from the declared goal."

    return "Major movement away from the declared goal."


def compute_embeddings_and_metrics(responses):
    valid = [
        r for r in responses
        if r.responseText and r.responseText.strip()
    ]

    texts = [r.responseText for r in valid]

    if len(texts) == 0:
        return {
            "valid": valid,
            "texts": texts,
            "embeddings": None,
            "centroid": None,
            "centroidDistances": [],
            "metrics": {
                "responseCount": 0,
                "centroidTightness": None,
                "meanPairwiseDistance": None,
                "topTerms": [],
                "outliers": [],
                "averageResponseLength": response_length_stats(texts),
                "lexicalDiversity": 0,
                "coOccurrence": co_occurrence_matrix(texts),
                "semanticClusters": []
            }
        }

    embeddings = model.encode(texts)

    centroid = np.mean(embeddings, axis=0)

    centroid_distances = np.array([
        np.linalg.norm(vec - centroid)
        for vec in embeddings
    ])

    if len(texts) == 1:
        mean_pairwise_distance = 0
    else:
        similarity_matrix = cosine_similarity(embeddings)
        pairwise_distances = []

        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                pairwise_distances.append(1 - similarity_matrix[i][j])

        mean_pairwise_distance = float(np.mean(pairwise_distances))

    centroid_tightness = float(np.mean(centroid_distances))

    outlier_indices = centroid_distances.argsort()[::-1][:3]

    outliers = []

    for idx in outlier_indices:
        outliers.append({
            "participantId": valid[int(idx)].participantId,
            "responseText": valid[int(idx)].responseText,
            "centroidDistance": round(float(centroid_distances[int(idx)]), 3)
        })

    metrics = {
        "responseCount": len(texts),
        "centroidTightness": round(centroid_tightness, 3),
        "meanPairwiseDistance": round(mean_pairwise_distance, 3),
        "topTerms": top_terms(texts),
        "outliers": outliers,
        "averageResponseLength": response_length_stats(texts),
        "lexicalDiversity": lexical_diversity(texts),
        "coOccurrence": co_occurrence_matrix(texts),
        "semanticClusters": discover_semantic_clusters(
            embeddings,
            texts,
            valid,
            k=5
        )
    }

    return {
        "valid": valid,
        "texts": texts,
        "embeddings": embeddings,
        "centroid": centroid,
        "centroidDistances": centroid_distances,
        "metrics": metrics
    }


def warnings_for(metrics, label):
    warnings = []

    n = metrics.get("responseCount", 0)
    avg_len = metrics.get("averageResponseLength", {}).get("averageTokenCount", 0)

    if n < 3:
        warnings.append(
            f"{label} has fewer than 3 responses. Semantic metrics are unstable."
        )
    elif n < 8:
        warnings.append(
            f"{label} has a small sample size. Interpret comparison cautiously."
        )

    if avg_len < 15 and n < 8:
        warnings.append(
            f"{label} has both limited sample size and short responses. "
            "Longer responses may partially improve interpretive value, but cannot fully replace group size."
        )

    if metrics.get("meanPairwiseDistance") == 0 and n <= 1:
        warnings.append(
            f"{label} has only one usable response, so pairwise distance is not meaningful."
        )

    return warnings


def semantic_haldane(centroid_a, centroid_b, metrics_a, metrics_b):
    if centroid_a is None or centroid_b is None:
        return {
            "value": None,
            "centroidShift": None,
            "pooledSemanticDispersion": None,
            "interpretation": "Semantic Haldane cannot be calculated without both centroids."
        }

    dispersion_a = metrics_a.get("centroidTightness")
    dispersion_b = metrics_b.get("centroidTightness")

    if dispersion_a is None or dispersion_b is None:
        return {
            "value": None,
            "centroidShift": None,
            "pooledSemanticDispersion": None,
            "interpretation": "Semantic Haldane cannot be calculated without dispersion values."
        }

    pooled = (dispersion_a + dispersion_b) / 2
    centroid_shift = float(np.linalg.norm(centroid_b - centroid_a))

    if pooled == 0:
        value = None
    else:
        value = centroid_shift / pooled

    if value is None:
        interpretation = "Semantic Haldane cannot be interpreted because pooled dispersion is zero."
    elif value < 0.2:
        interpretation = "Small semantic shift relative to within-group dispersion."
    elif value < 0.5:
        interpretation = "Moderate semantic shift relative to within-group dispersion."
    else:
        interpretation = "Large semantic shift relative to within-group dispersion."

    return {
        "value": round(value, 3) if value is not None else None,
        "centroidShift": round(centroid_shift, 3),
        "pooledSemanticDispersion": round(pooled, 3),
        "formula": "semanticHaldane = centroidShift / pooledSemanticDispersion",
        "interpretation": interpretation,
        "caution": (
            "This is an exploratory semantic analogue to the Haldane. "
            "It normalizes semantic centroid movement by within-session semantic dispersion."
        )
    }


def embedding_map(session_a, session_b):
    embeddings_a = session_a["embeddings"]
    embeddings_b = session_b["embeddings"]

    if embeddings_a is None or embeddings_b is None:
        return {
            "points": [],
            "centroids": [],
            "method": "PCA",
            "explanation": "Not enough data for coordinate projection."
        }

    combined = np.vstack([embeddings_a, embeddings_b])

    if combined.shape[0] < 2:
        return {
            "points": [],
            "centroids": [],
            "method": "PCA",
            "explanation": "Not enough data for coordinate projection."
        }

    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)

    n_a = len(session_a["texts"])

    points = []

    for i, response in enumerate(session_a["valid"]):
        points.append({
            "session": "A",
            "participantId": response.participantId,
            "x": round(float(coords[i][0]), 4),
            "y": round(float(coords[i][1]), 4),
            "responseText": response.responseText
        })

    for j, response in enumerate(session_b["valid"]):
        idx = n_a + j
        points.append({
            "session": "B",
            "participantId": response.participantId,
            "x": round(float(coords[idx][0]), 4),
            "y": round(float(coords[idx][1]), 4),
            "responseText": response.responseText
        })

    centroid_a = np.mean(coords[:n_a], axis=0)
    centroid_b = np.mean(coords[n_a:], axis=0)

    return {
        "method": "PCA",
        "explainedVarianceRatio": [
            round(float(v), 4)
            for v in pca.explained_variance_ratio_
        ],
        "points": points,
        "centroids": [
            {
                "session": "A",
                "x": round(float(centroid_a[0]), 4),
                "y": round(float(centroid_a[1]), 4)
            },
            {
                "session": "B",
                "x": round(float(centroid_b[0]), 4),
                "y": round(float(centroid_b[1]), 4)
            }
        ],
        "explanation": (
            "PCA projects high-dimensional sentence embeddings into two dimensions for visualization. "
            "Distances are approximate and meant for visual pattern recognition, not exact measurement."
        )
    }


def evidence_based_interpretation(
    session_a_id,
    session_b_id,
    metrics_a,
    metrics_b,
    term_comparison,
    haldane
):
    a_terms = term_comparison.get("moreCharacteristicOfA", [])[:5]
    b_terms = term_comparison.get("moreCharacteristicOfB", [])[:5]

    coherence_change = safe_difference(
        metrics_b.get("centroidTightness"),
        metrics_a.get("centroidTightness")
    )

    pairwise_change = safe_difference(
        metrics_b.get("meanPairwiseDistance"),
        metrics_a.get("meanPairwiseDistance")
    )

    lines = []

    lines.append(
        f"Compared {session_a_id} to {session_b_id} using transformer-based sentence embeddings."
    )

    lines.append(
        f"{session_a_id} had centroid tightness {metrics_a.get('centroidTightness')} "
        f"and mean pairwise distance {metrics_a.get('meanPairwiseDistance')}."
    )

    lines.append(
        f"{session_b_id} had centroid tightness {metrics_b.get('centroidTightness')} "
        f"and mean pairwise distance {metrics_b.get('meanPairwiseDistance')}."
    )

    if coherence_change is None:
        lines.append(
            "Coherence shift could not be calculated because one or both sessions lacked valid centroid tightness."
        )
    elif coherence_change > 0:
        lines.append(
            f"{session_b_id} was less coherent by {coherence_change}, meaning its responses were farther from their semantic center."
        )
    elif coherence_change < 0:
        lines.append(
            f"{session_b_id} was more coherent by {abs(coherence_change)}, meaning its responses clustered closer to their semantic center."
        )
    else:
        lines.append("There was no measured coherence shift.")

    if pairwise_change is None:
        lines.append(
            "Homogeneity shift could not be calculated because one or both sessions lacked valid mean pairwise distance."
        )
    elif pairwise_change > 0:
        lines.append(
            f"{session_b_id} was less homogeneous by {pairwise_change}, meaning responses were less similar to one another overall."
        )
    elif pairwise_change < 0:
        lines.append(
            f"{session_b_id} was more homogeneous by {abs(pairwise_change)}, meaning responses were more similar to one another overall."
        )
    else:
        lines.append("There was no measured homogeneity shift.")

    if a_terms:
        terms = ", ".join([t["term"] for t in a_terms])
        lines.append(
            f"Terms more characteristic of {session_a_id}: {terms}."
        )

    if b_terms:
        terms = ", ".join([t["term"] for t in b_terms])
        lines.append(
            f"Terms more characteristic of {session_b_id}: {terms}."
        )

    if haldane.get("value") is not None:
        lines.append(
            f"Semantic Haldane: {haldane['value']}. "
            f"{haldane['interpretation']} "
            f"This means the centroid moved {haldane['value']} pooled within-group dispersions."
        )

    return {
        "plainLanguageInterpretation": " ".join(lines),
        "evidenceUsed": {
            "centroidTightnessA": metrics_a.get("centroidTightness"),
            "centroidTightnessB": metrics_b.get("centroidTightness"),
            "meanPairwiseDistanceA": metrics_a.get("meanPairwiseDistance"),
            "meanPairwiseDistanceB": metrics_b.get("meanPairwiseDistance"),
            "distinctiveTermsA": a_terms,
            "distinctiveTermsB": b_terms,
            "semanticHaldane": haldane
        },
        "caution": (
            "This interpretation is inferential. It describes patterns in written language, "
            "not direct access to belief, motivation, or organizational truth."
        )
    }


def extract_declared_goal(payload, set_a=None, set_b=None, responses_a=None, responses_b=None):
    direct_goal = (
        payload.get("declaredGoal")
        or payload.get("sessionGoal")
        or payload.get("sessionFocus")
        or payload.get("focusConcept")
    )

    if direct_goal:
        return str(direct_goal).strip()

    for source in [set_b or {}, set_a or {}]:
        possible = (
            source.get("declaredGoal")
            or source.get("sessionGoal")
            or source.get("sessionFocus")
            or source.get("focusConcept")
        )

        if possible:
            return str(possible).strip()

    for response_list in [responses_b or [], responses_a or []]:
        for response in response_list:
            if response.focusConcept and response.focusConcept.strip():
                return response.focusConcept.strip()

    return ""


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

        session = compute_embeddings_and_metrics(responses)
        metrics = session["metrics"]

        declared_goal = extract_declared_goal(
            payload,
            responses_a=responses
        )

        return {
            "success": True,
            "analysisType": "single_session",
            "sessionId": payload.get("sessionId"),
            "declaredGoal": declared_goal,
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

        declared_goal = extract_declared_goal(
            payload,
            set_a=set_a,
            set_b=set_b,
            responses_a=responses_a,
            responses_b=responses_b
        )

        goal_embedding = None
        if declared_goal:
            goal_embedding = model.encode([declared_goal])[0]

        session_a = compute_embeddings_and_metrics(responses_a)
        session_b = compute_embeddings_and_metrics(responses_b)

        texts_a = session_a["texts"]
        texts_b = session_b["texts"]

        metrics_a = session_a["metrics"]
        metrics_b = session_b["metrics"]

        coherence_change = safe_difference(
            metrics_b.get("centroidTightness"),
            metrics_a.get("centroidTightness")
        )

        pairwise_change = safe_difference(
            metrics_b.get("meanPairwiseDistance"),
            metrics_a.get("meanPairwiseDistance")
        )

        coherence_direction = (
            "not calculable"
            if coherence_change is None
            else "more coherent"
            if coherence_change < 0
            else "less coherent"
            if coherence_change > 0
            else "no change"
        )

        homogeneity_direction = (
            "not calculable"
            if pairwise_change is None
            else "more homogeneous"
            if pairwise_change < 0
            else "less homogeneous"
            if pairwise_change > 0
            else "no change"
        )

        term_comparison = distinctive_terms(texts_a, texts_b)

        haldane = semantic_haldane(
            session_a["centroid"],
            session_b["centroid"],
            metrics_a,
            metrics_b
        )

        warnings = []
        warnings.extend(warnings_for(metrics_a, f"Session {set_a.get('sessionId')}"))
        warnings.extend(warnings_for(metrics_b, f"Session {set_b.get('sessionId')}"))

        interpretation = evidence_based_interpretation(
            set_a.get("sessionId"),
            set_b.get("sessionId"),
            metrics_a,
            metrics_b,
            term_comparison,
            haldane
        )

        global_semantic_clusters = discover_global_semantic_clusters(
            session_a,
            session_b,
            k=5,
            goal_embedding=goal_embedding
        )

        return {
            "success": True,
            "analysisType": "session_comparison",
            "declaredGoal": declared_goal,
            "summary": (
                f"Compared Session {set_a.get('sessionId')} "
                f"to Session {set_b.get('sessionId')}. "
                f"Coherence shift: {coherence_change} "
                f"({coherence_direction}). "
                f"Homogeneity shift: {pairwise_change} "
                f"({homogeneity_direction}). "
                f"Semantic Haldane: {haldane.get('value')}."
            ),
            "comparison": {
                "declaredGoal": declared_goal,
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
                    },
                    "semanticHaldane": haldane
                },
                "distinctiveTerms": term_comparison,
                "globalSemanticClusters": global_semantic_clusters,
                "embeddingMap": embedding_map(session_a, session_b),
                "evidenceBasedInterpretation": interpretation,
                "interpretiveWarnings": warnings
            },
            "receivedAt": datetime.now().isoformat()
        }

    return {
        "success": False,
        "error": "Unknown analysisType",
        "receivedPayloadKeys": list(payload.keys())
    }