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


def interpret_goal_alignment(similarity_b, similarity_change, relative_alignment=None, rank=None, total=None):
    if similarity_b is None:
        return "Goal alignment could not be calculated for this region."

    if similarity_b >= 0.70:
        alignment = "high absolute alignment"
    elif similarity_b >= 0.50:
        alignment = "moderate absolute alignment"
    elif similarity_b >= 0.25:
        alignment = "modest absolute alignment"
    else:
        alignment = "very low absolute alignment"

    relative_text = ""
    if relative_alignment and rank is not None and total is not None:
        relative_text = (
            f" Relative to the detected themes in this dataset, this region is the "
            f"{relative_alignment} (rank {rank} of {total})."
        )

    if similarity_change is None:
        return (
            f"Current alignment is {alignment}."
            f"{relative_text} "
            "No before/after similarity change could be calculated."
        )

    if similarity_change >= 0.10:
        movement = "Similarity to the declared goal increased meaningfully."
    elif similarity_change >= 0.03:
        movement = "Similarity to the declared goal increased slightly."
    elif similarity_change <= -0.10:
        movement = "Similarity to the declared goal decreased meaningfully."
    elif similarity_change <= -0.03:
        movement = "Similarity to the declared goal decreased slightly."
    else:
        movement = "Similarity to the declared goal changed little."

    return f"Current alignment is {alignment}.{relative_text} {movement}"


def annotate_relative_goal_alignment(clusters):
    valid_clusters = [
        cluster for cluster in clusters
        if cluster.get("goalSimilarityB") is not None
    ]

    if not valid_clusters:
        return clusters

    ranked = sorted(
        valid_clusters,
        key=lambda cluster: cluster.get("goalSimilarityB", -999),
        reverse=True
    )

    total = len(ranked)

    for index, cluster in enumerate(ranked):
        rank = index + 1

        if total == 1:
            percentile = 100
        else:
            percentile = round(100 * (1 - (index / (total - 1))), 1)

        if rank == 1:
            relative = "strongest detected alignment"
        elif rank == 2:
            relative = "second strongest detected alignment"
        elif rank <= max(2, int(np.ceil(total / 2))):
            relative = "moderate detected alignment"
        else:
            relative = "weak detected alignment"

        cluster["goalAlignmentRank"] = rank
        cluster["goalAlignmentTotal"] = total
        cluster["goalAlignmentPercentile"] = percentile
        cluster["relativeGoalAlignment"] = relative

        cluster["goalAlignmentInterpretation"] = interpret_goal_alignment(
            cluster.get("goalSimilarityB"),
            cluster.get("goalSimilarityChange"),
            relative_alignment=relative,
            rank=rank,
            total=total
        )

    return clusters


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
        goal_similarity_change = safe_difference(goal_similarity_b, goal_similarity_a)

        goal_distance_change = safe_difference(goal_distance_b, goal_distance_a)

        prevalence_change_normalized = safe_round(
            (percentage_b - percentage_a) / 100.0
        )

        persistence_score = safe_round(
            min(percentage_a, percentage_b) / 100.0
        )

        novelty_score = 0.0
        if response_count_a == 0 and response_count_b > 0:
            novelty_score = percentage_b / 100.0

        disappearance_score = 0.0
        if response_count_a > 0 and response_count_b == 0:
            disappearance_score = percentage_a / 100.0

        resistance_score = None
        if goal_similarity_a is not None and goal_similarity_b is not None:
            prevalence_weight = (percentage_a + percentage_b) / 200.0
            resistance_score = -(goal_similarity_change or 0) * prevalence_weight

        entrenchment_score = None
        if goal_similarity_b is not None:
            persistence_component = min(percentage_a, percentage_b) / 100.0
            distance_from_goal_component = max(0.0, 1.0 - ((goal_similarity_b + 1.0) / 2.0))
            entrenchment_score = persistence_component * distance_from_goal_component

        terrain_signals = {
            "goalAlignment": goal_similarity_change,
            "prevalenceChange": prevalence_change_normalized,
            "resistance": safe_round(resistance_score),
            "novelty": safe_round(novelty_score),
            "disappearance": safe_round(disappearance_score),
            "persistence": persistence_score,
            "entrenchment": safe_round(entrenchment_score)
        }

        clusters.append({
            "clusterId": int(cluster_index + 1),
            "responseCountA": response_count_a,
            "percentageA": percentage_a,
            "responseCountB": response_count_b,
            "percentageB": percentage_b,
            "percentageChange": percentage_change,
            "prevalenceChange": prevalence_change_normalized,
            "status": cluster_status(percentage_a, percentage_b),

            "topTerms": top_terms(cluster_texts, 10),
            "representativeResponsesA": representative_for_session("A"),
            "representativeResponsesB": representative_for_session("B"),

            "spreadA": safe_round(spread_a),
            "spreadB": safe_round(spread_b),
            "spreadChange": safe_difference(spread_b, spread_a),
            "pooledThemeDispersion": safe_round(pooled_theme_sd),

            "themeCentroidShift": safe_round(theme_centroid_shift),
            "themeHaldane": safe_round(semantic_shift_haldane),
            "semanticShiftHaldane": safe_round(semantic_shift_haldane),

            "goalDistanceA": safe_round(goal_distance_a),
            "goalDistanceB": safe_round(goal_distance_b),
            "goalDistanceChange": goal_distance_change,
            "goalSimilarityA": safe_round(goal_similarity_a),
            "goalSimilarityB": safe_round(goal_similarity_b),
            "goalSimilarityChange": goal_similarity_change,

            "resistanceScore": safe_round(resistance_score),
            "noveltyScore": safe_round(novelty_score),
            "disappearanceScore": safe_round(disappearance_score),
            "persistenceScore": persistence_score,
            "entrenchmentScore": safe_round(entrenchment_score),
            "terrainSignals": terrain_signals,

            "goalAlignmentRank": None,
            "goalAlignmentTotal": None,
            "goalAlignmentPercentile": None,
            "relativeGoalAlignment": None,
            "goalAlignmentInterpretation": interpret_goal_alignment(
                goal_similarity_b,
                goal_similarity_change
            )
        })

    clusters = annotate_relative_goal_alignment(clusters)

    clusters.sort(
        key=lambda c: abs(c["percentageChange"]),
        reverse=True
    )

    return clusters


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



def suggest_theme_name_from_terms(cluster):
    terms = [
        str(item.get("term", "")).lower()
        for item in cluster.get("topTerms", [])
        if item.get("term")
    ]

    term_set = set(terms)

    theme_rules = [
        (
            "Community Support",
            {"support", "help", "helping", "others", "together", "everyone", "belong", "included", "include", "care"}
        ),
        (
            "Shared Responsibility",
            {"responsibility", "responsible", "part", "participate", "participation", "contribute", "contributing", "rules", "order", "together"}
        ),
        (
            "Collective Learning",
            {"learn", "learning", "together", "understand", "growth", "practice", "students", "class"}
        ),
        (
            "Belonging and Inclusion",
            {"belonging", "belong", "include", "included", "everyone", "safe", "welcome", "connected"}
        ),
        (
            "Leadership and Direction",
            {"leader", "leadership", "goal", "goals", "direction", "mission", "vision", "guide"}
        ),
        (
            "Process and Structure",
            {"process", "system", "structure", "order", "rules", "organized", "expectations", "clear"}
        )
    ]

    best_theme = None
    best_score = 0

    for theme_name, keywords in theme_rules:
        score = len(term_set.intersection(keywords))
        if score > best_score:
            best_theme = theme_name
            best_score = score

    if best_theme:
        return best_theme

    if len(terms) >= 2:
        return f"{terms[0].title()} / {terms[1].title()}"

    if len(terms) == 1:
        return terms[0].title()

    return "Detected Theme"


def summarize_top_terms(cluster, limit=5):
    terms = [
        str(item.get("term", "")).strip()
        for item in cluster.get("topTerms", [])[:limit]
        if item.get("term")
    ]

    return ", ".join(terms)


def strongest_goal_clusters(clusters, limit=2):
    valid_clusters = [
        cluster for cluster in clusters
        if cluster.get("goalSimilarityB") is not None
    ]

    return sorted(
        valid_clusters,
        key=lambda cluster: cluster.get("goalSimilarityB", -999),
        reverse=True
    )[:limit]


def classify_primary_outcome(changes, clusters):
    coherence_change = changes.get("coherence", {}).get("rawChange")
    homogeneity_change = changes.get("homogeneity", {}).get("rawChange")
    haldane_value = changes.get("semanticHaldane", {}).get("value")

    goal_clusters = strongest_goal_clusters(clusters, limit=2)
    strongest_goal_similarity = None
    strongest_goal_change = None

    if goal_clusters:
        strongest_goal_similarity = goal_clusters[0].get("goalSimilarityB")
        strongest_goal_change = goal_clusters[0].get("goalSimilarityChange")

    if coherence_change is None or homogeneity_change is None:
        return "Minimal Change"

    small_global_change = (
        abs(coherence_change) < 0.03
        and abs(homogeneity_change) < 0.03
        and (haldane_value is None or haldane_value < 0.2)
    )

    if small_global_change:
        if strongest_goal_similarity is not None and strongest_goal_similarity >= 0.45:
            return "Goal Convergence"
        return "Minimal Change"

    if strongest_goal_similarity is not None and strongest_goal_similarity >= 0.45:
        if strongest_goal_change is not None and strongest_goal_change >= 0.03:
            return "Goal Convergence"

        if coherence_change > 0 and homogeneity_change > 0:
            return "Differentiation"

        if coherence_change < 0 or homogeneity_change < 0:
            return "Consolidation"

    if coherence_change > 0 and homogeneity_change > 0:
        return "Fragmentation"

    if coherence_change < 0 and homogeneity_change < 0:
        return "Consolidation"

    return "Differentiation"


def estimate_confidence(metrics_a, metrics_b, clusters):
    count_a = metrics_a.get("responseCount", 0) or 0
    count_b = metrics_b.get("responseCount", 0) or 0

    if count_a < 3 or count_b < 3:
        return "Low"

    if count_a < 8 or count_b < 8:
        return "Moderate"

    goal_clusters = strongest_goal_clusters(clusters, limit=1)
    has_goal_signal = bool(goal_clusters and goal_clusters[0].get("goalSimilarityB") is not None)

    if count_a >= 12 and count_b >= 12 and has_goal_signal:
        return "High"

    return "Moderate"


def build_client_executive_summary(
    declared_goal,
    session_a_id,
    session_b_id,
    metrics_a,
    metrics_b,
    changes,
    clusters,
    term_comparison,
    haldane
):
    primary_outcome = classify_primary_outcome(changes, clusters)
    confidence = estimate_confidence(metrics_a, metrics_b, clusters)

    goal_clusters = strongest_goal_clusters(clusters, limit=2)
    lead_cluster = goal_clusters[0] if goal_clusters else (clusters[0] if clusters else {})

    theme_name = suggest_theme_name_from_terms(lead_cluster) if lead_cluster else "Detected Themes"
    top_terms_text = summarize_top_terms(lead_cluster, limit=5) if lead_cluster else ""

    coherence = changes.get("coherence", {})
    homogeneity = changes.get("homogeneity", {})

    coherence_direction = coherence.get("direction", "not calculable")
    homogeneity_direction = homogeneity.get("direction", "not calculable")
    coherence_change = coherence.get("rawChange")
    homogeneity_change = homogeneity.get("rawChange")

    distinctive_b = term_comparison.get("moreCharacteristicOfB", [])[:5]
    distinctive_a = term_comparison.get("moreCharacteristicOfA", [])[:5]

    distinctive_b_terms = ", ".join([
        item.get("term", "") for item in distinctive_b if item.get("term")
    ])

    headline_theme = theme_name
    if primary_outcome in ["Differentiation", "Consolidation", "Goal Convergence"]:
        headline = f"{primary_outcome} Around {headline_theme}"
    else:
        headline = primary_outcome

    if primary_outcome == "Differentiation":
        summary_text = (
            f"Participants moved toward more varied language around {top_terms_text or 'the detected themes'}. "
            f"Overall coherence was {coherence_direction} and overall homogeneity was {homogeneity_direction}. "
            "Because the strongest goal-aligned regions remained connected to the declared training goal, "
            "this pattern is best read as conceptual differentiation rather than simple fragmentation."
        )
    elif primary_outcome == "Consolidation":
        summary_text = (
            f"Participants moved toward a more shared language pattern around {top_terms_text or 'the detected themes'}. "
            f"Overall coherence was {coherence_direction} and overall homogeneity was {homogeneity_direction}, "
            "suggesting that responses became more anchored around common concepts."
        )
    elif primary_outcome == "Goal Convergence":
        summary_text = (
            f"Participants' language showed stronger alignment with the declared goal"
            f"{f' ({declared_goal})' if declared_goal else ''}. "
            f"The strongest aligned theme was {theme_name}, with terms such as {top_terms_text or 'the top detected terms'}."
        )
    elif primary_outcome == "Fragmentation":
        summary_text = (
            f"Participants' language became more dispersed across the semantic field. "
            f"Overall coherence was {coherence_direction} and homogeneity was {homogeneity_direction}. "
            "The current pattern should be interpreted cautiously because it may indicate multiple competing interpretations rather than a single shared movement."
        )
    else:
        summary_text = (
            "The comparison shows limited overall movement between sessions. "
            "Any detected differences should be treated as early signals rather than strong evidence of a changed shared understanding."
        )

    key_findings = []

    key_findings.append(
        f"Coherence was {coherence_direction} from {session_a_id} to {session_b_id}"
        f"{f' ({coherence_change})' if coherence_change is not None else ''}."
    )

    key_findings.append(
        f"Homogeneity was {homogeneity_direction} from {session_a_id} to {session_b_id}"
        f"{f' ({homogeneity_change})' if homogeneity_change is not None else ''}."
    )

    if haldane.get("value") is not None:
        key_findings.append(
            f"Semantic Haldane was {haldane.get('value')}, indicating {haldane.get('interpretation', 'measured semantic movement').lower()}"
        )

    if distinctive_b_terms:
        key_findings.append(
            f"Terms more characteristic of {session_b_id} included: {distinctive_b_terms}."
        )

    if goal_clusters:
        cluster_findings = []
        for cluster in goal_clusters:
            cluster_findings.append({
                "clusterId": cluster.get("clusterId"),
                "themeName": suggest_theme_name_from_terms(cluster),
                "topTerms": summarize_top_terms(cluster, limit=5),
                "goalSimilarityB": cluster.get("goalSimilarityB"),
                "goalSimilarityChange": cluster.get("goalSimilarityChange"),
                "percentageB": cluster.get("percentageB"),
                "status": cluster.get("status")
            })
    else:
        cluster_findings = []

    evidence = {
        "sessionA": {
            "sessionId": session_a_id,
            "responseCount": metrics_a.get("responseCount"),
            "centroidTightness": metrics_a.get("centroidTightness"),
            "meanPairwiseDistance": metrics_a.get("meanPairwiseDistance")
        },
        "sessionB": {
            "sessionId": session_b_id,
            "responseCount": metrics_b.get("responseCount"),
            "centroidTightness": metrics_b.get("centroidTightness"),
            "meanPairwiseDistance": metrics_b.get("meanPairwiseDistance")
        },
        "changes": changes,
        "semanticHaldane": haldane,
        "goalAlignedClusters": cluster_findings,
        "distinctiveTermsA": distinctive_a,
        "distinctiveTermsB": distinctive_b
    }

    client_caution = (
        "This summary is an interpretive reporting layer based on semantic patterns in written responses. "
        "It should be read as evidence of language movement, not as direct proof of internal belief, motivation, or organizational reality."
    )

    if confidence == "Low":
        client_caution += " Confidence is low because one or both sessions have very small sample sizes."
    elif confidence == "Moderate":
        client_caution += " Confidence is moderate; the pattern is useful, but should be confirmed with additional responses or follow-up evidence."

    return {
        "headline": headline,
        "primaryOutcome": primary_outcome,
        "summaryText": summary_text,
        "keyFindings": key_findings,
        "evidence": evidence,
        "confidence": confidence,
        "clientCaution": client_caution
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

        client_executive_summary = build_client_executive_summary(
            declared_goal,
            set_a.get("sessionId"),
            set_b.get("sessionId"),
            metrics_a,
            metrics_b,
            {
                "coherence": {
                    "rawChange": coherence_change,
                    "direction": coherence_direction
                },
                "homogeneity": {
                    "rawChange": pairwise_change,
                    "direction": homogeneity_direction
                },
                "semanticHaldane": haldane
            },
            global_semantic_clusters,
            term_comparison,
            haldane
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
            "clientExecutiveSummary": client_executive_summary,
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
                "clientExecutiveSummary": client_executive_summary,
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