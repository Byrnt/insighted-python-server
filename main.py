from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------- DATA MODELS ----------

class ResponseItem(BaseModel):
    participantId: Optional[str] = ""
    organizationName: Optional[str] = ""
    focusConcept: Optional[str] = ""
    responseText: str
    timestamp: Optional[str] = None


# ---------- UTILITIES ----------

def compute_metrics(responses):

    texts = [
        r.responseText
        for r in responses
        if r.responseText and r.responseText.strip() != ""
    ]

    if len(texts) < 2:
        return {
            "responseCount": len(texts),
            "centroidTightness": 0,
            "meanPairwiseDistance": 0
        }

    embeddings = model.encode(texts)

    centroid = np.mean(embeddings, axis=0)

    centroid_distances = [
        np.linalg.norm(vec - centroid)
        for vec in embeddings
    ]

    centroid_tightness = float(np.mean(centroid_distances))

    similarity_matrix = cosine_similarity(embeddings)

    pairwise_distances = []

    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            pairwise_distances.append(1 - similarity_matrix[i][j])

    mean_pairwise_distance = float(np.mean(pairwise_distances))

    return {
        "responseCount": len(texts),
        "centroidTightness": round(centroid_tightness, 3),
        "meanPairwiseDistance": round(mean_pairwise_distance, 3)
    }


# ---------- ROOT ----------

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "InSightEd Analysis Server"
    }


# ---------- ANALYSIS ENDPOINT ----------

@app.post("/analyze-semantic")
async def analyze_semantic(request: Request):

    payload = await request.json()
    analysis_type = payload.get("analysisType")

    # ---------- SINGLE SESSION ----------

    if analysis_type == "single_session":

        responses = payload.get("responses", [])

        metrics = compute_metrics([
            ResponseItem(**r)
            for r in responses
        ])

        return {
            "success": True,
            "analysisType": "single_session",
            "sessionId": payload.get("sessionId"),
            "responseCount": metrics["responseCount"],
            "summary":
                f"Analyzed {metrics['responseCount']} responses "
                f"for Session {payload.get('sessionId')}. "
                f"Centroid tightness: "
                f"{metrics['centroidTightness']}; "
                f"mean pairwise distance: "
                f"{metrics['meanPairwiseDistance']}.",
            "metrics": metrics,
            "receivedAt": datetime.now().isoformat()
        }

    # ---------- SESSION COMPARISON ----------

    elif analysis_type == "session_comparison":

        setA = payload.get("setA", {})
        setB = payload.get("setB", {})

        responsesA = [
            ResponseItem(**r)
            for r in setA.get("responses", [])
        ]

        responsesB = [
            ResponseItem(**r)
            for r in setB.get("responses", [])
        ]

        metricsA = compute_metrics(responsesA)
        metricsB = compute_metrics(responsesB)

        coherence_change = round(
            metricsB["centroidTightness"] -
            metricsA["centroidTightness"],
            3
        )

        pairwise_change = round(
            metricsB["meanPairwiseDistance"] -
            metricsA["meanPairwiseDistance"],
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

        return {
            "success": True,
            "analysisType": "session_comparison",

            "summary":
                f"Compared Session {setA.get('sessionId')} "
                f"to Session {setB.get('sessionId')}. "
                f"Coherence shift: {coherence_change} "
                f"({coherence_direction}). "
                f"Homogeneity shift: {pairwise_change} "
                f"({homogeneity_direction}).",

            "comparison": {
                "sessionA": {
                    "sessionId": setA.get("sessionId"),
                    "metrics": metricsA
                },

                "sessionB": {
                    "sessionId": setB.get("sessionId"),
                    "metrics": metricsB
                },

                "changes": {
                    "coherence": {
                        "rawChange": coherence_change,
                        "direction": coherence_direction
                    },

                    "homogeneity": {
                        "rawChange": pairwise_change,
                        "direction": homogeneity_direction
                    }
                }
            },

            "receivedAt": datetime.now().isoformat()
        }

    # ---------- UNKNOWN ----------

    return {
        "success": False,
        "error": "Unknown analysisType",
        "receivedPayloadKeys": list(payload.keys())
    }