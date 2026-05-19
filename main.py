from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

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


@app.post("/analyze-semantic")
def analyze_semantic(payload: SemanticPayload):
    responses = payload.responses

    return {
        "success": True,
        "analysisType": "semantic_convergence",
        "sessionId": payload.sessionId,
        "responseCount": len(responses),
        "receivedAt": datetime.now().isoformat(),
        "summary": f"Received {len(responses)} responses for Session {payload.sessionId}.",
        "metrics": {
            "placeholder": True
        }
    }