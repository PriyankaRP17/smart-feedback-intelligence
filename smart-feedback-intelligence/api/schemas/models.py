from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=5000, description="Customer feedback text")
    include_absa: bool = Field(True, description="Include aspect-based sentiment analysis")
    include_entities: bool = Field(True, description="Include named entity recognition")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The delivery was extremely late but the product quality is excellent!",
                "include_absa": True,
                "include_entities": True,
            }
        }


class AspectSentiment(BaseModel):
    aspect: str
    sentiment: str
    confidence: Optional[float] = None


class EntityResult(BaseModel):
    entity_type: str
    values: List[str]


class FeedbackResponse(BaseModel):
    text: str
    sentiment: str
    sentiment_confidence: Optional[float]
    category: str
    category_confidence: Optional[float]
    urgency: str
    urgency_confidence: Optional[float]
    churn_risk: str
    churn_confidence: Optional[float]
    aspect_sentiments: Optional[Dict[str, str]]
    entities: Optional[Dict[str, List[str]]]
    processing_time_ms: float


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=500)
    include_absa: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Great product, fast delivery!",
                    "Terrible customer support, never resolved my issue.",
                ],
                "include_absa": False,
            }
        }


class BatchResponse(BaseModel):
    total: int
    results: List[FeedbackResponse]
    summary: Dict


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
