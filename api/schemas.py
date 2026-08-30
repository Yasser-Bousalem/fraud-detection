from pydantic import BaseModel, Field
from typing import Optional


class Transaction(BaseModel):
    """
    Incoming transaction to score. Only the most important features are
    required; the rest default to None (LightGBM handles missing natively).
    
    In a real system this schema would mirror the full feature set. For the
    demo we expose the key business fields and accept the rest as optional.
    """
    TransactionAmt: float = Field(..., description="Transaction amount in USD", ge=0)
    ProductCD:      Optional[str]   = Field(None, description="Product code (W/H/C/S/R)")
    card1:          Optional[int]   = Field(None, description="Card identifier 1")
    card2:          Optional[float] = Field(None)
    card4:          Optional[str]   = Field(None, description="Card network (visa/mastercard/...)")
    card6:          Optional[str]   = Field(None, description="Card type (credit/debit)")
    addr1:          Optional[float] = Field(None, description="Billing region code")
    P_emaildomain:  Optional[str]   = Field(None, description="Purchaser email domain")
    DeviceType:     Optional[str]   = Field(None)
    DeviceInfo:     Optional[str]   = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionAmt": 300.0,
                "ProductCD": "C",
                "card1": 13926,
                "card4": "visa",
                "card6": "credit",
                "addr1": 325.0,
                "P_emaildomain": "gmail.com",
                "DeviceType": "mobile",
                "DeviceInfo": "iOS Device",
            }
        }


class ReasonCode(BaseModel):
    feature:     str
    explanation: str
    shap_impact: float


class ScoreResponse(BaseModel):
    fraud_score:   float = Field(..., description="Calibrated fraud probability [0,1]")
    decision:      str   = Field(..., description="'review' or 'approve'")
    threshold:     float = Field(..., description="Operating threshold used")
    reason_codes:  list[ReasonCode]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str