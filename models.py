"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Literal


class BillItem(BaseModel):
    """Single line item from a bill."""
    item_name: str = Field(..., description="Item name as shown in bill")
    item_amount: float = Field(..., description="Final amount after discounts")
    item_rate: float = Field(..., description="Price per unit")
    item_quantity: float = Field(..., description="Number of units")


class PageLineItems(BaseModel):
    """Line items for a single page."""
    page_no: str
    page_type: Literal["Bill Detail", "Final Bill", "Pharmacy"]
    bill_items: List[BillItem] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """LLM token consumption."""
    total_tokens: int
    input_tokens: int
    output_tokens: int


class ExtractionData(BaseModel):
    """Extracted bill data."""
    pagewise_line_items: List[PageLineItems]
    total_item_count: int


class SuccessResponse(BaseModel):
    """Successful API response."""
    is_success: bool = True
    token_usage: TokenUsage
    data: ExtractionData


class ErrorResponse(BaseModel):
    """Error API response."""
    is_success: bool = False
    message: str


class ExtractionRequest(BaseModel):
    """API request body."""
    document: str = Field(..., description="URL to document (PDF or image)")
