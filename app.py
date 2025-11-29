"""
Bill Extraction API

REST API for extracting line items from bills/invoices using Gemini Vision.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from models import (
    ExtractionRequest, SuccessResponse, TokenUsage,
    ExtractionData, PageLineItems, BillItem
)
from extractor_v2 import GeminiExtractorV2

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log startup/shutdown."""
    if GEMINI_API_KEY:
        logger.info("API key loaded")
    else:
        logger.error("GEMINI_API_KEY not found!")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Bill Extraction API",
    description="Extract line items from bills using Gemini Vision AI",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check."""
    return {"status": "healthy", "message": "Bill Extraction API"}


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


@app.post("/extract-bill-data")
async def extract_bill_data(request: ExtractionRequest):
    """
    Extract line items from a bill document.
    
    Accepts PDF or image URL, returns structured line items.
    """
    try:
        logger.info(f"Processing: {request.document[:80]}...")
        
        if not GEMINI_API_KEY:
            return JSONResponse(
                status_code=500,
                content={"is_success": False, "message": "API key not configured"}
            )
        
        # Extract using Gemini 2.5 Pro
        extractor = GeminiExtractorV2(GEMINI_API_KEY)
        result = extractor.extract_from_url(request.document)
        
        # Build response
        pagewise_items = [
            PageLineItems(
                page_no=page["page_no"],
                page_type=page.get("page_type", "Bill Detail"),
                bill_items=[
                    BillItem(**item) for item in page.get("bill_items", [])
                ]
            )
            for page in result["pagewise_line_items"]
        ]
        
        response = SuccessResponse(
            is_success=True,
            token_usage=TokenUsage(**result["token_usage"]),
            data=ExtractionData(
                pagewise_line_items=pagewise_items,
                total_item_count=result["total_item_count"]
            )
        )
        
        logger.info(f"Extracted {result['total_item_count']} items")
        return response
        
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"is_success": False, "message": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
