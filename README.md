# Bill Extraction API

AI-powered REST API for extracting line items from medical bills and invoices using Google Gemini 2.5 Pro Vision.

## Features

- **Gemini 2.5 Pro** - Uses Google's most accurate vision model
- **Multi-page PDF support** - Processes all pages automatically
- **Two-pass verification** - Validates extraction against detected totals
- **Fuzzy deduplication** - Removes duplicates across pages (85% name similarity)
- **Image preprocessing** - Enhances contrast and sharpness for better OCR
- **Token tracking** - Reports LLM token usage per request

## Quick Start

### 1. Setup

```bash
# Clone repository
git clone <your-repo-url>
cd bajaj

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

Create `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 3. Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Reference

### POST /extract-bill-data

Extract line items from a bill document.

**Request:**
```json
{
  "document": "https://example.com/bill.pdf"
}
```

**Response:**
```json
{
  "is_success": true,
  "token_usage": {
    "total_tokens": 2554,
    "input_tokens": 2279,
    "output_tokens": 275
  },
  "data": {
    "pagewise_line_items": [
      {
        "page_no": "1",
        "page_type": "Pharmacy",
        "bill_items": [
          {
            "item_name": "PARACETAMOL 500MG",
            "item_amount": 25.00,
            "item_rate": 2.50,
            "item_quantity": 10
          }
        ]
      }
    ],
    "total_item_count": 1
  }
}
```

### GET /health

Health check endpoint.

```json
{"status": "healthy"}
```

## Project Structure

```
bajaj/
├── app.py              # FastAPI application
├── extractor_v2.py     # Gemini extraction engine
├── models.py           # Pydantic schemas
├── test_local.py       # Local testing script
├── requirements.txt    # Dependencies
├── .env               # API key (not in repo)
└── README.md
```

## How It Works

```
Document URL
     │
     ▼
┌─────────────────┐
│ Download & Detect│
│ (PDF/PNG/JPEG)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PDF → Images    │
│ (PyMuPDF)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Pass 1: Extract │────▶│ Total Mismatch? │
│ (Gemini 2.5 Pro)│     │     (>5%)       │
└─────────────────┘     └────────┬────────┘
                                 │ Yes
                                 ▼
                        ┌─────────────────┐
                        │ Pass 2: Verify  │
                        │ & Correct       │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Deduplicate     │
                        │ Across Pages    │
                        └────────┬────────┘
                                 │
                                 ▼
                           JSON Response
```

## Page Types

| Type | Description |
|------|-------------|
| `Pharmacy` | Medicines, drugs, pharmaceutical items |
| `Bill Detail` | Services, procedures, room charges, tests |
| `Final Bill` | Summary page (totals only, no line items) |

## Testing

### Test with local files

```bash
python test_local.py
```

### Test API with curl

```bash
curl -X POST "http://localhost:8000/extract-bill-data" \
  -H "Content-Type: application/json" \
  -d '{"document": "YOUR_DOCUMENT_URL"}'
```

## Configuration

### Model Selection

Default uses `gemini-2.5-pro` for maximum accuracy. For faster/cheaper processing:

```python
# In app.py, change:
extractor = GeminiExtractorV2(GEMINI_API_KEY, use_flash=True)
```

| Model | Speed | Accuracy | Cost |
|-------|-------|----------|------|
| gemini-2.5-pro | Slower | Higher | Higher |
| gemini-2.0-flash | Faster | Good | Lower |

## Deployment

### Render.com

1. Push to GitHub
2. Create Web Service on Render
3. Connect repo, set `GEMINI_API_KEY` env var
4. Deploy

### Railway

```bash
railway init
railway up
# Set GEMINI_API_KEY in dashboard
```

## Accuracy Optimization

The system ensures accuracy through:

1. **Enhanced prompts** - Detailed extraction rules with examples
2. **Two-pass verification** - Re-checks if extracted sum differs >5% from detected total
3. **Fuzzy deduplication** - Catches near-duplicate items (85% name similarity)
4. **Page type detection** - Skips summary pages to avoid double-counting

## Requirements

- Python 3.10+
- Google Gemini API key
- ~30MB disk space

## License

MIT
