"""
Bill Extraction Engine using Google Gemini Vision API.

Features:
- Gemini 2.5 Pro model (default) for high accuracy
- Two-pass extraction with verification
- Fuzzy deduplication across pages
- Image preprocessing for better OCR
"""

import google.generativeai as genai
import json
import re
import requests
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


def similar(a: str, b: str) -> float:
    """Calculate string similarity ratio (0-1)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


class GeminiExtractorV2:
    """
    Extracts line items from bill/invoice images using Gemini Vision.
    
    Args:
        api_key: Google Gemini API key
        use_flash: If True, use faster gemini-2.0-flash. Default uses gemini-2.5-pro.
    """
    
    def __init__(self, api_key: str, use_flash: bool = False):
        genai.configure(api_key=api_key)
        model_name = 'gemini-2.0-flash' if use_flash else 'gemini-2.5-pro'
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        logger.info(f"Initialized with model: {model_name}")
    
    def reset_token_count(self):
        """Reset token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage across all API calls."""
        return {
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens
        }
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhance image for better text extraction."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast and sharpen
        image = ImageEnhance.Contrast(image).enhance(1.3)
        image = image.filter(ImageFilter.SHARPEN)
        
        # Resize if too large (saves tokens)
        max_dim = 2048
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def download_document(self, url: str) -> Tuple[bytes, str]:
        """
        Download document and detect file type.
        
        Returns:
            Tuple of (file_bytes, file_type)
        """
        logger.info(f"Downloading: {url[:80]}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        file_bytes = response.content
        content_type = response.headers.get('content-type', '').lower()
        
        # Detect file type
        if 'pdf' in content_type or url.lower().endswith('.pdf') or file_bytes[:4] == b'%PDF':
            file_type = 'pdf'
        elif file_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            file_type = 'png'
        elif file_bytes[:2] == b'\xff\xd8':
            file_type = 'jpeg'
        else:
            file_type = 'png'
        
        logger.info(f"File type: {file_type}")
        return file_bytes, file_type
    
    def pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Convert PDF to list of preprocessed images."""
        import fitz  # PyMuPDF
        
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(self.preprocess_image(img))
        
        pdf_doc.close()
        logger.info(f"Converted PDF: {len(images)} page(s)")
        return images
    
    def _create_extraction_prompt(self) -> str:
        """Build the extraction prompt with examples."""
        return """You are a bill/invoice analyzer. Extract ALL line items precisely.

## EXTRACT ✓
- Medicines, drugs, tablets, injections
- Medical procedures, surgeries, tests
- Room/bed charges, nursing charges
- Consumables (syringes, IV sets, bandages)
- Doctor/consultation fees

## SKIP ✗
- Subtotals, Grand Totals, Net Amounts
- Tax rows (GST/CGST/SGST)
- Discount summary rows
- Headers and labels

## FIELDS
| Field | Description |
|-------|-------------|
| item_name | Exact name from bill |
| item_amount | Final line amount |
| item_rate | Unit price (or item_amount if not shown) |
| item_quantity | Units (or 1 if not shown) |

## EXAMPLES

"CROCIN 650MG | 20 | 5.50 | 110.00"
→ {"item_name": "CROCIN 650MG", "item_amount": 110.0, "item_rate": 5.5, "item_quantity": 20}

"MRI BRAIN ............ 8500.00"
→ {"item_name": "MRI BRAIN", "item_amount": 8500.0, "item_rate": 8500.0, "item_quantity": 1}

## PAGE TYPES
- "Pharmacy": Medicines/drugs
- "Bill Detail": Services, procedures, tests
- "Final Bill": Only totals (return empty bill_items)

## OUTPUT FORMAT (JSON only, no markdown)
{
  "page_type": "Pharmacy",
  "bill_items": [{"item_name": "...", "item_amount": 0.0, "item_rate": 0.0, "item_quantity": 0.0}],
  "detected_total": 0.0
}

detected_total: Grand total if visible, else 0."""
    
    def _create_verification_prompt(self, items: List[Dict], detected_total: float) -> str:
        """Build prompt for verification pass."""
        items_json = json.dumps(items, indent=2)
        items_sum = sum(i.get('item_amount', 0) for i in items)
        
        return f"""Verify this bill extraction:

EXTRACTED ITEMS:
{items_json}

DETECTED TOTAL: {detected_total}
CALCULATED SUM: {items_sum:.2f}

Check for:
1. Missing items
2. Duplicate items  
3. Incorrect amounts

Return JSON:
{{
  "items_to_add": [{{"item_name": "...", "item_amount": 0.0, "item_rate": 0.0, "item_quantity": 0.0}}],
  "items_to_remove": ["item_name"],
  "corrections": [{{"item_name": "...", "correct_amount": 0.0}}],
  "verification_notes": "..."
}}

If correct, return empty arrays."""
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown blocks."""
        text = text.strip()
        
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try finding JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"Failed to parse JSON: {text[:200]}")
        return {"page_type": "Bill Detail", "bill_items": [], "detected_total": 0}
    
    def _clean_bill_items(self, items: List[Dict]) -> List[Dict]:
        """Validate and deduplicate items within a page."""
        cleaned = []
        seen = set()
        
        for item in items:
            try:
                name = str(item.get("item_name", "")).strip()
                amount = float(item.get("item_amount", 0))
                rate = float(item.get("item_rate", amount))
                quantity = float(item.get("item_quantity", 1))
                
                if not name or amount <= 0:
                    continue
                
                key = f"{name.lower()}|{amount}"
                if key in seen:
                    continue
                seen.add(key)
                
                cleaned.append({
                    "item_name": name,
                    "item_amount": round(amount, 2),
                    "item_rate": round(rate, 2),
                    "item_quantity": round(quantity, 2)
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid item skipped: {item}, error: {e}")
        
        return cleaned
    
    def _extract_single_page(self, image: Image.Image, page_num: int) -> Dict[str, Any]:
        """
        Extract items from a single page.
        Runs verification pass if extracted total differs significantly from detected total.
        """
        prompt = self._create_extraction_prompt()
        
        response = self.model.generate_content(
            [prompt, image],
            generation_config=genai.GenerationConfig(temperature=0.05, max_output_tokens=8192)
        )
        
        # Track tokens
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
            self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
        
        extracted = self._parse_json_response(response.text)
        items = self._clean_bill_items(extracted.get("bill_items", []))
        detected_total = float(extracted.get("detected_total", 0))
        
        # Run verification if total mismatch > 5%
        items_sum = sum(i.get('item_amount', 0) for i in items)
        if detected_total > 0 and items and abs(items_sum - detected_total) / detected_total > 0.05:
            logger.info(f"Page {page_num}: Running verification (sum={items_sum:.2f}, detected={detected_total:.2f})")
            items = self._verify_and_correct(image, items, detected_total)
        
        return {
            "page_no": str(page_num),
            "page_type": extracted.get("page_type", "Bill Detail"),
            "bill_items": items,
            "detected_total": detected_total
        }
    
    def _verify_and_correct(self, image: Image.Image, items: List[Dict], detected_total: float) -> List[Dict]:
        """Second pass to verify and correct extraction errors."""
        try:
            prompt = self._create_verification_prompt(items, detected_total)
            response = self.model.generate_content(
                [prompt, image],
                generation_config=genai.GenerationConfig(temperature=0.05, max_output_tokens=4096)
            )
            
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
                self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            if not response.candidates:
                return items
            
            try:
                response_text = response.text
            except ValueError:
                return items
            
            corrections = self._parse_json_response(response_text)
            
            # Apply removals
            to_remove = set(n.lower() for n in corrections.get("items_to_remove", []))
            if to_remove:
                items = [i for i in items if i["item_name"].lower() not in to_remove]
            
            # Apply amount corrections
            amount_fixes = {c["item_name"].lower(): c["correct_amount"] 
                          for c in corrections.get("corrections", [])}
            for item in items:
                if item["item_name"].lower() in amount_fixes:
                    item["item_amount"] = amount_fixes[item["item_name"].lower()]
            
            # Add missing items
            for new_item in corrections.get("items_to_add", []):
                cleaned = self._clean_bill_items([new_item])
                items.extend(cleaned)
            
            if corrections.get("verification_notes"):
                logger.info(f"Verification: {corrections['verification_notes']}")
                
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
        
        return items
    
    def _deduplicate_across_pages(self, pages_data: List[Dict]) -> List[Dict]:
        """Remove duplicate items across pages using fuzzy matching (85% similarity)."""
        if len(pages_data) <= 1:
            return pages_data
        
        global_items: Dict[str, Tuple[str, Dict]] = {}
        
        for page in pages_data:
            if page.get('page_type') == 'Final Bill':
                page['bill_items'] = []
                continue
            
            unique_items = []
            for item in page.get('bill_items', []):
                name, amount = item['item_name'], item['item_amount']
                is_dup = False
                
                for _, (existing_page, existing_item) in global_items.items():
                    similarity = similar(name, existing_item['item_name'])
                    same_amount = abs(amount - existing_item['item_amount']) < 0.01
                    
                    if (similarity > 0.85 and same_amount) or similarity > 0.95:
                        is_dup = True
                        logger.info(f"Dedup: '{name}' (pg {page['page_no']}) = '{existing_item['item_name']}' (pg {existing_page})")
                        break
                
                if not is_dup:
                    global_items[f"{name.lower()}|{amount}"] = (page['page_no'], item)
                    unique_items.append(item)
            
            page['bill_items'] = unique_items
        
        return pages_data
    
    def extract_from_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Extract bill data from multiple page images.
        
        Returns:
            List of page data with bill_items per page.
        """
        all_pages = []
        
        for page_num, image in enumerate(images, start=1):
            logger.info(f"Processing page {page_num}/{len(images)}")
            try:
                page_data = self._extract_single_page(image, page_num)
                all_pages.append(page_data)
                logger.info(f"Page {page_num}: {len(page_data.get('bill_items', []))} items")
            except Exception as e:
                logger.error(f"Page {page_num} failed: {e}")
                all_pages.append({"page_no": str(page_num), "page_type": "Bill Detail", "bill_items": []})
        
        # Deduplicate and clean
        all_pages = self._deduplicate_across_pages(all_pages)
        for page in all_pages:
            page.pop('detected_total', None)
        
        return all_pages
    
    def verify_extraction(self, pages_data: List[Dict], expected_total: Optional[float] = None) -> Dict:
        """
        Verify extraction accuracy against expected total.
        
        Returns:
            Report with total_items, calculated_total, and accuracy_percentage.
        """
        total_items = sum(len(p.get('bill_items', [])) for p in pages_data)
        calculated_total = sum(
            item.get('item_amount', 0) 
            for p in pages_data 
            for item in p.get('bill_items', [])
        )
        
        report = {"total_items": total_items, "calculated_total": round(calculated_total, 2)}
        
        if expected_total:
            diff = abs(calculated_total - expected_total)
            accuracy = max(0, 1 - diff / expected_total) if expected_total > 0 else 0
            report.update({
                "expected_total": expected_total,
                "difference": round(diff, 2),
                "accuracy_percentage": round(accuracy * 100, 2)
            })
        
        return report
    
    def extract_from_url(self, url: str) -> Dict[str, Any]:
        """
        Main entry point: Download document and extract all bill data.
        
        Args:
            url: URL to PDF or image document
            
        Returns:
            Dict with pagewise_line_items, total_item_count, and token_usage.
        """
        self.reset_token_count()
        
        file_bytes, file_type = self.download_document(url)
        
        if file_type == 'pdf':
            images = self.pdf_to_images(file_bytes)
        else:
            image = Image.open(BytesIO(file_bytes))
            images = [self.preprocess_image(image)]
        
        pages_data = self.extract_from_images(images)
        total_items = sum(len(p.get("bill_items", [])) for p in pages_data)
        
        return {
            "pagewise_line_items": pages_data,
            "total_item_count": total_items,
            "token_usage": self.get_token_usage()
        }
