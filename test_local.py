"""
Test script for local training samples.
Usage: python test_local.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from extractor_v2 import GeminiExtractorV2

load_dotenv()


def test_file(extractor: GeminiExtractorV2, file_path: str):
    """Test extraction on a single local file."""
    print(f"\n{'='*60}")
    print(f"File: {file_path}")
    print('='*60)
    
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    
    # Convert to images
    if file_path.lower().endswith('.pdf'):
        images = extractor.pdf_to_images(file_bytes)
    else:
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(file_bytes))
        images = [extractor.preprocess_image(img)]
    
    print(f"Pages: {len(images)}")
    
    # Extract
    extractor.reset_token_count()
    pages = extractor.extract_from_images(images)
    
    # Results
    total_items = 0
    total_amount = 0.0
    
    for page in pages:
        items = page.get('bill_items', [])
        amount = sum(i.get('item_amount', 0) for i in items)
        total_items += len(items)
        total_amount += amount
        
        print(f"\nPage {page['page_no']} ({page['page_type']}): {len(items)} items, Rs. {amount:.2f}")
        for item in items[:3]:
            print(f"  - {item['item_name']}: Rs. {item['item_amount']}")
        if len(items) > 3:
            print(f"  ... +{len(items) - 3} more")
    
    print(f"\n--- TOTAL: {total_items} items, Rs. {total_amount:.2f} ---")
    print(f"Tokens: {extractor.get_token_usage()['total_tokens']}")
    
    return {'items': total_items, 'amount': total_amount}


def main():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found")
        return
    
    extractor = GeminiExtractorV2(api_key)
    
    # Find samples
    samples_dir = Path('TRAINING_SAMPLES/TRAINING_SAMPLES')
    if not samples_dir.exists():
        samples_dir = Path('TRAINING_SAMPLES')
    
    pdf_files = sorted(samples_dir.glob('*.pdf'))
    print(f"Found {len(pdf_files)} samples")
    
    # Menu
    print("\n[1] Test single file")
    print("[2] Test all files")
    print("[3] Test first 3 files")
    
    choice = input("\nChoice: ").strip()
    
    if choice == '2':
        files = pdf_files
    elif choice == '3':
        files = pdf_files[:3]
    else:
        print("\nFiles:")
        for i, f in enumerate(pdf_files, 1):
            print(f"  {i}. {f.name}")
        idx = int(input("Number: ").strip()) - 1
        files = [pdf_files[idx]] if 0 <= idx < len(pdf_files) else []
    
    results = []
    for f in files:
        try:
            results.append(test_file(extractor, str(f)))
        except Exception as e:
            print(f"ERROR: {e}")
    
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"TOTAL: {sum(r['items'] for r in results)} items, Rs. {sum(r['amount'] for r in results):.2f}")


if __name__ == '__main__':
    main()
