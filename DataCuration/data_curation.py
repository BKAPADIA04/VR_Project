import json
import csv
import os
import re
import time
import google.generativeai as genai
from PIL import Image

# Initialize Gemini API
genai.configure(api_key="AIzaSyDtPHufDr_GwIHdppj8QMWMzSUpYVRMz9A")
model = genai.GenerativeModel("gemini-2.0-flash")

def load_image_metadata(csv_path):
    metadata = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata[row['image_id']] = row['path']
    return metadata

def load_products(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_questions(product_json, image_path):
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
    
    prompt = (
        "Generate 5 Visual Question Answering (VQA) questions based on an image. Each question should have a one-word answer. "
        "Cover diverse aspects from the image and metadata. Don't just include binary (yes/no) questions."
    )

    response = model.generate_content(
        contents=[
            {"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_data}},
                {"text": json.dumps(product_json, indent=2)}
            ]}
        ]
    )
    
    return response.text

def parse_gemini_vqa_output(raw_text, item_id=None, image_id=None):
    qa_pairs = []

    # Match lines like: 1. Question? * Answer: Something
    pattern = re.findall(r'\d+\.\s*(.+?)\?\s*\*\s*Answer:\s*(.+)', raw_text, re.IGNORECASE)

    for question, answer in pattern:
        clean_question = question.strip() + "?"
        clean_answer = answer.strip()

        # Filter out empty or placeholder answers
        if clean_answer.lower() in {"answer", "none", ""}:
            continue

        qa_pairs.append({
            "question": clean_question,
            "answer": clean_answer
        })

    result = {
        "questions": qa_pairs
    }

    if item_id:
        result["item_id"] = item_id
    if image_id:
        result["image_id"] = image_id

    return result

def load_existing_outputs(output_path):
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def main():
    product_json_path = "../splits/chunk_1.json"
    metadata_csv_path = "../abo-images-small/images/metadata/images.csv"
    output_json_path = "output_1.json"
    
    products = load_products(product_json_path)
    image_metadata = load_image_metadata(metadata_csv_path)
    existing_outputs = load_existing_outputs(output_json_path)

    processed_ids = {entry["item_id"] for entry in existing_outputs}
    output_data = existing_outputs.copy()

    for product in products:
        item_id = product.get("item_id")
        if item_id in processed_ids:
            print(f"Skipping already processed item_id: {item_id}")
            continue

        image_id = product.get("main_image_id")
        image_rel_path = image_metadata.get(image_id)
        image_path = f"../abo-images-small/images/small/{image_rel_path}" if image_rel_path else None

        print(f"\nImage ID: {image_id}")
        print(f"Image Path: {image_path}")

        if image_path and os.path.exists(image_path):
            print(f"Processing item_id: {item_id}")

            try:
                result = generate_questions(product, image_path)
                print("Generated Questions:\n", result)

                structured_output = parse_gemini_vqa_output(result, item_id=item_id, image_id=image_id)

                if structured_output["questions"]:
                    output_data.append(structured_output)
                    # Save progress immediately
                    with open(output_json_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                # print("Waiting.......................................................................................................................")
                # Wait for 6 seconds to avoid hitting the API rate limit
                time.sleep(6)
                # print(f"Processed item_id: {item_id} successfully............................................................................................")

            except Exception as e:
                print(f"Error generating questions for {item_id} ({image_id}): {e}")
                break  # To resume later without repeating
        else:
            print(f"Image not found for ID: {image_id}")

    print(f"\nProgress saved to {output_json_path}")

if __name__ == "__main__":
    main()
