import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

# Example: Using transformers pipeline for VQA (replace with your model as needed)
# from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    # Load model and processor, move model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    # model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)
    # model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    base_model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base",
        device_map='auto'
    )
    # model = PeftModel.from_pretrained(base_model, "blip_lora_adapter2").to(device)
    model = PeftModel.from_pretrained(base_model, "bk45/blip-vqa-finetuned").to(device)
    model.eval()

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = processor(image, question, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(**encoding)
                answer = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        except Exception as e:
            print(f"Error processing image : {e}")
            answer = "error"
        # Ensure answer is one word and in English (basic post-processing)
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
    
#data - /Users/bhavyakapadia/Desktop/Semester_6/VR/Project/inference-setup/data
#metadata.cvs - /Users/bhavyakapadia/Desktop/Semester_6/VR/Project/inference-setup/data/metadata.csv