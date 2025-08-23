import os
import re
import glob
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from huggingface_hub import login


def authenticate_huggingface(token: str):
    try:
        login(token=token)
        print(" Hugging Face authentication successful.")
    except Exception as e:
        print(f" Authentication failed. Error: {e}")


def load_biomedical_ner_model(model_name: str = "d4data/biomedical-ner-all"):
    try:
        device = 0 if torch.cuda.is_available() else -1
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
        print(f" Model '{model_name}' loaded on {'GPU' if device == 0 else 'CPU'}.")
        return ner_pipeline
    except Exception as e:
        print(f" Failed to load NER model. Error: {e}")
        return None


corrections = {"moths": "months", "weak": "week"}


def clean_text(text):
    """Fix common OCR or transcription errors."""
    if not isinstance(text, str):
        return ""
    for wrong, right in corrections.items():
        text = re.sub(rf"\b{wrong}\b", right, text, flags=re.IGNORECASE)
    return text


def extract_symptom_phrases_modified(text, ents=None, min_score=0.4):
    LOCALIZABLE_SYMPTOMS = {
        "pain",
        "swelling",
        "numbness",
        "discomfort",
        "tingling",
        "cramps",
    }

    text = clean_text(text)
    if not text.strip():
        return []

    ents = ents or ner(text)
    ents = [e for e in ents if e["score"] >= min_score]
    ents = sorted(ents, key=lambda x: x["start"])

    consumed = [False] * len(ents)
    final_phrases = []
    i = 0

    while i < len(ents):
        if consumed[i]:
            i += 1
            continue

        ent = ents[i]
        word = ent["word"].strip().lower()
        label = ent["entity_group"].lower()

        # Rule 1: Multi-word symptoms
        if label == "sign_symptom" and " " in word:
            final_phrases.append(word)
            consumed[i] = True
            i += 1
            continue

        # Rule 2: Date + condition (e.g., 12 week pregnancy)
        if label == "date":
            phrase = [word]
            consumed[i] = True
            j = i + 1
            while j < len(ents) and not consumed[j]:
                next_word = ents[j]["word"].strip().lower()
                next_label = ents[j]["entity_group"].lower()
                if next_label in {"disease_disorder"} or next_word in {
                    "pregnancy",
                    "gestation",
                    "trimester",
                }:
                    phrase.append(next_word)
                    consumed[j] = True
                    j += 1
                else:
                    break
            final_phrases.append(" ".join(phrase))
            i = j
            continue

        # Rule 3: Diagnostic procedure with modifier
        if label == "diagnostic_procedure":
            modifier_words = []
            j = i - 1
            while (
                j >= 0
                and not consumed[j]
                and ents[j]["entity_group"].lower() == "detailed_description"
            ):
                modifier_words.insert(0, ents[j]["word"].strip().lower())
                consumed[j] = True
                j -= 1
            final_phrases.append(" ".join(modifier_words + [word]))
            consumed[i] = True
            i += 1
            continue

        # Rule 4: Localizable symptoms with anatomical context
        if label == "sign_symptom" and word in LOCALIZABLE_SYMPTOMS:
            consumed[i] = True
            consecutive_anatomy = []
            j = i + 1
            while (
                j < len(ents)
                and not consumed[j]
                and ents[j]["entity_group"].lower() == "biological_structure"
            ):
                consecutive_anatomy.append(ents[j]["word"].strip().lower())
                j += 1
            if len(consecutive_anatomy) >= 2:
                for k in range(len(consecutive_anatomy)):
                    consumed[i + 1 + k] = True
                final_phrases.append(f"{word} in {' and '.join(consecutive_anatomy)}")
                i = j
                continue
            elif (
                i + 1 < len(ents)
                and not consumed[i + 1]
                and ents[i + 1]["entity_group"].lower() == "biological_structure"
            ):
                anatomy = ents[i + 1]["word"].strip().lower()
                phrase_parts = [anatomy, word]
                consumed[i + 1] = True
                if (
                    i + 2 < len(ents)
                    and not consumed[i + 2]
                    and ents[i + 2]["entity_group"].lower() == "detailed_description"
                ):
                    phrase_parts.insert(0, ents[i + 2]["word"].strip().lower())
                    consumed[i + 2] = True
                final_phrases.append(" ".join(phrase_parts))
            else:
                final_phrases.append(word)
            i += 1
            continue

        # Rule 5: Generic sign_symptom or structure
        if label in {"sign_symptom", "biological_structure"}:
            final_phrases.append(word)
            consumed[i] = True
        i += 1

    return final_phrases


def extract_symptom_from_history(html_str):
    """Extracts HPI section from rich-text HTML."""
    if pd.isna(html_str):
        return ""
    soup = BeautifulSoup(html_str, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    match = re.search(
        r"History\s+Of\s+Present\s+Illness\s*:\s*(.*?)\s*(?:Personal History|Past Medical|Family History|Current And Recent Medications|Medical Allergies|Additional Notes)\s*:",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip().rstrip(",") if match else ""


def fill_symptoms(row):
    """Fills the 'symptoms' column using chief complaints or HPI."""
    if pd.notna(row["Chief Complaints"]) and row["Chief Complaints"].strip():
        return row["Chief Complaints"].strip()
    return row["Patient History Cleaned"] if row["Patient History Cleaned"] else ""


def process_symptom_files(input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    print(f" Output folder created: {output_folder}")

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print(f" No CSV files found in '{input_folder}'")
        return

    print(f"🔍 Found {len(csv_files)} files.")
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f" Processing: {file_name}...")
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df["Patient History Cleaned"] = df["Patient History"].apply(
                extract_symptom_from_history
            )
            df["symptoms"] = df.apply(fill_symptoms, axis=1)
            df.dropna(subset=["symptoms"], inplace=True)
            df = df[df["symptoms"].str.strip() != ""]

            if df.empty:
                print(f" No valid symptoms found in {file_name}. Skipping.")
                continue

            tqdm.pandas(desc=f"NER on {file_name}")
            df["cleaned_symptoms"] = df["symptoms"].progress_apply(
                extract_symptom_phrases_modified
            )

            output_filename = f"{os.path.splitext(file_name)[0]}_cleaned.csv"
            df.to_csv(os.path.join(output_folder, output_filename), index=False)
            print(f" Saved: {output_filename}")
        except Exception as e:
            print(f" Error in {file_name}: {e}")

    print("\n All files processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NER-based symptom extraction")
    parser.add_argument("--input_folder", required=True, help="Input folder with cleaned CSVs")
    parser.add_argument("--output_folder", required=True, help="Output folder for NER-cleaned files")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (optional)")

    args = parser.parse_args()

    access_token = args.hf_token or os.getenv("HF_TOKEN")
    authenticate_huggingface(access_token)

    ner = load_biomedical_ner_model()
    process_symptom_files(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
    )
