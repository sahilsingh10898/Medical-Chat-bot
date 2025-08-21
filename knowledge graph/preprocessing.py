import os
import re
import glob
import pandas as pd
from typing import Tuple
from bs4 import BeautifulSoup

def _strip_html_and_whitespace(text: str) -> str:
    """Removes HTML tags and excess whitespace."""
    no_html = re.sub(r"<[^>]*>", "", str(text))
    return no_html.replace("\xa0", " ").strip()

def _is_blank_like(value) -> bool:
    """Checks if a value is semantically empty."""
    if pd.isna(value): return True
    cleaned = _strip_html_and_whitespace(str(value))
    return cleaned.lower() in {"", "na", "n/a", "none", "null", "-", "--"}

def _clean_with_placeholder(value, placeholder: str) -> str:
    """Returns placeholder if value is blank-like, otherwise cleans it."""
    return placeholder if _is_blank_like(value) else _strip_html_and_whitespace(str(value))


def normalize_duration(value: str) -> str:
    """Normalizes duration field to HH:MM:SS format."""
    if _is_blank_like(value): return "00:00:00"
    lower_text = re.sub(r"<[^>]*>", "", str(value)).lower()
    match = re.search(r"(?:(\d+)\s*min(?:ute)?s?)?\s*(?:(\d+)\s*sec(?:ond)?s?)?", lower_text)
    if match:
        mins, secs = match.groups()
        minutes = int(mins) if mins else 0
        seconds = int(secs) if secs else 0
        minutes += seconds // 60
        seconds %= 60
        return f"00:{minutes:02d}:{seconds:02d}"
    return "12:59:59"  # fallback for malformed inputs

def normalize_age(value: str) -> str:
    """Normalizes age to 'X Years, YY Months'."""
    if _is_blank_like(value): return "Not Available"
    text = _strip_html_and_whitespace(str(value)).lower()
    match = re.search(r"(\d+)\s*y(?:ear)?s?\b.*(\d+)\s*m(?:onth)?s?\b", text)
    if match:
        years, months = map(int, match.groups())
        years += months // 12
        months %= 12
        return f"{years} Years, {months:02d} Months"
    match = re.search(r"(\d+)\s*y(?:ear)?s?\b", text)
    if match: return f"{int(match.group(1))} Years, 00 Months"
    match = re.search(r"(\d+)\s*m(?:onth)?s?\b", text)
    if match:
        total_months = int(match.group(1))
        years, months = divmod(total_months, 12)
        return f"{years} Years, {months:02d} Months"
    if text.isdigit(): return f"{int(text)} Years, 00 Months"
    return "Not Available"

def split_nurse_and_phc(value: str) -> Tuple[str, str]:
    """Splits 'Nurse Name' into nurse and PHC."""
    if _is_blank_like(value):
        return ("Nr.Not Available", "Not Applicable")
    value = value.strip()
    if value.lower().startswith("nr.camp"):
        return ("Nr.Camp", "Not Applicable")
    match = re.match(r"^(?:Nr\.)?\s*(.*?)\s+([A-Z\s]+)$", value)
    if match:
        nurse_name, phc = match.groups()
        return (nurse_name.strip() or "Nr.Not Available", phc.strip())
    return (value, "")

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to clean and normalize a clinical dataframe."""
    
    # Split nurse + PHC
    if "Nurse Name" in df.columns:
        split_cols = df["Nurse Name"].apply(lambda x: pd.Series(split_nurse_and_phc(x), index=["nurse_name", "phc"]))
        insert_at = df.columns.get_loc("Nurse Name")
        df.insert(insert_at, "nurse_name", split_cols["nurse_name"])
        df.insert(insert_at + 1, "phc", split_cols["phc"])
        df.drop(columns=["Nurse Name"], inplace=True)

    # Normalize duration and age
    if "Duration" in df.columns:
        df["Duration"] = df["Duration"].apply(normalize_duration)
    if "Age" in df.columns:
        df["Age"] = df["Age"].apply(normalize_age)

    # Clean key text fields
    text_cols = [
        "Doctor Name", "Specialization", "Chief Complaints", "Complaints",
        "Diagnosis", "Medicines", "Counseling", "Investigations",
        "Visit", "Status", "Gender"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _clean_with_placeholder(v, "Not Available"))

    # Clean location fields
    loc_cols = ["Country", "State", "District"]
    for col in loc_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: pd.NA if _is_blank_like(v) else _strip_html_and_whitespace(v))
            df[col] = df[col].ffill().bfill()

    return df
def run_cleaning_process(input_folder_path: str, output_folder_path: str):
    """
    Cleans all CSV files from input folder and writes cleaned versions to output folder.

    Args:
        input_folder_path (str): Path to folder containing raw CSV files.
        output_folder_path (str): Path to folder where cleaned files will be written.
    """
    # Validate and prepare output directory
    os.makedirs(output_folder_path, exist_ok=True)
    print(f"Output will be saved in: {output_folder_path}")

    # Find all CSVs
    csv_files = glob.glob(os.path.join(input_folder_path, '*.csv'))
    if not csv_files:
        print(f" No CSV files found in '{input_folder_path}'. Please check the path.")
        return

    print(f" Found {len(csv_files)} file(s) to process.")
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"Processing: {file_name}...")
        try:
            df = pd.read_csv(file_path, low_memory=False)
            cleaned_df = process_dataframe(df)

            output_filename = f"{os.path.splitext(file_name)[0]}_updated.csv"
            output_filepath = os.path.join(output_folder_path, output_filename)
            cleaned_df.to_csv(output_filepath, index=False)
            print(f" Saved cleaned file to {output_filepath}")
        except Exception as e:
            print(f" Error processing {file_name}: {e}")

    print("\n Data cleaning complete.")