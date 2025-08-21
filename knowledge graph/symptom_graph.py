import os
import re
import math
import ast
import numpy as np
import pandas as pd
from pathlib import Path

def robust_read_csv(path: Path) -> pd.DataFrame:
    """Reads CSVs with potential encoding or delimiter issues."""
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


MALFORMED_SYMPTOM_FIXES = {
    "zziness": "dizziness", "ziness": "dizziness", "zzing": "dizziness",
    "ching": "itching", "che": "headache", "kness": "weakness", "ingling": "tingling",
    "burn": "burning", "yspepsia": "dyspepsia", "yspnea": "dyspnea",
    "yers": "years", "yaer": "year", "yeras": "years", "2o": "20",
    "2o-days": "20 days", "mont": "month", "moth": "month", "mo": "month",
    "n-hand": "hand"
}

USELESS_PATTERNS = [
    r"^\d+(\s+)?$", r"^\d+[a-z]*$", r"^\d+[a-z-]+$",
    r"^\d+\s*(day|week|month|year)s?$", r"^\d+\s*(rd|th)\s+day$"
]

DURATION_UNITS = {"day", "days", "week", "weeks", "month", "months", "year", "years", "y", "d"}
JUNK_SUFFIXES = {"yer", "yr", "yrs", "mints", "says", "dat", "daysweak", "mnts", "mnt", "mon", "say"}


def repair_malformed_symptom(sym: str) -> str:
    if not sym.startswith("##"):
        return sym
    sym = sym.lstrip("#").strip()
    for broken, fixed in MALFORMED_SYMPTOM_FIXES.items():
        if sym.startswith(broken):
            return sym.replace(broken, fixed, 1)
    return sym


def is_useless_symptom_only_num(sym: str) -> bool:
    sym = sym.lower().strip()
    for pat in USELESS_PATTERNS:
        if re.match(pat, sym):
            return True
    return False


def is_duration_only(phrase: str) -> bool:
    phrase = re.sub(r"[^\w\s]", "", phrase.strip().lower())
    tokens = phrase.split()
    return all(t.isdigit() or t in DURATION_UNITS for t in tokens) and "pregnancy" not in tokens


def is_date_junk_symptom(sym: str) -> bool:
    sym = sym.lower().strip()
    tokens = sym.split()
    if is_duration_only(sym): return True
    if len(tokens) == 2 and tokens[0].isdigit() and tokens[1] in JUNK_SUFFIXES: return True
    if any(sym.endswith(sfx) for sfx in JUNK_SUFFIXES): return True
    if re.search(r"\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}", sym): return True
    return False


def is_junk_symptom(sym: str) -> bool:
    sym = sym.strip().lower()
    if sym.startswith("##"): sym = sym.lstrip("#").strip()
    if not sym or len(sym) < 4: return True
    if re.fullmatch(r"^[\W_]+$", sym): return True
    if re.fullmatch(r"\d+(\.\d+)?", sym): return True
    if re.match(r"^\d+(\s+\d+)+$", sym): return True
    if "day of birth" in sym or sym in ["3d leg pain", "5 daysfever"]: return True
    if re.fullmatch(r"(\d+\s*)+((day|week|month|months|year|years)\b)", sym): return True
    if re.match(r"^\d+(\s+\d+)?\s*(day|week|month|year)s?\b", sym): return True
    if re.match(r"^\d+\s+(last|since|in|for)\s+\d*", sym): return True
    if re.search(r"\bs\s+day\b", sym) or "weak sob" in sym: return True
    return False


LIST_SPLIT_RE = re.compile(r"\s*,\s*|\s*;\s*|\s*\|\s*")


def parse_listish(cell) -> list:
    if pd.isna(cell): return []
    s = str(cell).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    return [p.strip() for p in LIST_SPLIT_RE.split(s) if p.strip()]


def slugify(prefix: str, text: str, maxlen: int = 120) -> str:
    if isinstance(text, list): text = " ".join(str(x) for x in text)
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(text).strip().lower()).strip("-")
    return f"{prefix}:{s[:maxlen]}" if s else f"{prefix}:unknown"


def is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def extract_unit_from_name(name: str) -> tuple[str, str]:
    m = re.search(r"^(.*?)[\(\[]\s*([^)^\]]+)\s*[\)\]]\s*$", name.strip())
    return (m.group(1).strip(), m.group(2).strip()) if m else (name.strip(), "")


def classify_test_kind(col: str) -> str:
    l = col.lower()
    if any(k in l for k in ["x-ray", "cxr", "ultrasound", "ct ", "mri"]): return "imaging"
    if any(k in l for k in ["rapid", "rbs", "hba1c", "crp", "urine", "wbc", "spiro"]): return "lab"
    return "test"


def extract_symptom_kg(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    symptom_nodes = set()
    has_symptom_edges = []

    for idx, row in df.iterrows():
        id_no = str(row.get("I.D_No", "")).strip()
        visit = str(row.get("Visit", "")).strip()
        case_id = f"CASE:{id_no}:{visit}" if id_no and visit else f"CASE:row{idx}"

        raw_symptoms = row.get("cleaned_symptoms", "")
        if raw_symptoms is None or (isinstance(raw_symptoms, float) and pd.isna(raw_symptoms)):
            continue

        if isinstance(raw_symptoms, (list, np.ndarray)):
            symptoms = raw_symptoms.tolist() if isinstance(raw_symptoms, np.ndarray) else raw_symptoms
        elif isinstance(raw_symptoms, str):
            symptoms = parse_listish(raw_symptoms)
        else:
            symptoms = []

        for s in symptoms:
            s = str(s).strip()
            if not s: continue
            if s.startswith("##"): s = repair_malformed_symptom(s)
            if any([is_useless_symptom_only_num(s), is_junk_symptom(s), is_duration_only(s), is_date_junk_symptom(s)]): continue

            vid = slugify("SYMPTOM", s)
            symptom_nodes.add((vid, s))
            has_symptom_edges.append({"src_vid": vid, "dst_vid": case_id})

    symptoms_df = pd.DataFrame(sorted(list(symptom_nodes)), columns=["vid", "name"])
    has_symptom_df = pd.DataFrame(has_symptom_edges).drop_duplicates()
    return symptoms_df, has_symptom_df

def process_file_to_kg(input_path: Path, out_root: Path):
    print(f"\n📄 Processing file: {input_path.name}")
    df = robust_read_csv(input_path)

    # Detect available columns
    diagnosis_col = "Diagnosis" if "Diagnosis" in df.columns else None
    symptoms_col = "cleaned_symptoms" if "cleaned_symptoms" in df.columns else None
    duration_col = "Duration" if "Duration" in df.columns else None
    invest_col = "Investigations" if "Investigations" in df.columns else None
    meds_col = "Medicines" if "Medicines" in df.columns else None

    patient_nodes, disease_nodes, test_nodes, medicine_nodes = [], set(), set(), set()
    has_diag, underwent_test, treated_with = [], [], []

    def get_case_vid(row) -> str:
        id_no = str(row.get("I.D_No", "")).strip()
        visit = str(row.get("Visit", "")).strip()
        return f"CASE:{id_no}:{visit}" if id_no and visit else f"CASE:row{int(row.name)}"

    for idx, row in df.iterrows():
        case_id = get_case_vid(row)
        patient_nodes.append({
            "vid": case_id,
            "id_no": row.get("I.D_No", ""),
            "visit": row.get("Visit", ""),
            "gender": row.get("Gender", ""),
            "age": row.get("Age", ""),
            "date_of_exam": row.get("Date of Examination", "")
        })

        if diagnosis_col and not is_missing(row.get(diagnosis_col)):
            dx = str(row[diagnosis_col]).strip()
            dv = slugify("DISEASE", dx)
            disease_nodes.add((dv, dx))
            has_diag.append({"src_vid": case_id, "dst_vid": dv})

        if invest_col and not is_missing(row.get(invest_col)):
            for t in parse_listish(row[invest_col]):
                tname, tunit = extract_unit_from_name(t)
                tv = slugify("TEST", tname)
                test_nodes.add((tv, tname, classify_test_kind(tname)))
                underwent_test.append({"src_vid": case_id, "dst_vid": tv, "unit": tunit})

        if meds_col and not is_missing(row.get(meds_col)):
            for m in parse_listish(row[meds_col]):
                mv = slugify("MED", m)
                medicine_nodes.add((mv, m))
                treated_with.append({"src_vid": case_id, "dst_vid": mv})

    # Add symptoms via external extractor
    symptoms_df, has_symptom_df = extract_symptom_kg(df)

    # Final DataFrames
    patients_df = pd.DataFrame(patient_nodes).drop_duplicates("vid")
    diseases_df = pd.DataFrame(sorted(list(disease_nodes)), columns=["vid", "name"])
    tests_df = pd.DataFrame(sorted(list(test_nodes)), columns=["vid", "name", "kind"])
    medicines_df = pd.DataFrame(sorted(list(medicine_nodes)), columns=["vid", "name"])
    has_diag_df = pd.DataFrame(has_diag)
    underwent_df = pd.DataFrame(underwent_test)
    treated_with_df = pd.DataFrame(treated_with)

    # Output path
    base_name = input_path.stem.replace('_updated_cleaned', '')
    out_dir = out_root / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f" Writing to: {out_dir}")
    patients_df.to_csv(out_dir / "patient_cases.csv", index=False)
    symptoms_df.to_csv(out_dir / "symptoms.csv", index=False)
    diseases_df.to_csv(out_dir / "diseases.csv", index=False)
    tests_df.to_csv(out_dir / "tests.csv", index=False)
    medicines_df.to_csv(out_dir / "medicines.csv", index=False)
    has_symptom_df.to_csv(out_dir / "has_symptom.csv", index=False)
    has_diag_df.to_csv(out_dir / "has_diagnosis.csv", index=False)
    underwent_df.to_csv(out_dir / "underwent_test.csv", index=False)
    treated_with_df.to_csv(out_dir / "treated_with.csv", index=False)


def run_kg_builder(input_folder_path: str, output_root_path: str):
    input_folder = Path(input_folder_path)
    output_root = Path(output_root_path)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f" Reading input files from: {input_folder}")
    print(f" Output directory for KG files: {output_root}")

    final_files = list(input_folder.glob('*_updated_cleaned.csv'))
    if not final_files:
        print(f" No *_updated_cleaned.csv files found in '{input_folder}'")
        return

    print(f" Found {len(final_files)} file(s) to process.")
    for file_path in final_files:
        process_file_to_kg(file_path, output_root)

    print("\n Knowledge Graph construction complete.")
    