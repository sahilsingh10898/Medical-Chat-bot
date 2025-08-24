import os
import argparse
from jinja2 import Environment, FileSystemLoader
import glob

def get_schema_metadata(file_name):
    # Return metadata depending on file name
    mapping = {
        "patient_cases.csv": {
            "type": "vertex", "tag": "Case", "vid_index": 0, "props": [
                {"name": "case_id", "type": "string"},
                {"name": "patient_age", "type": "int"},
                {"name": "gender", "type": "string"},
                {"name": "visit", "type": "string"},
            ]
        },
        "symptoms.csv": {
            "type": "vertex", "tag": "Symptom", "vid_index": 0, "props": [
                {"name": "name", "type": "string"},
            ]
        },
        "diseases.csv": {
            "type": "vertex", "tag": "Disease", "vid_index": 0, "props": [
                {"name": "name", "type": "string"},
            ]
        },
        "medicines.csv": {
            "type": "vertex", "tag": "Medicine", "vid_index": 0, "props": [
                {"name": "name", "type": "string"},
            ]
        },
        "tests.csv": {
            "type": "vertex", "tag": "Test", "vid_index": 0, "props": [
                {"name": "name", "type": "string"},
            ]
        },
        "has_symptom.csv": {
            "type": "edge", "edge": "HAS_SYMPTOM", "src_index": 0, "dst_index": 1,"props": [] 
        },
        "has_diagnosis.csv": {
            "type": "edge", "edge": "HAS_DIAGNOSIS", "src_index": 0, "dst_index": 1
        },
        "treated_with.csv": {
            "type": "edge", "edge": "TREATED_WITH", "src_index": 0, "dst_index": 1,"props": [] 
        },
        "underwent_test.csv": {
            "type": "edge", "edge": "UNDERWENT_TEST", "src_index": 0, "dst_index": 1,"props": [
                {"name": "unit", "type": "string", "index": 2}
            ] 
        }
    }
    return mapping.get(file_name)

def render_import_yaml(dataset_dir: str, template_path: str, output_suffix: str):
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)), trim_blocks=True, lstrip_blocks=True)
    template = env.get_template(os.path.basename(template_path))
    dataset_folders = sorted(glob.glob(os.path.join(dataset_dir, "*")))

    for dataset in dataset_folders:
        if not os.path.isdir(dataset):
            continue

        csv_info = []
        for fname in os.listdir(dataset):
            if fname.endswith(".csv"):
                meta = get_schema_metadata(fname)
                if meta:
                    info = {"path": os.path.join(dataset, fname), "filename": fname, **meta}
                    csv_info.append(info)

        if not csv_info:
            print(f" No recognized CSVs in: {dataset}")
            continue

        folder_name = os.path.basename(dataset)
        output_file = os.path.join(dataset, folder_name + output_suffix)

        # Infer source (hardoi or maha_gov) for logging/debugging
        source_type = "hardoi" if "hardoi" in folder_name.lower() else "maha_gov" if "maha" in folder_name.lower() else "unknown"

        output_text = template.render(csv_info=csv_info, log_dir="./logs", space="medkg", source_type=source_type)

        with open(output_file, "w") as f:
            f.write(output_text)

        print(f" Generated: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render import YAMLs for Nebula from template")
    parser.add_argument("--base_dir", type=str, default="~/DATA/kg_csv", help="Folder containing dataset subfolders")
    parser.add_argument("--template", type=str, default="./import_template.yaml.j2")
    parser.add_argument("--suffix", type=str, default="_import.yaml")

    args = parser.parse_args()
    render_import_yaml(args.base_dir, args.template, args.suffix)