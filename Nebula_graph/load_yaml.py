import os
import subprocess
import argparse
from pathlib import Path

def run_importer_for_yaml(yaml_path):
    folder_path = yaml_path.parent
    #print(f"Importing: {yaml_path} ...")

    command = [
        "docker", "run", "--rm", "-ti",
        "-v", f"{folder_path}:/import",
        "vesoft/nebula-importer:v3.4.0",
        "--config", f"/import/{yaml_path.name}"
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f" Failed for {yaml_path}: {e}")

def main(base_dir: str):
    base_path = Path(base_dir).expanduser()
    if not base_path.exists():
        print(f"Base directory not found: {base_dir}")
        return

    # Recursively find all *_import.yaml files
    yaml_files = list(base_path.glob("*/**/*_import.yaml"))

    if not yaml_files:
        print(" No import YAML files found.")
        return

    print(f" Found {len(yaml_files)} import YAML files.")

    for yaml_file in yaml_files:
        run_importer_for_yaml(yaml_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk load YAML configs into NebulaGraph.")
    parser.add_argument("--base_dir", required=True, help="Base directory containing subfolders with *_import.yaml files.")
    args = parser.parse_args()

    main(args.base_dir)