import os
import argparse
from jinja2 import Environment, FileSystemLoader
import glob

def render_import_yaml(dataset_dir: str, template_path: str = "./import_template.yaml.j2", output_suffix: str = "_import.yaml"):
    """
    Renders import.yaml files from a Jinja2 template for each subdirectory in dataset_dir.

    Parameters:
        dataset_dir (str): Path to the root dataset folder containing subfolders with CSVs.
        template_path (str): Path to the Jinja2 template file.
        output_suffix (str): Suffix for the rendered YAML filename.
    """
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)), trim_blocks=True, lstrip_blocks=True)
    template = env.get_template(os.path.basename(template_path))

    dataset_folders = sorted(glob.glob(os.path.join(dataset_dir, "*")))

    for dataset in dataset_folders:
        if not os.path.isdir(dataset):
            continue

        if not any(fname.endswith('.csv') for fname in os.listdir(dataset)):
            print(f"⚠️  No CSV files found in '{dataset}'. Skipping.")
            continue

        folder_name = os.path.basename(dataset)
        output_file = os.path.join(dataset, folder_name + output_suffix)
        output_text = template.render(base_dir=dataset)

        with open(output_file, "w") as f:
            f.write(output_text)

        print(f"✅ Generated '{output_file}' for: {dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch render NebulaGraph import YAMLs from Jinja2 template.")
    parser.add_argument("--base_dir", type=str, default="~/DATA/kg_csv", help="Path to the root folder containing dataset subfolders.")
    parser.add_argument("--template", type=str, default="./import_template.yaml.j2", help="Path to the Jinja2 template.")
    parser.add_argument("--suffix", type=str, default="_import.yaml", help="Suffix for the output YAML file.")

    args = parser.parse_args()

    render_import_yaml(args.base_dir, args.template, args.suffix)