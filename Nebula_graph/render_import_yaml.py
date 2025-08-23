import os
import argparse
from jinja2 import Environment, FileSystemLoader

def render_import_yaml(dataset_dir: str, template_path: str = "./import_template.yaml.j2", output_suffix: str = "_import.yaml"):
    """
    Renders the import.yaml file from a Jinja2 template based on the dataset directory structure.

    Parameters:
        dataset_dir (str): Absolute or relative path to the dataset directory (e.g., ~/DATA/kg_csv/Maha_gov 2023_updated_cleaned).
        template_path (str): Path to the Jinja2 template file.
        output_file (str): Output file name to save rendered YAML.
    """
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)), trim_blocks=True, lstrip_blocks=True)
    template = env.get_template(os.path.basename(template_path))

    dataset_folder = sorted(glob.glob(os.path.join(dataset_dir, "*")))

    for dataset in dataset_folder:
        if not os.path.isdir(dataset):
            continue
        folder_name = os.path.basename(dataset)
        output_file = os.path.join(dataset,folder_name + output_suffix)
        output_text = template.render(base_dir=dataset)

        with open(output_file, "w") as f:
            f.write(output_text)
        print(f" Generated '{output_file}' for: {dataset}")
        if not any(fname.endswith('.csv') for fname in os.listdir(dataset)):
            print(f"⚠️  No CSV files found in '{dataset}'. Skipping.")
            continue


if __name__ == "__main__":
    base_data_dir = "~/DATA/kg_csv"
    render_import_yaml(base_data_dir)
