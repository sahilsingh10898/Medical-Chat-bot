import os
import argparse
from jinja2 import Environment, FileSystemLoader

def render_import_yaml(dataset_dir: str, template_path: str = "./import_template.yaml.j2", output_file: str = "import.yaml"):
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

    # Replace "~" or relative with absolute path
    output_text = template.render(base_dir=dataset_dir)

    with open(output_file, "w") as f:
        f.write(output_text)

    print(f"✅ Generated '{output_file}' for: {dataset_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render NebulaGraph import.yaml from Jinja2 template")
    parser.add_argument("dataset_folder", type=str, help="Path to the dataset folder (e.g., ~/DATA/kg_csv/Maha_gov 2023_updated_cleaned)")
    parser.add_argument("--template", type=str, default="./import_template.yaml.j2", help="Path to the Jinja2 template file")
    parser.add_argument("--output", type=str, default="import.yaml", help="Name of output YAML file")

    args = parser.parse_args()

    render_import_yaml(args.dataset_folder, args.template, args.output)