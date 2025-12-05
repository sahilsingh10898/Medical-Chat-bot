#!/home/ubuntu/Medical-Chat-bot/venv-finetune/bin/python
"""
Script to upload the DATA folder to Hugging Face Hub as a dataset
"""

from huggingface_hub import HfApi, create_repo
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('/home/ubuntu/Medical-Chat-bot/fine_tuning/.env')

# Configuration
DATA_PATH = "/home/ubuntu/DATA"
REPO_NAME = "sssahilsingh/fine-tuning-dataset"
REPO_TYPE = "dataset"  # This is a dataset, not a model
PRIVATE = False  # Set to True if you want a private dataset

def upload_dataset(hf_token):
    """Upload dataset to Hugging Face Hub"""

    # Check if dataset directory exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset directory not found: {DATA_PATH}")

    print(f"Dataset directory found: {DATA_PATH}")

    # Get directory structure and file counts
    total_files = 0
    total_size = 0
    subdirs = {}

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if not file.startswith('.'):  # Skip hidden files
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_files += 1
                total_size += file_size

                # Track files per subdirectory
                rel_path = os.path.relpath(root, DATA_PATH)
                if rel_path == '.':
                    rel_path = 'root'
                subdirs[rel_path] = subdirs.get(rel_path, 0) + 1

    print(f"\nDataset summary:")
    print(f"  Total files: {total_files}")
    print(f"  Total size: {total_size / (1024*1024):.2f} MB")
    print(f"\nFiles by directory:")
    for dir_name, count in sorted(subdirs.items()):
        print(f"  - {dir_name}: {count} files")

    # Initialize Hugging Face API with token
    api = HfApi(token=hf_token)

    print(f"\n{'='*60}")
    print(f"Uploading to: {REPO_NAME}")
    print(f"Repository type: {REPO_TYPE}")
    print(f"Private: {PRIVATE}")
    print(f"{'='*60}\n")

    # Create repository (if it doesn't exist)
    try:
        print("Creating dataset repository...")
        create_repo(
            repo_id=REPO_NAME,
            repo_type=REPO_TYPE,
            private=PRIVATE,
            exist_ok=True,
            token=hf_token
        )
        print("Repository created/verified")
    except Exception as e:
        print(f"Error creating repository: {e}")
        raise

    # Upload the entire folder
    try:
        print("\nUploading dataset files...")
        print("This may take a few minutes depending on your connection speed...")
        api.upload_folder(
            folder_path=DATA_PATH,
            repo_id=REPO_NAME,
            repo_type=REPO_TYPE,
            commit_message="Upload fine-tuning dataset",
            ignore_patterns=[".DS_Store", "*.pyc", "__pycache__", ".git"]
        )
        print("Upload completed successfully!")
        print(f"\nDataset available at: https://huggingface.co/datasets/{REPO_NAME}")
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        raise

if __name__ == "__main__":
    print("Hugging Face Dataset Upload Script")
    print("=" * 60)

    # Get token from environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("\nHugging Face token not found!")
        print("Make sure HUGGINGFACE_TOKEN is set in your .env file")
        exit(1)

    # Check if user is authenticated
    try:
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        print(f"Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"\nAuthentication failed: {e}")
        print("Please check your Hugging Face token")
        exit(1)

    # Confirm before uploading
    print(f"\nReady to upload from: {DATA_PATH}")
    print(f"Target repository: {REPO_NAME}")

    response = input("\nProceed with upload? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        upload_dataset(hf_token)
    else:
        print("Upload cancelled.")
