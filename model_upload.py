from huggingface_hub import HfApi, create_repo
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('/home/ubuntu/Medical-Chat-bot/fine_tuning/.env')

# Configuration
MODEL_PATH = "/home/ubuntu/logs/merged_model"
REPO_NAME = "sssahilsingh/medical-chat-bot-medication-recommendation"  # Change this to your HF username and desired model name
REPO_TYPE = "model"
PRIVATE = True  # Set to True if you want a private repository

def upload_model(hf_token):
    """Upload model to Hugging Face Hub"""

    # Check if model directory exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

    print(f"Model directory found: {MODEL_PATH}")

    # List files to be uploaded
    files = os.listdir(MODEL_PATH)
    print(f"\nFiles to upload ({len(files)}):")
    for file in sorted(files):
        file_path = os.path.join(MODEL_PATH, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  - {file} ({size_mb:.2f} MB)")

    # Initialize Hugging Face API with token
    api = HfApi(token=hf_token)

    

    # Create repository (if it doesn't exist)
    try:
        print("Creating repository...")
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
        print("\nUploading model files...")
        api.upload_folder(
            folder_path=MODEL_PATH,
            repo_id=REPO_NAME,
            repo_type=REPO_TYPE,
            commit_message="Upload merged model"
        )
        print("Upload completed successfully!")
        print(f"\nModel available at: https://huggingface.co/{REPO_NAME}")
    except Exception as e:
        print(f"Error uploading model: {e}")
        raise

if __name__ == "__main__":
    print("Hugging Face Model Upload Script")
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

    upload_model(hf_token)
