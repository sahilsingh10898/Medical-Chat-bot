from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8")

    model_name: str = Field(default="meta-llama/Llama-3.2-3B-Instruct", env="MODEL_NAME")
    output_dir: str = Field(default=str(Path.home() / "logs"), env="OUTPUT_DIR")
    dataset_path: Optional[str] = Field(default="/home/ubuntu/DATA/dataset/train_protocol_bot_combined.jsonl", env="DATASET_PATH")
    huggingface_token: Optional[str] = Field(default=None, env="HUGGINGFACE_TOKEN")
    hugging_face_api_key: Optional[str] = Field(default=None, env="hugging_face_api_key")

    def model_post_init(self, __context) -> None:
        if not self.huggingface_token and self.hugging_face_api_key:
            object.__setattr__(self, "huggingface_token", self.hugging_face_api_key)


settings = Config()
