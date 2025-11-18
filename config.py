from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8")

    model_name: str = Field(default="/home/ubuntu/logs/final_model_complete", env="MODEL_NAME")
    output_dir: str = Field(default=str(Path.home() / "logs_final"), env="OUTPUT_DIR")
    quantized_model_dir: str = Field(default=str(Path.home() / "quantized_models"), env="QUANTIZED_MODEL_DIR")
    
    dataset_path: Optional[str] = Field(default="/home/ubuntu/DATA/dataset/train_protocol_bot_combined (2).jsonl", env="DATASET_PATH")
    huggingface_token: Optional[str] = Field(default=None, env="HUGGINGFACE_TOKEN")
    hugging_face_api_key: Optional[str] = Field(default=None, env="hugging_face_api_key")


    # maintain a variable for the LLM provider as i will be experimenting with mutiple LLMs configurations
    llm_provider: str = Field(default="vllm", env="LLM_PROVIDER")

    # now for the vLLM settings
    vllm_model :str = Field(default="/home/ubuntu/logs/quantized_awq_model", env="VLLM_MODEL")
    vllm_temp : float = Field(default=0.2 , env="vllm_temp")
    vllm_token_limit : int = Field(default=384 , env = "vllm_token_limit")  # Reduced from 512 for faster inference
    vllm_top_p : float = Field(default=0.9 , env="vllm_top_p")

    # Default system prompt for the medical chatbot
    default_system_prompt: str = Field(
        default="Analyze the following patient case and output the common protocols.",
        env="DEFAULT_SYSTEM_PROMPT"
    )


    def model_post_init(self, __context) -> None:
        if not self.huggingface_token and self.hugging_face_api_key:
            object.__setattr__(self, "huggingface_token", self.hugging_face_api_key)


settings = Config()
