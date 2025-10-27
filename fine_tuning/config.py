from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Config(BaseSettings):
    # Model
    model_name : str ='meta-llama/Llama-3.2-3B-Instruct'


settings = Config()
