

from typing import List, Optional, Any
from pydantic import PrivateAttr, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from vllm import LLM, SamplingParams
from config import settings

import logging
logger = logging.getLogger(__name__)




class ChatModel(BaseChatModel):


    # these are the public attributes
    model_path: str = Field(default=settings.vllm_model)
    temperature: float = Field(default=settings.vllm_temp)
    max_token_limit: int = Field(default=settings.vllm_token_limit)

    # we also need private attributes for the LLM instance
    _llm: LLM = PrivateAttr(default=None) #Loads the actual model into GPU memory, holds model weights and GPU resources

    _sampling_params: SamplingParams = PrivateAttr(default=None) # Runtime configuration for inference

    # these attributes are initialized and checked before the model object is created once this is approved then we move to
    # the initialization of the model
    def model_post_init(self,_context : Any) -> None:
        stop = self.stop or ["</s>"]
        # now by default pydantic locks the private attributes of the object so we need to unlock it first
        object.__setattr__(self,"_sampling_params",SamplingParams(
            temp = self.temperature,
            max_token = self.max_token_limit,
            stop = stop

        ))
        object.__setattr__(self,"_llm",LLM(
            model = self.model_path,
            max_model_len = 8192

        ))
        logger.info(f"vLLM model loaded from {self.model_path} into GPU memory.")
    def _chat_formatting(self,messages : List[Any()]) -> str:
        
        pass

    












