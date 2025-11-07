import os
import time
from typing import Optional, Dict, Any
import logging
from langchain.memory import ConversationBufferMemory
from langchain_community.cache import InMemoryCache
from fine_tuning.config import settings
from .vllm_config import ChatModel


logger = logging.getLogger(__name__)

# we will define a global langchain configuration
class LangChainConfig:
    
    def __init__(self):
        self._llm = None
        self._memory = None


    # lets create a conversation memory for the LLM it wont be necessary for out use case but still lets keep it for the use case
    @property
    def _conversation_memory(self):
        if self._memory is None:
            self._memory = self._create_memory()
        return self._memory

    def _config_model(self):
        if self._llm is None:
            try:
                self._llm = ChatModel(
                    model = settings.vllm_model,
                    temp = settings.vllm_temp,
                    max_token_limit = settings.vllm_token_limit,
                    top_p = settings.vllm_top_p

                )
            except Exception as e:
                raise ImportError("failed to import the model check the config") from e

        return self._llm

    def _create_memory(self):
        """
        create the memory layer for the conversation 
        we will not use this with the SLM as we are only using it for a specific case
        but we will create it so that we can use this layer when there is a requirement
        """
        memory = ConversationBufferMemory(
            memory_key = "chat_memory",
            return_messages =True,
            output_key = "results"
        )
        return memory

        

    


