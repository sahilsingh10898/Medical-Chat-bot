

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
    top_p: float = Field(default=settings.vllm_top_p)

    # we also need private attributes for the LLM instance
    _llm: LLM = PrivateAttr(default=None) #Loads the actual model into GPU memory, holds model weights and GPU resources

    _sampling_params: SamplingParams = PrivateAttr(default=None) # Runtime configuration for inference

    _tokenizer: Any = PrivateAttr(default=None) # Tokenizer for chat template formatting

    # these attributes are initialized and checked before the model object is created once this is approved then we move to
    # the initialization of the model
    def model_post_init(self,_context : Any) -> None:
        from transformers import AutoTokenizer

        stop = self.stop or ["</s>"]
        # now by default pydantic locks the private attributes of the object so we need to unlock it first
        object.__setattr__(self,"_sampling_params",SamplingParams(
            temperature = self.temperature,
            max_tokens = self.max_token_limit,
            top_p = self.top_p,
            stop = stop

        ))
        object.__setattr__(self,"_llm",LLM(
            model=self.model_path,
            max_model_len=8192,
            tokenizer_mode="auto"
        ))

        # Initialize tokenizer for chat template formatting
        object.__setattr__(self,"_tokenizer",AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        ))

        logger.info(f"vLLM model loaded from {self.model_path} into GPU memory.")
        
    def _chat_formatting(self,messages : List[Any]) -> str:
        """
        Format messages using the tokenizer's chat template (like in test_script.py)

        Args:
            messages (List[Any]): these messages are in the form of langchain messages and we need to properly format them

        Returns:
            str: properly formatted string for LLM inferencing
        """
        # Convert langchain messages to the format expected by apply_chat_template
        formatted_messages = []
        for msg in messages:
            # Determine role from message type
            if hasattr(msg, '__class__'):
                class_name = msg.__class__.__name__
                if 'System' in class_name:
                    role = "system"
                elif 'Human' in class_name or 'User' in class_name:
                    role = "user"
                elif 'AI' in class_name or 'Assistant' in class_name:
                    role = "assistant"
                else:
                    role = "user"  # default to user
            else:
                role = "user"

            formatted_messages.append({
                "role": role,
                "content": msg.content
            })

        # Use tokenizer's chat template 
        prompt = self._tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def _generate(self,messages : List[Any],stop : Optional[List[Any]]) -> ChatResult:
        """Generate response from the model

        Args:
            messages (List[Any]): messages that we will pass to the model for the inferencing
            stop (Optional[List[Any]]): allows runtime override of stop tokens for text generation

        Returns:
            ChatResult: the chat result object that contains generations and other metadata
        """
        # first we need to check that the model is initialized or not and it is defensive part
        if not isinstance(self._llm,LLM) or not isinstance(self._sampling_params , SamplingParams):
            self.model_post_init(None)

        # lets convert the messages into the proper string format
        prompt = self._chat_formatting(messages)

        # if there is a custom stop token that we want to use then we will override the default SamplingParams
        if stop:
            sampling_params = SamplingParams(
                temperature = self.temperature,
                max_tokens = self.max_token_limit,
                top_p = self.top_p,
                stop = stop,
            )
        else:
            sampling_params = self._sampling_params

        # now we can pass the prompt to the model for the inferencing
        results = self._llm.generate([prompt], sampling_params)

        # the results object contains the generations and other metadata
        text = results[0].outputs[0].text.strip() if results[0].outputs else ""

        # now we need to convert the text into the ChatResult format
        generation = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )

        return generation

    @property
    def _llm_type(self) -> str:
        """Return identifier for the LLM type"""
        return "vllm-chat"




    












