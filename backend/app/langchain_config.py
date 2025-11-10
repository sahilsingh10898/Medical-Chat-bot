# langchain_congig.py

import os
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
from langchain.memory import ConversationBufferMemory
from langchain_community.cache import InMemoryCache
from config import settings
from .vllm_config import ChatModel
from .patient_data import PatientDataFormat
from .schema import ValidatePatientData, ProtocolResponse

logger = logging.getLogger(__name__)

# we will define a global langchain configuration
class LangChainConfig:
    
    def __init__(self):
        self._llm = None
        self._memory = None
        self._formatter = None


    # lets create a conversation memory for the LLM it wont be necessary for out use case but still lets keep it for the use case
    @property
    def _conversation_memory(self):
        if self._memory is None:
            self._memory = self._create_memory()
        return self._memory

    @property
    def formatter(self):
        if self._formatter is None:
            self._formatter = PatientDataFormat(
                system_prompt=settings.default_system_prompt
            )
        return self._formatter

    def _config_model(self):
        if self._llm is None:
            try:
                self._llm = ChatModel(
                    model_path=settings.vllm_model,
                    temperature=settings.vllm_temp,
                    max_token_limit=settings.vllm_token_limit,
                    top_p=settings.vllm_top_p
                )
                logger.info("Successfully initialized ChatModel")
            except Exception as e:
                logger.error(f"Failed to initialize ChatModel: {e}")
                raise ImportError("Failed to import the model, check the config") from e

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

    # we will create a method to process the patient query

    def process_patient_query(self, request: ValidatePatientData,
                             stop_tokens: Optional[List[str]] = None,
                             include_input: bool = True) -> ProtocolResponse:
        """
        This method will tie up everything - it will validate the request from the schema.py
        and generate a proper prompt format from the patient_data.py

        Args:
            request (ValidatePatientData): Validated patient data
            stop_tokens (Optional[List[str]]): Optional stop tokens for generation
            include_input (bool): Whether to include patient data in response

        Returns:
            ProtocolResponse: Response containing protocol and metadata
        """

        start = time.time()
        try:
            # Ensure model is initialized
            llm = self._config_model()

            # create the chat message from the formatter
            messages = self.formatter.create_chat_message(request)
            logger.debug(f"Processing the case: {request.chief_complaint}")

            # lets pass the message to the model
            result = llm._generate(messages, stop=stop_tokens)

            # lets extract the generated results
            raw_response = result.generations[0].message.content

            # Extract protocol (just use raw response for now since extract_protocol_from_response doesn't exist)
            protocol = raw_response.strip()

            inference_time_ms = (time.time() - start) * 1000

            metadata = {
                "inference_time_ms": round(inference_time_ms, 2),
                "model_path": settings.vllm_model,
                "temperature": settings.vllm_temp,
                "max_tokens": settings.vllm_token_limit,
            }

            patient_summary = None
            if include_input:
                patient_summary = {
                    "age": request.age,
                    "gender": request.gender,
                    "chief_complaint": request.chief_complaint,
                    "vitals": request.vitals.model_dump() if request.vitals else None,
                    "past_medical_history": request.past_medical_history
                }

            response = ProtocolResponse(
                protocol=protocol,
                patient_summary=patient_summary,
                metadata=metadata,
                timestamp=datetime.now().isoformat()
            )

            logger.info(f"Successfully generated protocol in {inference_time_ms:.2f}ms")
            return response
        except Exception as e:
            logger.error(f"Error processing patient request: {e}", exc_info=True)
            raise


langchain_config = LangChainConfig()

    


