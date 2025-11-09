from typing import List,Optional
from langchain_core.messages import HumanMessage,SystemMessage,BaseMessage
from .schema import ValidatePatientData,VitalsValidation
import logging

logger = logging.getLogger(__name__)


class PatientDataFormat():

    DEFAULT_SYSTEM_PROMPT = "Analyze the following patient case and output the common protocols."

    def __init__(self,system_prompt : Optional[str] =None):
        """ we will initialize the class with default system propmt or with custom system prompt

        Args:
            system_prompt (Optional[str], optional): _description_. Defaults to None.
        """

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def format_vitals(self , vitals:Optional[VitalsValidation]) -> str:

        """
        Format vital signs into a comma-separated string.
        Args:
        VitalsModel object with optional vital sign fields
        Returns:
        Formatted vitals string (e.g., "bp 130/80, temperature 101°F, hemoglobin 13.5, SpO2 94%")
            Returns "Not recorded" if no vitals provided
        """

        if not vitals:
            return "Not recorded"

        vitals_present = []

        # Check all possible field names (including aliases)
        if vitals.bp:
            vitals_present.append(f"bp {vitals.bp}")

        temp_value = vitals.temperature or vitals.temp
        if temp_value:
            vitals_present.append(f"temperature {temp_value}")

        hb_value = vitals.hemoglobin or vitals.hb
        if hb_value:
            vitals_present.append(f"hemoglobin {hb_value}")

        if vitals.spo2:
            vitals_present.append(f"SpO2 {vitals.spo2}")

        rbs_value = vitals.rbs or vitals.ppbs
        if rbs_value:
            vitals_present.append(f"RBS {rbs_value}")

        # if no vitals are provided return default
        if not vitals_present:
            return "Not recorded"

        return ", ".join(vitals_present)

    # now we will create a method which will take all the patient data and 
    # return a properly formatted prompt

    def format_prompt(self, request: ValidatePatientData) -> str:

        """
        we will take the request from the patient and format them in a specific 
        format 
        """

        # format the vitals
        vitals_str = self.format_vitals(request.vitals)

        # Get past medical history (use past_medical_history or fall back to history)
        pmh = request.past_medical_history or request.history or "No significant past medical history"

        # now lets take the vitals and the patient data and then return them in a proper format
        formatted_prompt = f"""Patient: {request.age} {request.gender}
CC: {request.chief_complaint}
Vitals: {vitals_str}
PMH: {pmh}
Protocol?"""

        logger.debug(f'Formatting the patient data:\n{formatted_prompt}')

        return formatted_prompt
    

    # lets create a method which will take the patient data and return a list 
    # of langchain message object [system message , human message]

    def create_chat_message(self, request: ValidatePatientData, system_prompt: Optional[str] = None) -> List[BaseMessage]:
        """Create LangChain message objects from patient data"""

        # we will have a human message as well as system message
        system_msg_content = system_prompt or self.system_prompt
        human_msg_content = self.format_prompt(request)

        # create the message object
        messages = [
            SystemMessage(content=system_msg_content),
            HumanMessage(content=human_msg_content)
        ]

        logger.debug(f"Created message objects: {len(messages)} messages")
        return messages
        
    def extract_response(self,response : str)-> str:
        pass

    

         



    

        
