
import re
from typing import Optional, Dict, Any
from pydantic import BaseModel,Field,field_validator
from datetime import datetime

"""
we will use this script for validating the data that we will recieve as an input 
and also make some fallbacks so that we can validate as well as also correct 
we will use 2 classes to perform the data validation 
"""

class VitalsValidation(BaseModel):

    bp : Optional[str] = Field(None,description="for giving the bp values")
    temperature : Optional[str] = Field(None,description="for giving the temp values")
    temp : Optional[str] = Field(None,description="for giving the temp values (alias)")
    hemoglobin : Optional[float] = Field(None,description="for giving the hemoglobin values")
    hb : Optional[float] = Field(None,description="for giving the hemoglobin values (alias)")
    spo2 : Optional[str] = Field(None,description="for giving the spo2 values")
    rbs : Optional[float] = Field(None,description="random blood sugar values")
    ppbs : Optional[float] = Field(None,description="post prandial blood sugar values (alias)")


    # for api documentation
    class Config:
        json_schema_sample = {
            "example":{
                "bp":"130/80",
                "temperature": "101°F",
                "hemoglobin": 13.5,
                "spo2": "94%",
                "ppbs":280
            }
        }
        
class ValidatePatientData(BaseModel):

    age : str = Field(...,description="age of the patient")
    gender : str = Field(...,description="gender of patient")
    chief_complaint : str = Field(...,description="chief complaints of the patient")
    vitals : Optional[VitalsValidation] = Field(
        default_factory=VitalsValidation,
        description="vitals of the patient"
        )

    past_medical_history : str = Field(
        default="No significant past medical history",
        description="past medical history of the patient"
    )
    # Alias for backwards compatibility
    history : Optional[str] = Field(
        default=None,
        description="history of the patient (alias for past_medical_history)"
    )

    """
    lets make some field validators so that we can actually validate the datatype we will use
    decorators from pydantic and python

    """

    @field_validator('age')
    @classmethod
    def validate_age(cls, age:str) -> str: 
        """
        lets do one thing is lets accept all the inputs and make sure that whatever the input is 
        we make sure that after the age if any character is given we treat is year

        Raises:
            ValueError: incorrect age range

        Returns:
            _type_: properly formatted age
        """
        if not isinstance(age,str):
            age = str(age)

        # clean the whitespace
        age = age.strip()

        # This regex pattern looks for digits at the START of the string
        # ^(\d+)   -> Capture group 1: one or more digits (e.g., "48")
        # (.*)$    -> Capture group 2: any other characters (e.g., "yearsss")

        match = re.match(r"^(\d+)(.*)$",age)
        # here itself we can match that if match doesnot start with valid age we can simply raise an error 
        # here itself
        if not match:
            raise ValueError("please enter a valid age between 10-70")
        try:
            age_value = int(match.group(1))
            if age_value <= 10 or age_value >= 70:
                raise ValueError("please enter a valid age between 10-70")
            else:
                return f"{match.group(1)}y"
        except ValueError:
            raise ValueError(f"not a valid age {age}")


    @field_validator('gender')
    @classmethod
    def validate_gender(cls,gender :str) -> str:
        """
        we need to make sure that we only take 2 genders male and female

        Args:
            gender (str): gender of the patient

        Returns:
            str: properly formatted gender of the patient
        """
        gender_norm = gender.strip().lower()
        
        allowed = {"male","female"}
        if gender_norm in allowed:
            return gender_norm

        raise ValueError(f"not a valid gender {gender}")

    @field_validator('chief_complaint')
    @classmethod
    def validate_complain(cls,chief_complaint : str) -> str:
        # we need to validate that the complaints are not empty and if they are then raise the error
        comp = chief_complaint.strip().lower()
        if comp and len(comp) > 0:
            return comp
        raise ValueError("symptoms can not be empty")


class ProtocolResponse(BaseModel):
    """Response model for protocol generation"""
    protocol: str = Field(..., description="Generated medical protocol")
    patient_summary: Optional[Dict[str, Any]] = Field(
        None, 
        description="Summary of input patient data"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata (inference time, model info, etc.)"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp of response generation"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "protocol": "1. Administer aspirin 325mg...",
                "patient_summary": {
                    "age": "65y",
                    "gender": "male",
                    "chief_complaint": "Chest pain"
                },
                "metadata": {
                    "inference_time_ms": 1234.5,
                    "model_version": "gemma-2-9b-it"
                },
                "timestamp": "2024-01-15T10:30:00"
            }
        }

    
class ErrorResponse(BaseModel):

    """
    error response model
    """
    error :str = Field(...,description="errors type")
    detail :str = Field(...,description="detail of the error")
    timestamp :str = Field(
        default_factory= lambda : datetime.now().isoformat(),
        description="time stamp of the error"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Chief complaint cannot be empty",
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    

                
                




        





