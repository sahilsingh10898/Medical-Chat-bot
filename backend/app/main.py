#main.py

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Add project root to Python path to allow imports from config.py
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import settings
from .langchain_config import langchain_config
from .vllm_config import ChatModel
from . import vllm_config
from .schema import ValidatePatientData, ProtocolResponse, ErrorResponse


from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# so we need to define a function for loading and shutting the model at the start or termination of the uvicorn server

@asynccontextmanager
async def start_termination(app : FastAPI):
    """

    handles the start and shutdown events
    """
    try:
        # initialzing the model
        langchain_config._config_model()
        logger.debug("model loaded successfully")
    except Exception as e:
        logger.error("Failed to load the model")
        raise
    yield

    logger.info("successfully shut down")

# now we need to create the app object of the Fastapi class
app = FastAPI(
    title="Medicine Reccomendation System API",
    description="API for generating medicine prescription based on patient data using fine-tuned LLM",
    version='1.0.0',
    lifespan=start_termination
)



# create a custom error handling function
@app.exception_handler(ValueError)
async def handle_value_error(request:Request,exc:ValueError):
    logger.error(f"Validation Error : {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error = 'Validation',
            detail = str(exc)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            detail="An unexpected error occurred while processing your request"
        ).model_dump()
    )

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "message": "Medicine Reccomendation System API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get('/health')
async def get_health():
    """
    health check endpoint
    """
    try:
        #check whether the model is initialized or not
        model_status = 'initialized' if langchain_config._llm is not None else 'not initialized'
        return {
            'status':'healthy',
            'model_status':model_status,
            'model_path':vllm_config.model_path
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                'status':'unhealthy',
                'error':str(e)

            }
        )
@app.post('/generate-response',response_model=ProtocolResponse)
async def generate_response(
    patient_data: ValidatePatientData = Body(..., embed=True),
    include_input: bool = Body(default=True),
    stop_tokens: Optional[list[str]] = Body(default=None)

) -> ProtocolResponse:


    """
    Argu:
        patient data = validated patient data like age symptoms history vital etc
        include_inputs = Whether to include patient summary in the response (default: True)
        stop tokens =  Optional custom stop tokens for generation
    Returns:
         Protocol response = Validated response from the model
    Raise:
        HTTPException : if protocol generation fails
    """

    try:
        logger.info(f"generating the response for the patient {patient_data}")
        # use the model for generating the response from the model
        results = langchain_config.process_patient_query(
            request=patient_data,
            stop_tokens=stop_tokens,
            include_input=include_input
        )
        logger.info("response generated")
        return results
    except ValueError as e:
        logger.error(f"value error {e}")
        raise HTTPException(status_code=422,detail=str(e))
    except Exception as e:
        logger.error(f"failed to generate the result")
        raise HTTPException(status_code=500,detail=f'Failed to generate protocol: {str(e)}')



@app.post('/validate-data')
async def validate_patientdata(
    patient_data: ValidatePatientData = Body(..., embed=True)
):
    """
    Validate patient data without generating a protocol
    """

    try:

        validate = langchain_config.formatter
        formated_prompt = validate.format_prompt(patient_data)
        result  = {
            'status':'valid',
            'message':'patient data is correct',
            'formatted_prompt':formated_prompt ,
            'patient_summary':{
                'age':patient_data.age,
                'gender':patient_data.gender,
                'symptoms':patient_data.chief_complaint,
                'vital':patient_data.vitals.model_dump() if patient_data.vitals else None,
                'past medical history':patient_data.past_medical_history
            }
        }

        return result
    except ValueError as e:
        logging.error(f"data validation failed {e}")
        raise HTTPException(status_code=422,detail=str(e))

@app.get("/model-info")
async def model_info():
    """
    Get information about the loaded model
    
    Returns:
        dict: Model configuration and status
    """
    try:
        return {
            "model_path": settings.vllm_model,
            "temperature": settings.vllm_temp,
            "max_tokens": settings.vllm_token_limit,
            "top_p": settings.vllm_top_p,
            "system_prompt": settings.default_system_prompt,
            "status": "loaded" if langchain_config._llm is not None else "not_loaded"
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))





    

   


    




