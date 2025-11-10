import requests
from typing import Optional, Dict, Any
import json


class ProtocolAPIClient:
    """
    Simple client wrapper for the Medical Protocol Generation API
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        response = self.session.get(f"{self.base_url}/model-info")
        response.raise_for_status()
        return response.json()
    
    def validate_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate patient data without generating a protocol

        Args:
            patient_data: Dictionary containing patient information

        Returns:
            Validation result with formatted patient data
        """
        response = self.session.post(
            f"{self.base_url}/validate-data",
            json=patient_data
        )
        response.raise_for_status()
        return response.json()
    
    def generate_response(
        self,
        patient_data: Dict[str, Any],
        include_input: bool = True,
        stop_tokens: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate medical protocol based on patient data
        
        Args:
            patient_data: Dictionary containing patient information
            include_input: Whether to include patient summary in response
            stop_tokens: Optional custom stop tokens for generation
            
        Returns:
            Generated protocol with metadata
        """
        params = {"include_input": include_input}
        if stop_tokens:
            params["stop_tokens"] = stop_tokens
        
        response = self.session.post(
            f"{self.base_url}/generate-response",
            json=patient_data,
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def get_protocol_text(self, patient_data: Dict[str, Any]) -> str:
        """
        Convenience method to get just the protocol text
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Generated protocol as string
        """
        result = self.generate_response(patient_data, include_input=False)
        return result['protocol']