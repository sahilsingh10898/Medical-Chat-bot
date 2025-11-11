#!/usr/bin/env python3
"""
Test script for Medical Chatbot API
Tests all endpoints and displays model outputs
"""

import requests
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime


class APITester:
    """Test harness for Medical Chatbot API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []

    def print_header(self, text: str):
        """Print formatted header"""
        print("\n" + "="*80)
        print(f"  {text}")
        print("="*80)

    def print_section(self, text: str):
        """Print formatted section"""
        print("\n" + "-"*80)
        print(f"  {text}")
        print("-"*80)

    def test_health_check(self) -> bool:
        """Test the /health endpoint"""
        self.print_section("Testing /health endpoint")

        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()

            data = response.json()
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(data, indent=2)}")

            self.test_results.append({
                "endpoint": "/health",
                "status": "PASS",
                "details": data
            })
            return True

        except Exception as e:
            print(f"ERROR: {str(e)}")
            self.test_results.append({
                "endpoint": "/health",
                "status": "FAIL",
                "error": str(e)
            })
            return False

    def test_model_info(self) -> bool:
        """Test the /model-info endpoint"""
        self.print_section("Testing /model-info endpoint")

        try:
            response = self.session.get(f"{self.base_url}/model-info")
            response.raise_for_status()

            data = response.json()
            print(f"Status Code: {response.status_code}")
            print(f"Model Information:")
            print(f"  Model Path: {data.get('model_path')}")
            print(f"  Temperature: {data.get('temperature')}")
            print(f"  Max Tokens: {data.get('max_tokens')}")
            print(f"  Top P: {data.get('top_p')}")
            print(f"  Status: {data.get('status')}")
            print(f"  System Prompt: {data.get('system_prompt')}")

            self.test_results.append({
                "endpoint": "/model-info",
                "status": "PASS",
                "details": data
            })
            return True

        except Exception as e:
            print(f"ERROR: {str(e)}")
            self.test_results.append({
                "endpoint": "/model-info",
                "status": "FAIL",
                "error": str(e)
            })
            return False

    def test_validate_data(self, patient_data: Dict[str, Any]) -> bool:
        """Test the /validate-data endpoint"""
        self.print_section("Testing /validate-data endpoint")

        try:
            print(f"Input Patient Data:")
            print(json.dumps(patient_data, indent=2))

            response = self.session.post(
                f"{self.base_url}/validate-data",
                json=patient_data
            )
            response.raise_for_status()

            data = response.json()
            print(f"\nStatus Code: {response.status_code}")
            print(f"Validation Status: {data.get('status')}")
            print(f"Message: {data.get('message')}")
            print(f"\nFormatted Prompt:")
            print(data.get('formatted_prompt'))

            self.test_results.append({
                "endpoint": "/validate-data",
                "status": "PASS",
                "patient_data": patient_data,
                "formatted_prompt": data.get('formatted_prompt')
            })
            return True

        except Exception as e:
            print(f"ERROR: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            self.test_results.append({
                "endpoint": "/validate-data",
                "status": "FAIL",
                "error": str(e)
            })
            return False

    def test_generate_response(self, patient_data: Dict[str, Any],
                               include_input: bool = True,
                               stop_tokens: Optional[list] = None) -> Dict[str, Any]:
        """Test the /generate-response endpoint"""
        self.print_section(f"Testing /generate-response endpoint")

        try:
            print(f"Input Patient Data:")
            print(json.dumps(patient_data, indent=2))

            payload = {
                "patient_data": patient_data,
                "include_input": include_input
            }

            if stop_tokens:
                payload["stop_tokens"] = stop_tokens

            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/generate-response",
                json=payload
            )
            client_latency = (time.time() - start_time) * 1000

            response.raise_for_status()

            data = response.json()
            print(f"\nStatus Code: {response.status_code}")
            print(f"Client Latency: {client_latency:.2f}ms")

            if data.get('metadata'):
                print(f"Server Inference Time: {data['metadata'].get('inference_time_ms')}ms")
                print(f"Model Path: {data['metadata'].get('model_path')}")
                print(f"Temperature: {data['metadata'].get('temperature')}")
                print(f"Max Tokens: {data['metadata'].get('max_tokens')}")

            print(f"\n{'='*40}")
            print("MODEL OUTPUT (Protocol):")
            print('='*40)
            print(data.get('protocol', 'No protocol generated'))
            print('='*40)

            if data.get('patient_summary'):
                print(f"\nPatient Summary:")
                print(json.dumps(data['patient_summary'], indent=2))

            self.test_results.append({
                "endpoint": "/generate-response",
                "status": "PASS",
                "patient_data": patient_data,
                "protocol": data.get('protocol'),
                "inference_time_ms": data.get('metadata', {}).get('inference_time_ms'),
                "client_latency_ms": client_latency
            })

            return data

        except Exception as e:
            print(f"ERROR: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            self.test_results.append({
                "endpoint": "/generate-response",
                "status": "FAIL",
                "error": str(e)
            })
            return {}

    def run_test_suite(self):
        """Run complete test suite"""
        self.print_header("MEDICAL CHATBOT API TEST SUITE")
        print(f"Base URL: {self.base_url}")
        print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Test 1: Health Check
        self.test_health_check()

        # Test 2: Model Info
        self.test_model_info()

        # Test 3: Validate Data - Simple Case
        simple_patient = {
            "age": "45",
            "gender": "male",
            "chief_complaint": "chest pain and shortness of breath"
        }
        self.test_validate_data(simple_patient)

        # Test 4: Generate Response - Simple Case
        self.print_header("TEST CASE 1: Simple Patient (Chest Pain)")
        self.test_generate_response(simple_patient)

        # Test 5: Generate Response - Complex Case with Vitals
        self.print_header("TEST CASE 2: Complex Patient (Diabetes with Vitals)")
        complex_patient = {
            "age": "58",
            "gender": "female",
            "chief_complaint": "increased thirst, frequent urination, fatigue",
            "vitals": {
                "bp": "150/95",
                "temperature": "98.6°F",
                "hemoglobin": 11.2,
                "spo2": "96%",
                "ppbs": 280
            },
            "past_medical_history": "Type 2 diabetes for 5 years, hypertension"
        }
        self.test_generate_response(complex_patient)

        # Test 6: Generate Response - Acute Emergency Case
        self.print_header("TEST CASE 3: Emergency Case (Severe Abdominal Pain)")
        emergency_patient = {
            "age": "35",
            "gender": "male",
            "chief_complaint": "severe abdominal pain in right lower quadrant, nausea, vomiting",
            "vitals": {
                "bp": "120/80",
                "temperature": "102°F",
                "spo2": "98%"
            },
            "past_medical_history": "No significant past medical history"
        }
        self.test_generate_response(emergency_patient)

        # Test 7: Generate Response - Respiratory Case
        self.print_header("TEST CASE 4: Respiratory Case (Chronic Cough)")
        respiratory_patient = {
            "age": "62",
            "gender": "male",
            "chief_complaint": "persistent cough with yellowish sputum, difficulty breathing",
            "vitals": {
                "bp": "130/85",
                "temperature": "100.5°F",
                "spo2": "92%"
            },
            "past_medical_history": "Smoker for 30 years, COPD"
        }
        self.test_generate_response(respiratory_patient)

        # Test 8: Generate Response - Pediatric Case
        self.print_header("TEST CASE 5: Young Patient (Fever and Rash)")
        pediatric_patient = {
            "age": "22",
            "gender": "female",
            "chief_complaint": "high fever, skin rash, body aches",
            "vitals": {
                "bp": "110/70",
                "temperature": "103°F",
                "spo2": "97%"
            },
            "past_medical_history": "No significant past medical history"
        }
        self.test_generate_response(pediatric_patient)

        # Print Summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        self.print_header("TEST SUMMARY")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed_tests = total_tests - passed_tests

        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        # Show inference times for successful generation tests
        generation_tests = [r for r in self.test_results
                          if r['endpoint'] == '/generate-response'
                          and r['status'] == 'PASS']

        if generation_tests:
            print("\n" + "-"*80)
            print("Inference Performance:")
            print("-"*80)
            for i, test in enumerate(generation_tests, 1):
                print(f"  Test Case {i}:")
                print(f"    Server Inference Time: {test.get('inference_time_ms')}ms")
                print(f"    Client Total Latency: {test.get('client_latency_ms', 0):.2f}ms")

            avg_inference = sum(t.get('inference_time_ms', 0) for t in generation_tests) / len(generation_tests)
            print(f"\n  Average Server Inference Time: {avg_inference:.2f}ms")

        print("\n" + "="*80)
        print(f"Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Test Medical Chatbot API')
    parser.add_argument('--url', default='http://localhost:8000',
                       help='Base URL of the API (default: http://localhost:8000)')
    parser.add_argument('--test', choices=['health', 'model-info', 'validate', 'generate', 'all'],
                       default='all', help='Specific test to run (default: all)')

    args = parser.parse_args()

    tester = APITester(base_url=args.url)

    if args.test == 'all':
        tester.run_test_suite()
    elif args.test == 'health':
        tester.test_health_check()
    elif args.test == 'model-info':
        tester.test_model_info()
    elif args.test == 'validate':
        sample_patient = {
            "age": "45",
            "gender": "male",
            "chief_complaint": "chest pain"
        }
        tester.test_validate_data(sample_patient)
    elif args.test == 'generate':
        sample_patient = {
            "age": "45",
            "gender": "male",
            "chief_complaint": "chest pain and shortness of breath"
        }
        tester.test_generate_response(sample_patient)


if __name__ == "__main__":
    main()
