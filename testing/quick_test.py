#!/usr/bin/env python3
"""
Quick test script to check model outputs
Simple and focused on showing what the model generates
"""

import requests
import json
import sys


def test_patient_case(patient_data: dict, base_url: str = "http://localhost:8000"):
    """
    Quick test function to generate and display model output

    Args:
        patient_data: Dictionary with patient information
        base_url: API base URL
    """
    print("\n" + "="*80)
    print("PATIENT INPUT")
    print("="*80)

    # Display patient info
    print(f"Age: {patient_data.get('age')}")
    print(f"Gender: {patient_data.get('gender')}")
    print(f"Chief Complaint: {patient_data.get('chief_complaint')}")

    if patient_data.get('vitals'):
        print(f"Vitals: {json.dumps(patient_data['vitals'], indent=2)}")

    if patient_data.get('past_medical_history'):
        print(f"Past Medical History: {patient_data['past_medical_history']}")

    print("\n" + "="*80)
    print("GENERATING MODEL OUTPUT...")
    print("="*80)

    try:
        # Make API request
        response = requests.post(
            f"{base_url}/generate-response",
            json={
                "patient_data": patient_data,
                "include_input": True
            },
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        # Display results
        print("\n" + "="*80)
        print("MODEL OUTPUT (PROTOCOL)")
        print("="*80)
        print(result.get('protocol', 'No output generated'))
        print("="*80)

        # Display metadata
        if result.get('metadata'):
            metadata = result['metadata']
            print(f"\nInference Time: {metadata.get('inference_time_ms')}ms")
            print(f"Model: {metadata.get('model_path')}")
            print(f"Temperature: {metadata.get('temperature')}")

        return result

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API server")
        print(f"   Make sure the server is running at {base_url}")
        print("   Start it with: cd backend/app && python -m uvicorn main:app --reload")
        sys.exit(1)

    except requests.exceptions.Timeout:
        print("\n❌ ERROR: Request timed out")
        print("   The model might be taking too long to respond")
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        print(f"\n❌ ERROR: HTTP {e.response.status_code}")
        print(f"   {e.response.text}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        sys.exit(1)


def main():
    """Run predefined test cases or custom input"""
    import argparse

    parser = argparse.ArgumentParser(description='Quick test for Medical Chatbot')
    parser.add_argument('--url', default='http://localhost:8000',
                       help='API base URL (default: http://localhost:8000)')
    parser.add_argument('--case', type=int, choices=[1, 2, 3, 4, 5],
                       help='Predefined test case number (1-5)')
    parser.add_argument('--age', help='Patient age')
    parser.add_argument('--gender', choices=['male', 'female'], help='Patient gender')
    parser.add_argument('--complaint', help='Chief complaint')

    args = parser.parse_args()

    # Predefined test cases
    test_cases = {
        1: {
            "name": "Chest Pain",
            "data": {
                "age": "45",
                "gender": "male",
                "chief_complaint": "chest pain and shortness of breath",
                "vitals": {
                    "bp": "140/90",
                    "spo2": "94%"
                }
            }
        },
        2: {
            "name": "Diabetes with High Blood Sugar",
            "data": {
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
        },
        3: {
            "name": "Acute Abdominal Pain",
            "data": {
                "age": "35",
                "gender": "male",
                "chief_complaint": "severe abdominal pain in right lower quadrant, nausea, vomiting",
                "vitals": {
                    "bp": "120/80",
                    "temperature": "102°F",
                    "spo2": "98%"
                }
            }
        },
        4: {
            "name": "Respiratory Infection",
            "data": {
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
        },
        5: {
            "name": "Fever and Rash",
            "data": {
                "age": "22",
                "gender": "female",
                "chief_complaint": "high fever, skin rash, body aches",
                "vitals": {
                    "bp": "110/70",
                    "temperature": "103°F",
                    "spo2": "97%"
                }
            }
        }
    }

    # Custom patient data from command line
    if args.age and args.gender and args.complaint:
        print("\n📋 Testing with CUSTOM patient data")
        patient_data = {
            "age": args.age,
            "gender": args.gender,
            "chief_complaint": args.complaint
        }
        test_patient_case(patient_data, args.url)

    # Predefined test case
    elif args.case:
        case = test_cases[args.case]
        print(f"\n📋 Testing Case {args.case}: {case['name']}")
        test_patient_case(case['data'], args.url)

    # Run all test cases
    else:
        print("\n📋 Running ALL test cases...\n")
        for case_num, case in test_cases.items():
            print(f"\n{'#'*80}")
            print(f"# TEST CASE {case_num}: {case['name']}")
            print(f"{'#'*80}")
            test_patient_case(case['data'], args.url)
            print("\n")


if __name__ == "__main__":
    main()
