# Backend Testing Guide

This guide explains how to test your Medical Chatbot backend and view model outputs.

## Prerequisites

1. **Install dependencies** (if not already installed):
```bash
pip install requests
```

2. **Start the FastAPI server**:
```bash
cd /home/ubuntu/Medical-Chat-bot/backend/app
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Or with auto-reload for development:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Test Scripts

### 1. Quick Test (`quick_test.py`)

**Purpose**: Fast, simple testing to see model outputs

**Usage**:

Run all predefined test cases:
```bash
python backend/quick_test.py
```

Run a specific test case (1-5):
```bash
python backend/quick_test.py --case 1
```

Test with custom patient data:
```bash
python backend/quick_test.py \
  --age "45" \
  --gender "male" \
  --complaint "chest pain and shortness of breath"
```

**Test Cases Available**:
- Case 1: Chest Pain
- Case 2: Diabetes with High Blood Sugar
- Case 3: Acute Abdominal Pain
- Case 4: Respiratory Infection
- Case 5: Fever and Rash

**Example Output**:
```
================================================================================
PATIENT INPUT
================================================================================
Age: 45
Gender: male
Chief Complaint: chest pain and shortness of breath

================================================================================
MODEL OUTPUT (PROTOCOL)
================================================================================
[Your model's protocol recommendation will appear here]
================================================================================

Inference Time: 1234.5ms
Model: /home/ubuntu/logs/final model
Temperature: 0.2
```

---

### 2. Full Test Suite (`test_api.py`)

**Purpose**: Comprehensive testing of all API endpoints with detailed reporting

**Usage**:

Run complete test suite:
```bash
python backend/test_api.py
```

Test specific endpoints:
```bash
# Health check only
python backend/test_api.py --test health

# Model info only
python backend/test_api.py --test model-info

# Data validation only
python backend/test_api.py --test validate

# Response generation only
python backend/test_api.py --test generate
```

Custom API URL:
```bash
python backend/test_api.py --url http://192.168.1.100:8000
```

**What it tests**:
- ✅ `/health` - Server health and model status
- ✅ `/model-info` - Model configuration
- ✅ `/validate-data` - Patient data validation
- ✅ `/generate-response` - Protocol generation (5 different cases)

**Example Output**:
```
================================================================================
  MEDICAL CHATBOT API TEST SUITE
================================================================================
Base URL: http://localhost:8000
Test Started: 2024-11-11 10:30:00

--------------------------------------------------------------------------------
  Testing /health endpoint
--------------------------------------------------------------------------------
Status Code: 200
Response: {
  "status": "healthy",
  "model_status": "initialized",
  "model_path": "/home/ubuntu/logs/final model"
}

...

================================================================================
  TEST SUMMARY
================================================================================

Total Tests: 8
Passed: 8
Failed: 0
Success Rate: 100.0%

Inference Performance:
  Test Case 1:
    Server Inference Time: 1234.5ms
    Client Total Latency: 1245.67ms
  Test Case 2:
    Server Inference Time: 1456.7ms
    Client Total Latency: 1467.89ms
  ...

  Average Server Inference Time: 1345.6ms
```

---

## Direct API Testing with curl

### Health Check
```bash
curl http://localhost:8000/health
```

### Model Info
```bash
curl http://localhost:8000/model-info
```

### Validate Patient Data
```bash
curl -X POST http://localhost:8000/validate-data \
  -H "Content-Type: application/json" \
  -d '{
    "age": "45",
    "gender": "male",
    "chief_complaint": "chest pain"
  }'
```

### Generate Protocol
```bash
curl -X POST http://localhost:8000/generate-response \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "age": "45",
      "gender": "male",
      "chief_complaint": "chest pain and shortness of breath",
      "vitals": {
        "bp": "140/90",
        "spo2": "94%"
      }
    },
    "include_input": true
  }'
```

---

## Python Code Testing

You can also test programmatically:

```python
import requests

# Test patient data
patient_data = {
    "age": "45",
    "gender": "male",
    "chief_complaint": "chest pain and shortness of breath",
    "vitals": {
        "bp": "140/90",
        "spo2": "94%"
    }
}

# Generate response
response = requests.post(
    "http://localhost:8000/generate-response",
    json={
        "patient_data": patient_data,
        "include_input": True
    }
)

result = response.json()
print(result['protocol'])
```

---

## Understanding the Response

### Success Response Structure
```json
{
  "protocol": "1. Administer aspirin 325mg...",
  "patient_summary": {
    "age": "45y",
    "gender": "male",
    "chief_complaint": "chest pain and shortness of breath",
    "vitals": {
      "bp": "140/90",
      "spo2": "94%"
    },
    "past_medical_history": "No significant past medical history"
  },
  "metadata": {
    "inference_time_ms": 1234.5,
    "model_path": "/home/ubuntu/logs/final model",
    "temperature": 0.2,
    "max_tokens": 512
  },
  "timestamp": "2024-11-11T10:30:00.123456"
}
```

### Error Response Structure
```json
{
  "error": "ValidationError",
  "detail": "Chief complaint cannot be empty",
  "timestamp": "2024-11-11T10:30:00.123456"
}
```

---

## Troubleshooting

### Server not running
```
❌ ERROR: Could not connect to the API server
```
**Solution**: Start the server first:
```bash
cd backend/app
python -m uvicorn main:app --reload
```

### Model not loaded
```json
{
  "status": "unhealthy",
  "model_status": "not initialized"
}
```
**Solution**: Check the model path in `config.py` and ensure the model files exist

### Validation errors
```
HTTPError: 422 Unprocessable Entity
```
**Solution**: Check your patient data format:
- Age must be between 10-70
- Gender must be "male" or "female"
- Chief complaint cannot be empty

---

## Performance Benchmarking

To benchmark your model's performance, use the full test suite:

```bash
python backend/test_api.py > test_results.txt
```

Key metrics to monitor:
- **Server Inference Time**: Time taken by model to generate protocol
- **Client Total Latency**: Total round-trip time including network
- **Success Rate**: Percentage of successful requests

---

## Next Steps

1. **Customize test cases**: Edit the test scripts to add your own patient scenarios
2. **Load testing**: Use tools like `locust` or `ab` for concurrent request testing
3. **Integration testing**: Connect your frontend to test the full pipeline
4. **Monitor logs**: Check server logs for detailed inference information

---

## Additional Resources

- API Documentation: http://localhost:8000/docs (when server is running)
- FastAPI Docs: https://fastapi.tiangolo.com/
- vLLM Docs: https://docs.vllm.ai/
