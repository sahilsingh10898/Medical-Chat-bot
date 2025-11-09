import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import defaultdict
from datetime import datetime



from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction




class ModelTester:
    def __init__(self, model_path, use_quantization=False):
        """
        Initialize the model tester

        Args:
            model_path: Path to the fine-tuned model
            use_quantization: Enable 4-bit quantization for faster inference (requires bitsandbytes)
        """
        self.model_path = model_path
        self.use_quantization = use_quantization

        self.tokenizer = None
        self.model = None
        #self.test_dataset = None
        self.results = defaultdict(list)
        
    def load_model(self):

        print(f"loading the model from the path{self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Prepare model loading arguments
        model_kwargs = {
            'device_map': 'auto',
            'trust_remote_code': True,
        }

        # Add quantization config if enabled
        if self.use_quantization:
            print("Using 4-bit quantization for faster inference...")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs['quantization_config'] = quantization_config
        else:
            model_kwargs['torch_dtype'] = torch.bfloat16

        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs
        )

        self.model.eval()
        self.model = torch.compile(self.model, mode="reduce-overhead")

    def ask_questions(self, patient_data, system_prompt=None, max_new_tokens=512):
        # define a chat template

        if system_prompt is None:
            system_prompt = "Analyze the following patient case and output the common protocols."

        if isinstance(patient_data, dict):
            patient_data = self._format_patient_data(patient_data)
        else:
            patient_data = patient_data

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": patient_data
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=None,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  
            )

        # Decode only the newly generated tokens (not the input prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up the response
        response = response.strip()

        return response

    def generate_responses(self, system_prompt=None):

        results = []


        while True:
            print("\n" + "─"*70)
            print("Enter patient case:")
            
            age = input("  Age (years): ").strip()
            if age.lower() in ['quit', 'exit', 'q']:
                break
            
            gender = input("  Gender: ").strip()
            cc = input("  Chief Complaint: ").strip()
            
            # Vitals
            print("  Vitals (press Enter to skip any):")
            bp = input("    BP (e.g., 130/80): ").strip()
            temp = input("    Temperature (e.g., 101°F): ").strip()
            hb = input("    Hemoglobin (e.g., 13.5): ").strip()
            #bp = input("    Diastolic BP (e.g., 80): ").strip()
            spo2 = input("    SpO2 (e.g., 94%): ").strip()
            rbs = input("    RBS (e.g., 150 mg/dL): ").strip()
            
            pmh = input("  Past Medical History: ").strip()
            if not pmh:
                pmh = "No significant past medical history"
            
            # Build vitals dict
            vitals = {}
            if bp: vitals['bp'] = bp
            if temp: vitals['temperature'] = temp
            if hb: vitals['hemoglobin'] = hb
            if spo2: vitals['spo2'] = spo2
            if rbs: vitals['rbs'] = rbs
            #if bp: vitals['diastolic_bp'] = bp
            
            # Create patient data
            patient_data = {
                'age': age,
                'gender': gender,
                'chief_complaint': cc,
                'vitals': vitals,
                'past_medical_history': pmh
            }
            print(patient_data)
            
            print("\n Analyzing case...\n")
            
            try:
                response = self.ask_questions(patient_data, system_prompt)
                
                print("─"*70)
                print(" PROTOCOL OUTPUT:")
                print("─"*70)
                print(response)
                print("─"*70)
                
                # Save result
                result = {
                    "patient_data": patient_data,
                    "protocol": response,
                    "timestamp": datetime.now().isoformat()
                }
                
                
                
                results.append(result)
                
            except Exception as e:
                print(f"\n Error: {e}")
                continue
        
        
        
        return results

        

       

    def _format_patient_data(self, data):
        """Format patient data dictionary into expected string format"""
        
        # Extract fields
        age = data.get('age', 'Unknown')
        gender = data.get('gender', 'Unknown')
        chief_complaint = data.get('chief_complaint', data.get('cc', ''))
        vitals = data.get('vitals', {})
        pmh = data.get('past_medical_history', data.get('pmh', 'No significant past medical history'))
        
        # Format vitals
        vital_strs = []
        if isinstance(vitals, dict):
            if 'bp' in vitals:
                vital_strs.append(f"bp {vitals['bp']}")
            if 'temperature' in vitals or 'temp' in vitals:
                temp = vitals.get('temperature', vitals.get('temp'))
                vital_strs.append(f"temperature {temp}")
            if 'hemoglobin' in vitals or 'hb' in vitals:
                hb = vitals.get('hemoglobin', vitals.get('hb'))
                vital_strs.append(f"hemoglobin {hb}")
            if 'spo2' in vitals:
                spo2 = vitals.get('spo2')
                vital_strs.append(f"SpO2 {spo2}")
            if 'rbs' in vitals:
                vital_strs.append(f"RBS {vitals['rbs']}")

                
        elif isinstance(vitals, str):
            vital_strs.append(vitals)
        
        vitals_str = ", ".join(vital_strs) if vital_strs else "Not recorded"
        
        # Build formatted string (matching training format)
        formatted = f"""Patient: {age}y {gender}
                        CC: {chief_complaint}
                        Vitals: {vitals_str}
                        PMH: {pmh}
                        Protocol?"""
        
        return formatted


        
def main():

    model_path = "/home/ubuntu/logs/final_model_complete"
    tester = ModelTester(model_path, use_quantization=True)
    tester.load_model()
    test_results = tester.generate_responses()
    print(test_results)


if __name__ == "__main__":
    main()
    
    
        
        
    
        
    
    
    
    
    
    
    
    
    
    