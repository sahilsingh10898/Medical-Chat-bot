import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftModel



from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


from fine_tuning.helper import data_handler, test_dataset

class ModelTester:
    def __init__(self, model_path, dataset_path, output_dir="test_results"):
        """
        Initialize the model tester
        
        Args:
            model_path: Path to the fine-tuned model
            dataset_path: Path to test dataset
            output_dir: Directory to save test results
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.test_dataset = None
        self.results = defaultdict(list)
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print("Loading model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        self.model.eval()
        
        
    def load_test_dataset(self):        
        
        self.test_dataset = load_test_data(self.dataset_path)
        print(f"✓ Test dataset loaded: {len(self.test_dataset)} examples")
        
    def compute_perplexity(self, text):
        """Compute perplexity for a given text"""
        encodings = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings.input_ids)
            loss = outputs.loss
            
        perplexity = torch.exp(loss).item()
        return perplexity, loss.item()
    
    def compute_token_accuracy(self, input_ids, labels):
        """Compute token-level accuracy (excluding masked tokens)"""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Only consider non-masked tokens (labels != -100)
        mask = labels != -100
        correct = (predictions == labels) & mask
        
        accuracy = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0
        return accuracy
    
    def compute_rouge_scores(self, prediction, reference):
        """Compute ROUGE scores"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def compute_bleu_score(self, prediction, reference):
        """Compute BLEU score"""
        # Tokenize
        pred_tokens = prediction.split()
        ref_tokens = [reference.split()]
        
        # Use smoothing for short sequences
        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        
        return bleu
    
    def generate_response(self, messages, max_new_tokens=512):
        """Generate response for given messages"""
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the model's response
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1].strip()
        
        return response
    
    def run_comprehensive_test(self, num_samples=None, save_predictions=True):
        """Run comprehensive testing on the test dataset"""
        print("\n" + "="*70)
        print("STARTING COMPREHENSIVE MODEL EVALUATION")
        print("="*70 + "\n")
        
        if num_samples:
            test_subset = self.test_dataset.select(range(min(num_samples, len(self.test_dataset))))
        else:
            test_subset = self.test_dataset
        
        all_metrics = {
            'perplexities': [],
            'losses': [],
            'rouge1_scores': [],
            'rouge2_scores': [],
            'rougeL_scores': [],
            'bleu_scores': [],
            'token_accuracies': []
        }
        
        predictions_log = []
        
        print(f"Testing on {len(test_subset)} examples...\n")
        
        for idx, example in enumerate(tqdm(test_subset, desc="Evaluating")):
            try:
                messages = example['messages']
                
                # Get ground truth (model's response)
                ground_truth = next(msg['content'] for msg in messages if msg['role'] == 'model')
                
                # Generate prediction
                prompt_messages = [msg for msg in messages if msg['role'] != 'model']
                prediction = self.generate_response(prompt_messages)
                
                # Compute all metrics
                # 1. Perplexity and loss
                full_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                perplexity, loss = self.compute_perplexity(full_text)
                all_metrics['perplexities'].append(perplexity)
                all_metrics['losses'].append(loss)
                
                # 2. ROUGE scores
                rouge_scores = self.compute_rouge_scores(prediction, ground_truth)
                all_metrics['rouge1_scores'].append(rouge_scores['rouge1'])
                all_metrics['rouge2_scores'].append(rouge_scores['rouge2'])
                all_metrics['rougeL_scores'].append(rouge_scores['rougeL'])
                
                # 3. BLEU score
                bleu = self.compute_bleu_score(prediction, ground_truth)
                all_metrics['bleu_scores'].append(bleu)
                
                # 4. Token accuracy
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                tokenized = self.tokenizer(
                    formatted_text,
                    truncation=True,
                    max_length=1024,
                    return_tensors='pt'
                ).to(self.model.device)
                
                # Create labels (simplified - you might need to adjust based on your masking logic)
                labels = tokenized['input_ids'].clone()
                
                token_acc = self.compute_token_accuracy(
                    tokenized['input_ids'],
                    labels
                )
                all_metrics['token_accuracies'].append(token_acc)
                
                # Log predictions
                if save_predictions:
                    predictions_log.append({
                        'index': idx,
                        'input': prompt_messages[-1]['content'],
                        'ground_truth': ground_truth,
                        'prediction': prediction,
                        'metrics': {
                            'perplexity': perplexity,
                            'loss': loss,
                            'rouge1': rouge_scores['rouge1'],
                            'rouge2': rouge_scores['rouge2'],
                            'rougeL': rouge_scores['rougeL'],
                            'bleu': bleu,
                            'token_accuracy': token_acc
                        }
                    })
                
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                continue
        
        # Calculate aggregate statistics
        results_summary = self._calculate_summary_stats(all_metrics)
        
        # Save results
        self._save_results(results_summary, predictions_log)
        
        # Print summary
        self._print_summary(results_summary)
        
        return results_summary, predictions_log
    
    
    
    def _save_results(self, summary, predictions_log):
        """Save results to JSON and generate plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary statistics
        summary_path = os.path.join(self.output_dir, f"test_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Summary saved to {summary_path}")
        
        # Save predictions
        predictions_path = os.path.join(self.output_dir, f"predictions_{timestamp}.json")
        with open(predictions_path, 'w') as f:
            json.dump(predictions_log, f, indent=2)
        print(f"✓ Predictions saved to {predictions_path}")
        
        # Generate plots
        self._generate_plots(summary, timestamp)
    
    
    
    
    