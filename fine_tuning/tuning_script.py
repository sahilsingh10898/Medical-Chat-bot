import json
import torch
import os
import re
import traceback
import numpy as np
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from transformers import TrainingArguments, EarlyStoppingCallback, Trainer
from transformers import DataCollatorForSeq2Seq
from huggingface_hub import login
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer
import gc
from .config import settings


class FineTuning:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.train_dataset= None
        self.valid_dataset = None


    def login_huggingface(self, hf_token=None):
        try:
            token_to_use = hf_token or settings.huggingface_token
            if token_to_use:
                login(token_to_use)
                print("Logged in to huggingface successfully")
            else:
                raise ValueError("Huggingface token not provided in config.")
        except Exception as e:
            print(f"Failed to login to huggingface: {e}")
            traceback.print_exc()


    def load_model_and_tokenize(self,max_seq=1024):


        # use the unsloth to load the model and tokenizer
        self.model , self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = max_seq,
            dtype = None,
            load_in_4bit = True # loading the model in 4 bit

        )
        

    def preprocessing_dataset(self):
        """Format dataset with chat template and mask prompt tokens"""
        print("Preprocessing dataset with label masking...")
        
        def format_chat(example):
            messages = example['messages']
            
            # Apply chat template to get full conversation
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Get prompt only (everything before model's response)
            # Filter to only system and user messages
            prompt_messages = [msg for msg in messages if msg['role'] != 'model']
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True  # Adds model turn start token
            )
            
            # Tokenize both
            full_tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=1024,
                padding=False,
                return_tensors=None
            )
            
            prompt_tokenized = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=1024,
                padding=False,
                return_tensors=None
            )
            
            # Create labels - mask prompt with -100 (ignore index)
            labels = full_tokenized['input_ids'].copy()
            prompt_len = len(prompt_tokenized['input_ids'])
            
            # Mask all prompt tokens
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100
            
            return {
                "input_ids": full_tokenized['input_ids'],
                "attention_mask": full_tokenized['attention_mask'],
                "labels": labels
            }
        
        self.train_dataset = self.train_dataset.map(
            format_chat,
            remove_columns=self.train_dataset.column_names,
            desc="Formatting train dataset"
        )
        
        self.valid_dataset = self.valid_dataset.map(
            format_chat,
            remove_columns=self.valid_dataset.column_names,
            desc="Formatting validation dataset"
        )
        
        print(" Dataset formatting complete with masked labels")

    def apply_lora(self):

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
            use_rslora=False,
            loftq_config=None
            
        )
        self.model.print_trainable_parameters()

    def fine_tune(self, epochs=3, lr=1e-4, batch_size=16, grad_accum=2, resume_from_checkpoint=None,max_seq_length=1024):

        # unsloath specific training args
        # enable the tf32 datatype
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # first check wheter we have any checkpoint stored that we can use to run the training
        # checkpoint_to_resume=None

        # if checkpoint_to_resume:
        #     checkpoint_to_resume = resume_from_checkpoint
        #     print(f"resuming from check point {checkpoint_to_resume}")
        # elif auto_resume:






        effective_batch_size = batch_size * grad_accum
        total_steps = (len(self.train_dataset) // effective_batch_size) * epochs
        warmup_steps = min(100, int(total_steps *0.10))
        eval_steps = max(50, total_steps // 10)
        save_steps = eval_steps

        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Total examples: {len(self.train_dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Effective batch size: {effective_batch_size}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Eval every: {eval_steps} steps")
        print(f"{'='*60}\n")



        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,

            # Learning rate scheduling
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,

            # Optimization
            weight_decay=0.01,
            optim="paged_adamw_8bit",  # Memory efficient optimizer for Colab

            # Precision
            fp16= not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),  

            # Memory optimizations
            gradient_checkpointing='unsloth',  # Enable gradient checkpointing
            max_grad_norm=1.0,  # Gradient clipping for stability

            # Logging and saving
            logging_steps=5,
            logging_dir=f"{self.output_dir}/complete_logs",
            save_steps=save_steps,
            save_total_limit=2,  # Keep only 2 checkpoints to save space

            # Evaluation
            eval_strategy="steps",
            eval_steps=eval_steps,
            eval_on_start=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Data handling
            dataloader_pin_memory=True,  
            dataloader_num_workers=16,  
            remove_unused_columns=False,
            seed=42,

            # unsloth specific optimization
            ddp_find_unused_parameters=False,
            disable_tqdm=False,
            report_to="tensorboard",
        )

        response_template = "<start_of_turn>model\n"

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100
        )



        trainer =Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=None,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3)
            ]
        )

        print(f"Starting training with effective batch size: {effective_batch_size}")
        print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

        print("Testing data collator...")
        try:
            sample_batch = [self.train_dataset[i] for i in range(batch_size)]
            collated = data_collator(sample_batch)
            print(f"✓ Data collator works! Batch shape: {collated['input_ids'].shape}")
        except Exception as e:
            print(f" Data collator failed: {e}")
            traceback.print_exc()
            
            

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print("Fine-tuning complete.")
        self.save_model()

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation
        Returns perplexity and token accuracy
        """
        predictions, labels = eval_pred
        
        # Predictions are logits, take argmax to get predicted tokens
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Convert logits to predictions
        predictions = np.argmax(predictions, axis=-1)
        
        # Flatten arrays and remove padding (-100 labels)
        predictions_flat = predictions.flatten()
        labels_flat = labels.flatten()
        
        # Create mask for valid labels (not -100)
        mask = labels_flat != -100
        
        # Filter out padding
        predictions_valid = predictions_flat[mask]
        labels_valid = labels_flat[mask]
        
        # Calculate token accuracy
        accuracy = (predictions_valid == labels_valid).mean()
        
        # Loss is automatically computed by Trainer
        # We can access it through eval_pred but it's already tracked
        
        return {
            "accuracy": float(accuracy),
            # Perplexity will be computed from eval_loss automatically
        }


    def save_model(self,trainer=None):

        final_model_path = f"{self.output_dir}/final model"
        if trainer:
            self.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
        else:
            self.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)

        print(f"Model saved to {final_model_path}")
        gc.collect()
        torch.cuda.empty_cache()


    # lets manually add a method to find the checkpoints and resume the training from there
    def find_checkpoint(self):
        if not os.path.exists(self.output_dir):
            return None
        checkpoint_dirs = [
            d for d in os.listdir(self.output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(self.output_dir, d))
        ]

        if not checkpoint_dirs:
            return None
        # sort them by latest and use that
        checkpoint_dirs.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        latest_checkpoint = os.path.join(self.output_dir,checkpoint_dirs[-1])

        return latest_checkpoint




    
