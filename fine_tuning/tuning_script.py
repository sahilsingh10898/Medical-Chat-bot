import json
import torch
import traceback
import numpy as np
from transformers import pipeline
from huggingface_hub import login
from transformers import AutoTokenizer,AutoModelForCausalLM, DataCollatorForSeq2Seq , DataCollatorForLanguageModeling,TrainingArguments,EarlyStoppingCallback,Trainer
from datasets import load_dataset, DatasetDict
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training

from fine_tuning.helper import data_handler, train_dataset, test_dataset






class FineTuning:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.train_dataset= None
        self.valid_dataset = None



    def load_model_and_tokenize(self):


        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True)

        # we need to define the padding here gemma uses the EOS as pad
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side ='right'


        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16)


        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False # required for gradient checkpoint

        self.model.gradient_checkpointing_enable()
        print(f"Model and tokenizer loaded successfully")

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

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"],
            bias="none",
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def fine_tune(self, epochs=3, lr=2e-4, batch_size=4, grad_accum=8, resume_from_checkpoint=None,max_seq_length=1024):


        effective_batch_size = batch_size * grad_accum
        total_steps = (len(self.train_dataset) // effective_batch_size) * epochs

        warmup_steps = min(100, int(total_steps *0.10))

        eval_steps = max(50, total_steps // 10)

        save_steps = eval_steps

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
            bf16=True,  # Better for Gemma

            # Memory optimizations
            gradient_checkpointing=True,
            max_grad_norm=1.0,  # Gradient clipping for stability

            # Logging and saving
            logging_steps=10,
            logging_dir=f"{self.output_dir}/logs",
            save_steps=save_steps,
            save_total_limit=2,  # Keep only 2 checkpoints to save space

            # Evaluation
            eval_strategy="steps",
            eval_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Data handling
            dataloader_pin_memory=False,  # Can cause issues on Colab
            dataloader_num_workers=0,  # Safer for Colab
            remove_unused_columns=True,
            seed=42,
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

    def save_model(self,trainer=None):

      final_model_path = f"{self.output_dir}/final model"
      if trainer:
        trainer.save_model(final_model_path)
      else:
        self.model.save_pretrained(final_model_path)

      # save the tokenizer
      self.tokenizer.save_pretrained(final_model_path)
      print(f"Model saved to {final_model_path}")
      gc.collect()
      torch.cuda.empty_cache()


    