
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_adapter(
    base_model_name: str = "meta-llama/Llama-3.2-3B-Instruct", 
    adapter_path: str = "/home/ubuntu/logs/final_model_complete",
    output_path: str = "/home/ubuntu/logs/merged_model"
):
    
    print(f"Loading base model: {base_model_name}")

    # Load base model in fp16 (quantized models can't be merged)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {adapter_path}")

    # Load the LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        trust_remote_code=True
    )

    print("Merging adapter with base model...")

    # Merge the adapter weights into the base model
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")

    # Save the merged model
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="2GB"
    )

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,  # Use adapter path as it has the tokenizer
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)

    print(f" merged model saved to: {output_path}")
    print(f'vllm_model = "{output_path}"')

if __name__ == "__main__":
    merge_adapter()
