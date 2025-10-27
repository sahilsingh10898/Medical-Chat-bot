import subprocess

def launch():
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen3-4B-Thinking-2507",
        "--tokenizer", "Qwen/Qwen3-4B-Thinking-2507",
        "--port", "8000",
        "--gpu-memory-utilization", "1.0",
        "--max-model-len", "32768",
        "--dtype", "float16"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    print(" Starting Qwen3-4B server using vLLM...")
    launch()



