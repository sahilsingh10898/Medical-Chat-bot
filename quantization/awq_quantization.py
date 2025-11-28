import json
import logging
import random
import sys

from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings


    

logger = logging.getLogger(__name__)


@dataclass
class AWQQuantizationConfig:
    model_path : str
    output_path : str
    dataset_path : str
    samples : int = 512
    w_bit : int = 4
    q_group_size : int = 128
    zero_point: bool = True
    version: str = "GEMM"

    def validate(self) -> None:

        # for validating the config params
        if self.w_bit not in [4,8]:
            raise ValueError(f"w_bit value must be between 4 and 8 got {self.w_bit}")

        if self.q_group_size not in [64,128,256]:
            raise ValueError(f"q_group_size must be one of the following 64,128,256 got {self.q_group_size}")
        
        if self.samples < 128:
            logger.warning(f"samples value is too low may result in poor quantization quality got {self.samples}")

        if self.version not in ["GEMM","GEMV"]:
            raise ValueError(f"version must be one of the following GEMM or GEMV got {self.version}")
        


class AWQQuantizerModule:

    def __init__(self,config : AWQQuantizationConfig):

        self.config = config
        self.config.validate()

        # initialize paths from config
        self.path = config.model_path
        self.output_path = config.output_path
        self.dataset_path = config.dataset_path

        # load the model and tokenizer
        self.tokenizer = None
        self.model = None
        self.calibration_data = None

    def _validate_path(self):
        # validation for the model path
        if not self.path or not Path(self.path).exists():
            raise FileNotFoundError(f"Model path {self.path} does not exist.")
        if not self.dataset_path or not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist.")
    def load_tokenizer(self):
        logger.info(f"Loading tokenizer from {self.path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path,trust_remote_code=True)
        logger.info(f"Loading tokenizer successful")
    def load_calibration_data(self) -> None:
        """
        Load and prepare calibration data from JSONL file.
        Converts chat format to text format for calibration.
        """
        logger.info(f"Loading calibration data from {self.dataset_path}...")
        
        calibration_texts = []
        samples = []
        
        # Load samples from JSONL
        with open(self.dataset_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        
        # Sample if needed
        if len(samples) > self.config.samples:
            random.seed(42) 
            samples = random.sample(samples, self.config.samples)

        logger.info(f"Selected {len(samples)} samples for calibration")
        
        # Convert to text format
        for i, sample in enumerate(samples):
            try:
                messages = sample.get("messages", [])
                
                # Normalize roles (model -> assistant)
                for msg in messages:
                    if msg.get("role") == "model":
                        msg["role"] = "assistant"
                
                # Apply chat template
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                calibration_texts.append(text)
                
            except Exception as e:
                logger.warning(f"Skipping sample {i} due to error: {e}")
        
        if not calibration_texts:
            raise ValueError("No valid calibration data could be loaded!")
        
        self.calibration_data = calibration_texts
        logger.info(f" Prepared {len(self.calibration_data)} calibration texts")
    def load_model(self):
        
        self.model = AutoAWQForCausalLM.from_pretrained(
            str(self.path),
            trust_remote_code=True,
            device_map="auto"
        )

        logger.info(f"Model loaded successfully from {self.path}")
    def quantize_model(self):

        quantize_config = {
            "zero_point": self.config.zero_point,
            "w_bit": self.config.w_bit,
            "q_group_size": self.config.q_group_size,
            "version": self.config.version
        }
        try:

            self.model.quantize(
                quant_config=quantize_config,
                calib_data=self.calibration_data,
                tokenizer=self.tokenizer
            )

            logger.info("Model quantization completed successfully.")
        except Exception as e:
            logger.error(f"Error during model quantization: {e}")
            raise e
    def save_quantized_model(self):
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # AWQ models use save_quantized instead of save_pretrained
        self.model.save_quantized(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

        logger.info(f"Quantized model saved to {output_path}")

    def run(self):
        self._validate_path()
        self.load_tokenizer()
        self.load_calibration_data()
        self.load_model()
        self.quantize_model()
        self.save_quantized_model()
    

def main():
    """Main execution function for standalone usage."""
    # Configuration
    config = AWQQuantizationConfig(
        model_path="/home/ubuntu/logs/merged_model",
        output_path="/home/ubuntu/logs/quantized_awq_model",
        dataset_path="/home/ubuntu/DATA/dataset/train_protocol_bot_combined (2).jsonl",
        samples=512,
        w_bit=4,
        q_group_size=128,
        zero_point=True,
        version="GEMM"
    )

    # Create quantizer and run
    quantizer = AWQQuantizerModule(config)
    quantizer.run()


if __name__ == "__main__":
    main()




        






