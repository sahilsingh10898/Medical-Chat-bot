from datasets import load_dataset, DatasetDict
import os
import json

DATASET_PATH = os.getenv("/DATA/dataset/train_protocol_bot_combined.jsonl")

class DataHandle:
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path

    def create_dataset(dataset_path ,test_split=0.1, seed=42):
        
        # Load the whole dataset as a HuggingFace Dataset object
        full_dataset = load_dataset("json", data_files=dataset_path, split="train")

        dataset_split = full_dataset.train_test_split(test_size=test_split, seed=seed)
        train_dataset = dataset_split["train"]
        test_dataset = dataset_split["test"]
        print(f"length of the training dataset {len(train_dataset)}")
        print(f"length of the testing dataset {len(test_dataset)}")

        return train_dataset, test_dataset



data_handler = DataHandle(DATASET_PATH)
train_dataset, test_dataset = data_handler.create_dataset()
    


