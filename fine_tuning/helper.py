from pathlib import Path

from datasets import load_dataset

from .config import settings


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "data" / "sample_training.jsonl"


def _resolve_dataset_path() -> Path:
    if settings.dataset_path:
        candidate = Path(settings.dataset_path).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Dataset file not found at {candidate}. "
            "Update DATASET_PATH to a valid file."
        )

    if DEFAULT_DATASET_PATH.exists():
        return DEFAULT_DATASET_PATH

    raise RuntimeError(
        f"Dataset path not configured. Set DATASET_PATH, or place the dataset at {DEFAULT_DATASET_PATH}."
    )


class DataHandle:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def create_dataset(self, test_split: float = 0.1, seed: int = 42):
        data_file = self.dataset_path
        if not data_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found at {data_file}. "
                "Update DATASET_PATH to point to your JSONL dataset."
            )

        full_dataset = load_dataset("json", data_files=str(data_file), split="train")

        dataset_split = full_dataset.train_test_split(test_size=test_split, seed=seed)
        train_dataset = dataset_split["train"]
        test_dataset = dataset_split["test"]
        print(f"length of the training dataset {len(train_dataset)}")
        print(f"length of the testing dataset {len(test_dataset)}")

        return train_dataset, test_dataset


data_handler = DataHandle(_resolve_dataset_path())
train_dataset, test_dataset = data_handler.create_dataset()
    
