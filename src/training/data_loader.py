import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from pathlib import Path

class QTGTextDataset(Dataset):
    """
    Dataset for QTG model training
    Handles text tokenization and preprocessing
    """

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_length: int = 512,
        stride: int = 256,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize dataset

        Args:
            data_path (str): Path to data file (JSONL format)
            tokenizer: Tokenizer (will use basic tokenization if None)
            max_length (int): Maximum sequence length
            stride (int): Stride for sliding window
            cache_dir (str, optional): Cache directory for processed data
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.cache_dir = cache_dir

        # Load and process data
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and preprocess data"""
        cache_file = None
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"processed_{Path(self.data_path).stem}.pt")

        # Try to load from cache
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            return torch.load(cache_file)

        # Load raw data
        data = []
        print(f"Loading data from {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    data.append(item)

        # Process data
        processed_data = []
        for item in data:
            text = item.get('text', '')

            # Basic tokenization (replace with proper tokenizer)
            tokens = self._basic_tokenize(text)

            # Create sliding windows or pad/truncate if needed
            if len(tokens) >= self.max_length:
                # Create sliding windows for long sequences
                for i in range(0, len(tokens) - self.max_length + 1, self.stride):
                    chunk = tokens[i:i + self.max_length]
                    processed_data.append({
                        'tokens': chunk,
                        'text': text
                    })
            elif len(tokens) > 0:
                # Pad short sequences
                chunk = tokens + [0] * (self.max_length - len(tokens))  # Pad with zeros
                processed_data.append({
                    'tokens': chunk,
                    'text': text
                })

        # Cache processed data
        if cache_file:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(processed_data, cache_file)
            print(f"Cached processed data to {cache_file}")

        return processed_data

    def _basic_tokenize(self, text: str) -> List[int]:
        """Basic tokenization (placeholder for real tokenizer)"""
        # This is a placeholder - replace with proper tokenization
        # For now, just split by spaces and assign arbitrary token IDs

        words = text.split()
        # Mock token IDs (0-29999 for vocab size 30000)
        tokens = [hash(word) % 30000 for word in words]

        return tokens[:self.max_length]  # Truncate if too long

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]['tokens']

        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)

        # For language modeling, labels are the same as input_ids
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'labels': labels,
            'text': self.data[idx]['text']
        }


class MockQTGTokenizer:
    """
    Mock tokenizer for development and testing
    Replace with proper tokenizer (e.g., GPT-2, BERT) in production
    """

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs"""
        # Basic word-level tokenization
        words = text.split()
        tokens = [hash(word) % (self.vocab_size - 100) + 100 for word in words]  # Reserve first 100 tokens

        if max_length:
            tokens = tokens[:max_length]

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        # Mock decoding - just return token IDs as string
        return f"Tokens: {token_ids}"

    def __call__(self, text: str, return_tensors: str = "pt", max_length: Optional[int] = None) -> torch.Tensor:
        """Tokenizer interface compatible with transformers"""
        tokens = self.encode(text, max_length)
        return torch.tensor(tokens).unsqueeze(0)  # Add batch dimension


def create_mock_dataset(
    num_samples: int = 1000,
    max_length: int = 512,
    output_path: str = "data/mock_dataset.jsonl"
) -> str:
    """
    Create mock dataset for testing

    Args:
        num_samples (int): Number of samples to generate
        max_length (int): Maximum sequence length
        output_path (str): Output file path

    Returns:
        str: Path to created dataset
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Simple mock texts
    templates = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Quantum computing uses quantum mechanics for computation.",
        "Deep learning models can process complex patterns in data.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning trains agents through trial and error.",
        "Neural networks are inspired by biological brain structures.",
        "Data science combines statistics and programming for insights.",
        "Artificial intelligence is transforming many industries today."
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # Create varied text by combining templates
            text = " ".join(templates[i % len(templates)].split() * ((i % 5) + 1))
            text = text[:max_length * 10]  # Rough character limit

            item = {
                'text': text,
                'id': i,
                'source': 'mock'
            }

            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Created mock dataset with {num_samples} samples at {output_path}")
    return output_path


def load_qtg_dataset(
    data_path: str,
    tokenizer=None,
    max_length: int = 512,
    stride: int = 256,
    cache_dir: str = "data/cache"
) -> QTGTextDataset:
    """
    Load QTG dataset with proper configuration

    Args:
        data_path (str): Path to data file
        tokenizer: Tokenizer to use
        max_length (int): Maximum sequence length
        stride (int): Sliding window stride
        cache_dir (str): Cache directory

    Returns:
        QTGTextDataset: Configured dataset
    """
    if tokenizer is None:
        tokenizer = MockQTGTokenizer()

    return QTGTextDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        cache_dir=cache_dir
    )


def create_train_val_split(
    dataset_path: str,
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[str, str]:
    """
    Create train/validation split from dataset

    Args:
        dataset_path (str): Path to original dataset
        train_ratio (float): Ratio of training data
        seed (int): Random seed

    Returns:
        Tuple[str, str]: Paths to train and validation files
    """
    torch.manual_seed(seed)

    # Read all lines
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Shuffle
    indices = torch.randperm(len(lines))
    train_size = int(len(lines) * train_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create output paths
    base_path = os.path.splitext(dataset_path)[0]
    train_path = f"{base_path}_train.jsonl"
    val_path = f"{base_path}_val.jsonl"

    # Write train split
    with open(train_path, 'w', encoding='utf-8') as f:
        for idx in train_indices:
            f.write(lines[idx.item()])

    # Write validation split
    with open(val_path, 'w', encoding='utf-8') as f:
        for idx in val_indices:
            f.write(lines[idx.item()])

    print(f"Created train/val split:")
    print(f"  Train: {train_path} ({len(train_indices)} samples)")
    print(f"  Val: {val_path} ({len(val_indices)} samples)")

    return train_path, val_path
