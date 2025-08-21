import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# Simplified amino acid mapping
AMINO_ACID_TO_TOKEN = {
    'A': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'K': 10,
    'L': 11, 'M': 12, 'N': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17, 'T': 18,
    'V': 19, 'W': 20, 'Y': 21, '<PAD>': 0, '<MASK>': 1
}

TOKEN_TO_AMINO_ACID = {v: k for k, v in AMINO_ACID_TO_TOKEN.items()}
VOCAB_SIZE = 22


class ProteinDataset(Dataset):
    def __init__(self, data, max_length=512, mask_prob=0.15):
        # Clean and validate data
        valid_data = self._filter_data(data, max_length)

        self.sequences = valid_data['sequences']
        self.temperatures = valid_data['temperatures']
        self.max_length = max_length
        self.mask_prob = mask_prob

        # Temperature normalization
        temps = np.array(self.temperatures)
        self.temp_mean = temps.mean()
        self.temp_std = max(temps.std(), 1.0)

        print(f"Dataset: {len(self.sequences)} sequences, "
              f"temp range: [{temps.min():.1f}, {temps.max():.1f}]°C")

    def _filter_data(self, data, max_length):
        """Efficiently filter and clean data"""
        # Basic filtering
        mask = (data['Full Sequence'].notna() &
                (data['Organism Temperature'] != 'Unknown') &
                (data['Organism Temperature'].notna()))
        clean_data = data[mask].copy()

        sequences, temperatures = [], []
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')

        for _, row in clean_data.iterrows():
            seq = str(row['Full Sequence']).upper()
            try:
                temp = float(row['Organism Temperature'])
                # Validate sequence and length
                if (30 <= len(seq) <= max_length and
                        all(aa in valid_aa for aa in seq)):
                    sequences.append(seq)
                    temperatures.append(temp)
            except (ValueError, TypeError):
                continue

        return {'sequences': sequences, 'temperatures': temperatures}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        temperature = self.temperatures[idx]

        # Tokenize
        tokens = [AMINO_ACID_TO_TOKEN[aa] for aa in sequence]

        # Create masked version for MLM
        input_ids, labels = self._create_mlm_sample(tokens)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'temperature': torch.tensor(temperature, dtype=torch.float),
            'length': len(input_ids)
        }

    def _create_mlm_sample(self, tokens):
        """Create masked language modeling sample"""
        input_ids = tokens.copy()
        labels = [-100] * len(tokens)  # -100 = ignore in loss

        # Randomly mask tokens
        for i, token in enumerate(tokens):
            if np.random.random() < self.mask_prob:
                labels[i] = token  # Store original token

                prob = np.random.random()
                if prob < 0.8:  # 80% mask
                    input_ids[i] = AMINO_ACID_TO_TOKEN['<MASK>']
                elif prob < 0.9:  # 10% random
                    input_ids[i] = np.random.randint(2, VOCAB_SIZE)
                # 10% keep original

        return input_ids, labels


def collate_fn(batch):
    """Efficient batch collation with padding"""
    max_len = max(item['length'] for item in batch)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    temperatures = torch.tensor([item['temperature'] for item in batch])

    for i, item in enumerate(batch):
        length = item['length']
        input_ids[i, :length] = item['input_ids']
        labels[i, :length] = item['labels']
        attention_mask[i, :length] = True

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'temperature': temperatures
    }


def create_dataloader(data, batch_size=16, shuffle=True, max_length=512):
    """Create optimized dataloader"""
    if data is None or len(data) == 0:
        return None

    try:
        dataset = ProteinDataset(data, max_length=max_length)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,  # Keep 0 for stability
            pin_memory=False
        )
    except Exception as e:
        print(f"DataLoader creation failed: {e}")
        return None


def analyze_dataset(data, sample_size=1000):
    """Quick dataset analysis"""
    if data is None:
        return None

    # Sample for efficiency
    sample = data.sample(min(sample_size, len(data)))

    # Filter valid data
    valid_mask = (sample['Full Sequence'].notna() &
                  sample['Organism Temperature'].notna())
    valid_data = sample[valid_mask]

    if len(valid_data) == 0:
        return None

    # Convert temperatures
    temps = pd.to_numeric(valid_data['Organism Temperature'], errors='coerce')
    temps = temps.dropna()

    # Sequence lengths
    seq_lengths = valid_data['Full Sequence'].str.len()

    print(f"Analysis of {len(valid_data)} samples:")
    print(f"Temperature: {temps.min():.1f}-{temps.max():.1f}°C (μ={temps.mean():.1f})")
    print(f"Seq length: {seq_lengths.min()}-{seq_lengths.max()} (μ={seq_lengths.mean():.1f})")

    return {
        'temp_stats': {
            'min': temps.min(), 'max': temps.max(),
            'mean': temps.mean(), 'std': temps.std()
        },
        'length_stats': {
            'min': seq_lengths.min(), 'max': seq_lengths.max(),
            'mean': seq_lengths.mean(), 'p95': seq_lengths.quantile(0.95)
        }
    }