# Thermal Adaptive Protein Predictor

A deep learning model for predicting and generating protein sequences conditioned on temperature using masked language modeling (MLM) and transformer architecture.

## Overview

This project implements a temperature-aware protein language model that learns the relationship between amino acid sequences and their thermal stability. The model uses a transformer-based architecture with temperature conditioning to understand how proteins adapt to different thermal environments.

## Features

- **Temperature-Conditioned Generation**: Generate protein sequences optimized for specific temperature ranges (0-100°C)
- **Masked Language Modeling**: Pre-train on protein sequences using MLM objective
- **Transformer Architecture**: Efficient multi-head attention with 6 layers by default
- **PyTorch Lightning**: Built on Lightning for easy training and distributed computing
- **Adaptive Learning**: Cosine annealing with warm restarts for optimal convergence

## Model Architecture

The model consists of several key components:

- **Token Embeddings**: Learned representations for 22 amino acid tokens
- **Positional Embeddings**: Position information for sequences up to 512 residues
- **Temperature Embeddings**: Normalized temperature conditioning (0-100°C range)
- **Transformer Blocks**: 6-layer transformer with 8 attention heads (configurable)
- **MLM Head**: Prediction head for masked token reconstruction

## Installation

```bash
pip install torch pytorch-lightning numpy
```

### Data Format

The model expects batches with the following structure:

```python
batch = {
    'input_ids': torch.tensor,       # Shape: (batch_size, seq_length)
    'labels': torch.tensor,          # Shape: (batch_size, seq_length), -100 for non-masked
    'temperature': torch.tensor,     # Shape: (batch_size,)
    'attention_mask': torch.tensor   # Shape: (batch_size, seq_length)
}
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 22 | Number of amino acid tokens |
| `d_model` | 256 | Model dimension |
| `nhead` | 8 | Number of attention heads |
| `num_layers` | 6 | Number of transformer blocks |
| `max_length` | 512 | Maximum sequence length |
| `temp_range` | (0, 100) | Temperature range in °C |
| `lr` | 1e-4 | Learning rate |

## Model Details

### Temperature Conditioning

Temperature is normalized to [0, 1] range and projected to the model dimension. This conditioning vector is added to the token and positional embeddings, allowing the model to learn temperature-dependent amino acid patterns.

### Training Objective

The model uses standard MLM where 15% of tokens are masked, and the model learns to predict them based on context and temperature conditions.

## Use Cases

- Generate thermostable proteins for industrial applications
- Predict amino acid mutations for thermal adaptation
- Study protein evolution across thermal environments
- Design cold-adapted enzymes for low-temperature processes

## Future Improvements

- Multi-property conditioning (pH, salinity, pressure)
- Structure-aware attention mechanisms
- Integration with protein structure prediction
- Transfer learning from large protein language models
