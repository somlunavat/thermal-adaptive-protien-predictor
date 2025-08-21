import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import numpy as np
from dataloader import create_dataloader, analyze_dataset, TOKEN_TO_AMINO_ACID
from mlmModel import ProteinMLM
import json
from datetime import datetime


def setup_training(datasets):
    """Setup training with smart defaults"""

    # Analyze training data
    print("Analyzing datasets...")
    train_stats = analyze_dataset(datasets['train'])
    if not train_stats:
        raise ValueError("Invalid training data")

    # Smart configuration
    temp_range = (train_stats['temp_stats']['min'], train_stats['temp_stats']['max'])
    max_length = min(int(train_stats['length_stats']['p95']), 512)  # Cap at 512

    print(f"Configuration: max_length={max_length}, temp_range={temp_range}")

    # Create dataloaders
    train_loader = create_dataloader(
        datasets['train'],
        batch_size=16,
        shuffle=True,
        max_length=max_length
    )

    val_loader = None
    if datasets.get('val') is not None:
        val_loader = create_dataloader(
            datasets['val'],
            batch_size=32,
            shuffle=False,
            max_length=max_length
        )

    if not train_loader:
        raise ValueError("Failed to create training dataloader")

    # Model configuration
    model_config = {
        'vocab_size': 22,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'max_length': max_length,
        'temp_range': temp_range,
        'lr': 5e-4
    }

    return train_loader, val_loader, model_config


def train_model(datasets, max_epochs=50):
    """Streamlined training process"""

    # Setup
    train_loader, val_loader, model_config = setup_training(datasets)

    # Create model
    model = ProteinMLM(**model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='protein-mlm-{epoch:02d}-{val_loss:.3f}' if val_loader else 'protein-mlm-{epoch:02d}',
            save_top_k=3,
            monitor='val_loss' if val_loader else 'train_loss',
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    if val_loader:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            min_delta=0.01
        ))

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=TensorBoardLogger("tb_logs", name="protein_mlm"),
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        precision=16,  # Mixed precision for efficiency
        deterministic=False,
        enable_progress_bar=True
    )

    # Training
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    return model, trainer


def test_model(test_data, checkpoint_path=None):
    """Test trained model"""

    # Analyze test data
    test_stats = analyze_dataset(test_data)
    if not test_stats:
        print("Invalid test data")
        return None

    temp_range = (test_stats['temp_stats']['min'], test_stats['temp_stats']['max'])
    max_length = min(int(test_stats['length_stats']['p95']), 512)

    # Create test loader
    test_loader = create_dataloader(
        test_data,
        batch_size=32,
        shuffle=False,
        max_length=max_length
    )

    if not test_loader:
        print("Failed to create test dataloader")
        return None

    # Load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model = ProteinMLM.load_from_checkpoint(checkpoint_path)
    else:
        # Try to find latest checkpoint
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path:
            model = ProteinMLM.load_from_checkpoint(checkpoint_path)
        else:
            print("No checkpoint found")
            return None

    # Test
    trainer = pl.Trainer(enable_progress_bar=True)
    results = trainer.test(model, test_loader, verbose=True)

    # Generate sample sequences
    generate_sample_sequences(model, temp_range)

    return results


def find_latest_checkpoint():
    """Find the latest checkpoint"""
    if not os.path.exists('checkpoints'):
        return None

    checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.ckpt')]
    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda x: os.path.getctime(os.path.join('checkpoints', x)))
    return os.path.join('checkpoints', latest)


def generate_sample_sequences(model, temp_range, output_dir='generated_sequences'):
    """Generate sample sequences at different temperatures"""
    os.makedirs(output_dir, exist_ok=True)

    temperatures = np.linspace(temp_range[0], temp_range[1], 5)
    results = {}

    print("\nGenerating sample sequences...")
    for temp in temperatures:
        print(f"Temperature: {temp:.1f}°C")
        sequences = []

        for length in [50, 100, 150]:
            try:
                seq_tokens = model.generate_sequence(temp, length=length)
                seq_string = ''.join([TOKEN_TO_AMINO_ACID.get(token, 'X') for token in seq_tokens])
                sequences.append({
                    'sequence': seq_string,
                    'length': len(seq_string),
                    'temperature': temp
                })
                print(f"  Generated {len(seq_string)} AA sequence")
            except Exception as e:
                print(f"  Generation failed: {e}")

        results[float(temp)] = sequences  # Convert numpy float to Python float for JSON

    # Save results
    with open(f'{output_dir}/sequences.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved sequences to {output_dir}/sequences.json")
    return results


def main():
    """Main training/testing pipeline"""

    print("Protein Language Model Training")
    print("=" * 50)

    # Load datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        filename = f'{split}_set.csv'
        try:
            datasets[split] = pd.read_csv(filename)
            print(f"Loaded {split}: {len(datasets[split])} samples")
        except FileNotFoundError:
            datasets[split] = None
            print(f"No {split} data found")

    # Training or testing mode
    if datasets['train'] is not None:
        print("\nTraining mode")
        try:
            model, trainer = train_model(datasets)
            print("Training completed!")

            # Test if available
            if datasets.get('test') is not None:
                print("\nTesting...")
                test_results = test_model(datasets['test'])
                print("Testing completed!")

        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

    elif datasets['test'] is not None:
        print("\nTest-only mode")
        test_results = test_model(datasets['test'])

    else:
        print("No data found! Place train_set.csv and/or test_set.csv in current directory.")


if __name__ == "__main__":
    # Windows compatibility
    if os.name == 'nt':
        torch.set_num_threads(1)
        import multiprocessing

        multiprocessing.set_start_method('spawn', force=True)

    main()