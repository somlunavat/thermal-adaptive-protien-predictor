import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
import math
import numpy as np


class TemperatureEmbedding(nn.Module):
    """Simple temperature conditioning"""

    def __init__(self, d_model, temp_range=(0, 100)):
        super().__init__()
        self.temp_min, self.temp_max = temp_range
        self.temp_proj = nn.Linear(1, d_model)

    def forward(self, temperature):
        # Normalize temperature to [0, 1]
        temp_norm = torch.clamp(
            (temperature - self.temp_min) / (self.temp_max - self.temp_min), 0, 1
        )
        return self.temp_proj(temp_norm.unsqueeze(-1))


class TransformerBlock(nn.Module):
    """Efficient transformer block"""

    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class ProteinMLM(pl.LightningModule):
    def __init__(self, vocab_size=22, d_model=256, nhead=8, num_layers=6,
                 max_length=512, temp_range=(0, 100), lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_length, d_model)
        self.temp_embedding = TemperatureEmbedding(d_model, temp_range)

        # Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size)
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, temperature, attention_mask=None):
        B, L = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(L, device=device).expand(B, -1)

        x = (self.token_embedding(input_ids) +
             self.pos_embedding(positions) +
             self.temp_embedding(temperature).unsqueeze(1))

        # Convert attention mask for transformer
        if attention_mask is not None:
            attention_mask = ~attention_mask  # Invert for pytorch convention

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.norm(x)
        logits = self.mlm_head(x)
        return logits

    def _shared_step(self, batch, stage):
        logits = self(batch['input_ids'], batch['temperature'], batch['attention_mask'])
        loss = self.criterion(logits.view(-1, self.hparams.vocab_size), batch['labels'].view(-1))

        # Accuracy on masked tokens only
        preds = logits.argmax(dim=-1)
        mask = batch['labels'] != -100
        if mask.sum() > 0:
            acc = (preds[mask] == batch['labels'][mask]).float().mean()
        else:
            acc = torch.tensor(0.0, device=self.device)

        self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_acc', acc, prog_bar=True, sync_dist=True)

        if stage == 'train':
            self.log('temp_mean', batch['temperature'].mean(), sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Cosine scheduler with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1000, T_mult=2, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def generate_sequence(self, target_temperature, length=100, top_k=10):
        """Generate sequence conditioned on temperature"""
        self.eval()
        device = next(self.parameters()).device

        # Start with random amino acid
        sequence = [torch.randint(2, self.hparams.vocab_size, (1,)).item()]
        temp_tensor = torch.tensor([target_temperature], device=device, dtype=torch.float)

        with torch.no_grad():
            for _ in range(length - 1):
                input_ids = torch.tensor([sequence], device=device)
                logits = self(input_ids, temp_tensor)

                # Sample from top-k
                next_logits = logits[0, -1, 2:]  # Exclude special tokens
                top_logits, top_indices = torch.topk(next_logits, top_k)
                probs = F.softmax(top_logits / 0.8, dim=-1)  # Temperature scaling

                next_idx = torch.multinomial(probs, 1).item()
                next_token = top_indices[next_idx].item() + 2
                sequence.append(next_token)

        return sequence