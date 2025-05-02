import math
import pandas as pd
import numpy as np

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger


# ----------------------------------------------------------------------
# Autoencoder
# ----------------------------------------------------------------------

# TopK activation function
# by Gao et al (2024)
# https://arxiv.org/abs/2406.04093
#
# Based on
# https://github.com/openai/sparse_autoencoder
# MIT license

class TopK(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        print(f'TopK: set k={self.k}.')
        self.postact_fn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error (shape: [1])
    """
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1) / (original_input**2).mean(dim=1)
    ).mean()

def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized L1 loss (shape: [1])
    """
    return (latent_activations.abs().sum(dim=1) / original_input.norm(dim=1)).mean()


# Autoencoder as LightningModule
class Autoencoder(pl.LightningModule):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(4096, 32768),
            TopK(2048)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32768, 4096)
        )
        # Sparsity
        self.sparsity_loss_weight = 0.01
    
    def forward(self, input):
        embeddings = self.encoder(input)
        reconstruction = self.decoder(embeddings)
        return embeddings, reconstruction

    def training_step(self, batch, batch_idx):
        embeddings, reconstruction = self.forward(batch)

        recon_loss = normalized_mean_squared_error(reconstruction, batch)
        sparsity_loss = normalized_L1_loss(embeddings, batch)
        loss = recon_loss + self.sparsity_loss_weight * sparsity_loss
        self.log('recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('sparsity_loss', sparsity_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss_epoch'}


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------

activations = torch.load('storage/activations_mean-pooling_layer15.pt', weights_only=True)
print("\nActivations shape:", activations.shape)

data_tensor_nrow = activations.shape[0]
data_tensor_ncol = activations.shape[1]

data_tensor_loader = torch.utils.data.DataLoader(
    activations, 
    batch_size=32, 
    shuffle=True
    )

logger_test_name = "111-sae_l15-GB-IT-NY_to32768top2048_300epochs"
logger_folder = "../storage/lightning_logs"
logger_tb = TensorBoardLogger(logger_folder, name=logger_test_name)
logger_csv = CSVLogger(logger_folder, name=logger_test_name)

autoencoder = Autoencoder()
# print(autoencoder)

trainer = Trainer(
    devices=1, 
    accelerator='gpu',
    logger=[logger_tb, logger_csv], 
    enable_progress_bar=False,
    max_epochs=300
    )

trainer.fit(
    model=autoencoder, 
    train_dataloaders=data_tensor_loader
    )



# ----------------------------------------------------------------------
# Save model
# ----------------------------------------------------------------------

torch.save(autoencoder, '../storage/111-sae_l15-GB-IT-NY_to32768top2048_300epochs.pth')
