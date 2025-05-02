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
# Load data and model
# ----------------------------------------------------------------------

activations = torch.load('storage/activations_mean-pooling_layer15.pt', weights_only=True)
print("\nActivations shape:", activations.shape)

print('\nLoading sparse autoencoder...')
autoencoder_l15 = torch.load('storage/111-sae_l15-GB-IT-NY_to32768top2048_300epochs.pth')
print('done.')


# ----------------------------------------------------------------------
# Generate features
# ----------------------------------------------------------------------

print('\nGenerating features...')
encoded  = autoencoder_l15.encoder(activations)
features = encoded.detach().cpu()
print('done.')


# ----------------------------------------------------------------------
# Save features
# ----------------------------------------------------------------------

print('\nSaving features...')


# Extract features to a dictionary
features_dict = {}
features_dict_nonzero = {}

for feat in range(features.shape[1]):
    col_name = f'saef{feat:06d}'
    col_vals = features[:, feat].numpy()

    features_dict[col_name] = col_vals
    if features[:, feat].nonzero().shape[0] > 0:
        features_dict_nonzero[col_name] = col_vals
    
    del col_name, col_vals


# Combine features_dict with info_df
features_df = pd.concat([
    pd.read_pickle('storage/activations_mean-pooling_layer15_info.pkl'), 
    pd.DataFrame(features_dict)
    ], axis=1)

features_df_nonzero = pd.concat([
    pd.read_pickle('storage/activations_mean-pooling_layer15_info.pkl'), 
    pd.DataFrame(features_dict_nonzero)
    ], axis=1)


# Save features tensor
torch.save(features,           'storage/features_sae_l15-GB-IT-NY_to32768top2048_300epochs.pt')
# Save features dataframe with info
features_df.to_pickle(         'storage/features_sae_l15-GB-IT-NY_to32768top2048_300epochs.pkl')
features_df.to_csv(            'storage/features_sae_l15-GB-IT-NY_to32768top2048_300epochs.csv', index=False)
features_df.to_parquet(        'storage/features_sae_l15-GB-IT-NY_to32768top2048_300epochs.parquet', index=False)
# Save features dataframe with info (nonzero)
features_df_nonzero.to_pickle( 'storage/features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')
features_df_nonzero.to_csv(    'storage/features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.csv', index=False)
features_df_nonzero.to_parquet('storage/features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.parquet', index=False)


# Merge with GeoNames data
gb_places_df = pd.read_csv('storage/geonames_GB.csv')
it_places_df = pd.read_csv('storage/geonames_IT.csv')
ny_places_df = pd.read_csv('storage/geonames_NYmetro.csv')

features_gb = features_df_nonzero[features_df_nonzero['area'] == 'GB']
features_it = features_df_nonzero[features_df_nonzero['area'] == 'IT']
features_ny = features_df_nonzero[features_df_nonzero['area'] == 'NYmetro']

features_gb = features_gb.merge(gb_places_df, on='geonameid', how='inner')
features_it = features_it.merge(it_places_df, on='geonameid', how='inner')
features_ny = features_ny.merge(ny_places_df, on='geonameid', how='inner')

features_gb.to_pickle( 'storage/GB_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')
features_gb.to_csv(    'storage/GB_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.csv', index=False)
features_gb.to_parquet('storage/GB_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.parquet', index=False)

features_it.to_pickle( 'storage/IT_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')
features_it.to_csv(    'storage/IT_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.csv', index=False)
features_it.to_parquet('storage/IT_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.parquet', index=False)

features_ny.to_pickle( 'storage/NYmetro_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')
features_ny.to_csv(    'storage/NYmetro_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.csv', index=False)
features_ny.to_parquet('storage/NYmetro_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.parquet', index=False)


print('done.')
