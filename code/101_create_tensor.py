import pandas as pd
import torch

# Function to create tensors
def create_tensor(df_gb, df_it, df_ny, layer):

    # Filter layer
    results_layer_gb = df_gb[(df_gb['layer']==layer)]
    results_layer_it = df_it[(df_it['layer']==layer)]
    results_layer_ny = df_ny[(df_ny['layer']==layer)]

    print(f'\n\nGB layer {layer:02d} info')
    print(results_layer_gb.info())
    print(f'\n\nIT layer {layer:02d} info')
    print(results_layer_it.info())
    print(f'\n\nNYmetro layer {layer:02d} info')
    print(results_layer_ny.info())

    # Collect info
    info_df_gb         = results_layer_gb[['geonameid', 'prompt', 'layer']].copy()
    info_df_gb         = info_df_gb.reset_index(drop=True)
    info_df_gb['area'] = 'GB'
    info_df_it         = results_layer_it[['geonameid', 'prompt', 'layer']].copy()
    info_df_it         = info_df_it.reset_index(drop=True)
    info_df_it['area'] = 'IT'
    info_df_ny         = results_layer_ny[['geonameid', 'prompt', 'layer']].copy()
    info_df_ny         = info_df_ny.reset_index(drop=True)
    info_df_ny['area'] = 'NYmetro'
    info_df            = pd.concat([info_df_gb, info_df_it, info_df_ny], ignore_index=True)

    print('\n\nInfo dataframe')
    print(info_df.info())
    print(info_df.head())

    # Create activations tensor
    activations_gb = torch.stack([torch.tensor(row[0]) for row in results_layer_gb['mean_pooling']])
    activations_it = torch.stack([torch.tensor(row[0]) for row in results_layer_it['mean_pooling']])
    activations_ny = torch.stack([torch.tensor(row[0]) for row in results_layer_ny['mean_pooling']])
    activations    = torch.cat((activations_gb, activations_it, activations_ny), dim=0)

    print(f'\n\nAll activations layer {layer:02d}')
    print(activations.shape)
    print('\n')
    print(activations)

    # Checks
    assert activations.shape[0] == (len(results_layer_gb) + len(results_layer_it) + len(results_layer_ny)), 'The number of cases in the activations tensor is not correct'
    assert activations.shape[0] == (len(info_df)),                                                          'The number of cases in the activations tensor is not correct'
    assert activations.shape[1] == 4096,                                                                    'The number of columns in the activations tensor is not correct'

    # Save files
    torch.save(activations, f'storage/activations_mean-pooling_layer{layer:02d}.pt')
    info_df.to_pickle(      f'storage/activations_mean-pooling_layer{layer:02d}_info.pkl')
    info_df.to_csv(         f'storage/activations_mean-pooling_layer{layer:02d}.csv', index=False)
    info_df.to_parquet(     f'storage/activations_mean-pooling_layer{layer:02d}_info.parquet', index=False)

    print('\n\nTesors and info correctly saved.')


# Load activations dataframes
results_df_gb = pd.read_pickle('storage/activations_GB.pkl')
results_df_it = pd.read_pickle('storage/activations_IT.pkl')
results_df_ny = pd.read_pickle('storage/activations_NYmetro.pkl')

# Create tensors
create_tensor(results_df_gb, results_df_it, results_df_ny,  7)
create_tensor(results_df_gb, results_df_it, results_df_ny, 15)
create_tensor(results_df_gb, results_df_it, results_df_ny, 31)
