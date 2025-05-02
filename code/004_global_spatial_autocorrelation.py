import os
from tqdm import tqdm
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

from pysal.explore import esda
from pysal.lib import weights


# Create results directory if it doesn't exist
results_dir = 'results/spatial-autocorrelation'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# Spatial autocorrelation -------------------------------------------------

def calculate_spatial_autocorrelations(activations_df, activations_crs, name):

    # Create geopandas dataframe
    activations_df['geometry'] = gpd.points_from_xy(activations_df['longitude'], activations_df['latitude'], crs="EPSG:4326")
    activations_gdf = gpd.GeoDataFrame(activations_df, geometry='geometry')
    print(f'Created geopandas dataframe (CRS: {activations_gdf.crs})')

    activations_gdf = activations_gdf.to_crs(f'EPSG:{activations_crs}')
    print(f'Projected to CRS: {activations_gdf.crs}')


    # Create spatial weights
    activations_gdf_w = weights.KNN.from_dataframe(activations_gdf, k=8)
    activations_gdf_w.transform = 'R'

    # Plot spatial weights
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    activations_gdf_w.plot(
        activations_gdf,
        ax=ax,
        edge_kws=dict(linewidth=1, color='orangered'),
        node_kws=dict(marker='*')
    )
    ax.set_aspect('equal')
    plt.savefig(f'{results_dir}/{name}_spatial-weights-map.png', dpi=300, bbox_inches='tight')
    plt.close()


    # Get activation columns
    activation_columns = [col for col in activations_gdf.columns if col.startswith('act')]
    assert len(activation_columns) == 4096, 'Unexpected numnber of activation columns found'

    # Calculate global Moran's I for each activation
    moran_dict = {}
    for act_col in tqdm(activation_columns, desc=name):

        # Calculate Moran's I
        moran = esda.moran.Moran(
            activations_gdf[act_col], 
            activations_gdf_w
            )
        
        # Save results
        moran_dict[act_col] =  {
            'layer':           name,
            'activation_code':     act_col.replace('act', ''),
            'activation_idx':  int(act_col.replace('act', '')),
            'moran_i':         moran.I, 
            'moran_p_sim':     moran.p_sim
            }
        
        del moran


    # Create dataframe
    moran_df = pd.DataFrame.from_dict(moran_dict, orient='index')
    moran_df = moran_df.reset_index().rename(columns={'index': 'activation_name'})

    # Save dataframe
    moran_df.to_csv(    f'{results_dir}/{name}_activations-mean-pooling_spatial-autocorrelations_df.csv', index=False)
    moran_df.to_pickle( f'{results_dir}/{name}_activations-mean-pooling_spatial-autocorrelations_df.pkl')
    moran_df.to_parquet(f'{results_dir}/{name}_activations-mean-pooling_spatial-autocorrelations_df.parquet')
    
    return moran_df


# Load data ---------------------------------------------------------------

amp_gb_l07 = pd.read_pickle('storage/GB-l07_activations-mean-pooling_df.pkl')
amp_gb_l15 = pd.read_pickle('storage/GB-l15_activations-mean-pooling_df.pkl')
amp_gb_l31 = pd.read_pickle('storage/GB-l31_activations-mean-pooling_df.pkl')

amp_it_l07 = pd.read_pickle('storage/IT-l07_activations-mean-pooling_df.pkl')
amp_it_l15 = pd.read_pickle('storage/IT-l15_activations-mean-pooling_df.pkl')
amp_it_l31 = pd.read_pickle('storage/IT-l31_activations-mean-pooling_df.pkl')

amp_ny_l07 = pd.read_pickle('storage/NYmetro-l07_activations-mean-pooling_df.pkl')
amp_ny_l15 = pd.read_pickle('storage/NYmetro-l15_activations-mean-pooling_df.pkl')
amp_ny_l31 = pd.read_pickle('storage/NYmetro-l31_activations-mean-pooling_df.pkl')


# Calculate spatial autocorrelations --------------------------------------

amp_gb_l07_moran = calculate_spatial_autocorrelations(amp_gb_l07, 27700, 'GB-l07')
amp_gb_l15_moran = calculate_spatial_autocorrelations(amp_gb_l15, 27700, 'GB-l15')
amp_gb_l31_moran = calculate_spatial_autocorrelations(amp_gb_l31, 27700, 'GB-l31')

amp_it_l07_moran = calculate_spatial_autocorrelations(amp_it_l07, 25832, 'IT-l07')
amp_it_l15_moran = calculate_spatial_autocorrelations(amp_it_l15, 25832, 'IT-l15')
amp_it_l31_moran = calculate_spatial_autocorrelations(amp_it_l31, 25832, 'IT-l31')

amp_ny_l07_moran = calculate_spatial_autocorrelations(amp_ny_l07, 32618, 'NYmetro-l07')
amp_ny_l15_moran = calculate_spatial_autocorrelations(amp_ny_l15, 32618, 'NYmetro-l15')
amp_ny_l31_moran = calculate_spatial_autocorrelations(amp_ny_l31, 32618, 'NYmetro-l31')


# Combine all Moran's I results
all_moran_df = pd.concat([
    amp_gb_l07_moran, amp_gb_l15_moran, amp_gb_l31_moran,
    amp_it_l07_moran, amp_it_l15_moran, amp_it_l31_moran,
    amp_ny_l07_moran, amp_ny_l15_moran, amp_ny_l31_moran
], ignore_index=True)

# Save combined results
all_moran_df.to_csv(    f'{results_dir}/all_activations-mean-pooling_spatial-autocorrelations_df.csv', index=False)
all_moran_df.to_pickle( f'{results_dir}/all_activations-mean-pooling_spatial-autocorrelations_df.pkl')
all_moran_df.to_parquet(f'{results_dir}/all_activations-mean-pooling_spatial-autocorrelations_df.parquet')
