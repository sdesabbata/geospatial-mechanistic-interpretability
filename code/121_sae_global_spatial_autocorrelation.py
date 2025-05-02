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

def calculate_spatial_autocorrelations(features_df, features_crs, name):

    # Create geopandas dataframe
    features_df['geometry'] = gpd.points_from_xy(features_df['longitude'], features_df['latitude'], crs="EPSG:4326")
    features_gdf = gpd.GeoDataFrame(features_df, geometry='geometry')
    print(f'Created geopandas dataframe (CRS: {features_gdf.crs})')

    features_gdf = features_gdf.to_crs(f'EPSG:{features_crs}')
    print(f'Projected to CRS: {features_gdf.crs}')


    # Create spatial weights
    features_gdf_w = weights.KNN.from_dataframe(features_gdf, k=8)
    features_gdf_w.transform = 'R'

    # Plot spatial weights
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    features_gdf_w.plot(
        features_gdf,
        ax=ax,
        edge_kws=dict(linewidth=1, color='orangered'),
        node_kws=dict(marker='*')
    )
    ax.set_aspect('equal')
    plt.savefig(f'{results_dir}/{name}_saef_spatial-weights-map.png', dpi=300, bbox_inches='tight')
    plt.close()


    # Get feature columns
    feature_columns = [col for col in features_gdf.columns if ((col.startswith('saef')) & (col not in ['feature class', 'feature code']))]
    print(f'Found {len(feature_columns)} features columns')

    # Calculate global Moran's I for each feature
    moran_dict = {}
    for feat_col in tqdm(feature_columns, desc=name):

        if (features_gdf[feat_col] == 0).all():
            continue

        # Calculate Moran's I
        moran = esda.moran.Moran(
            features_gdf[feat_col], 
            features_gdf_w
            )
        
        # Save results
        moran_dict[feat_col] =  {
            'layer':            name,
            'sae_feat_name':    feat_col,
            'sae_feat_code':    feat_col.replace('saef', ''),
            'sae_feat_idx': int(feat_col.replace('saef', '')),
            'moran_i':          moran.I, 
            'moran_p_sim':      moran.p_sim
            }
        
        del moran


    # Create dataframe
    moran_df = pd.DataFrame.from_dict(moran_dict, orient='index')

    # Save dataframe
    moran_df.to_csv(    f'{results_dir}/{name}_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_spatial-autocorrelations_df.csv', index=False)
    moran_df.to_pickle( f'{results_dir}/{name}_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_spatial-autocorrelations_df.pkl')
    moran_df.to_parquet(f'{results_dir}/{name}_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_spatial-autocorrelations_df.parquet')
    
    return moran_df


# Load data ---------------------------------------------------------------

feat_gb_l15 = pd.read_pickle('storage/GB_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')
feat_it_l15 = pd.read_pickle('storage/IT_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')
feat_ny_l15 = pd.read_pickle('storage/NYmetro_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')

# Calculate spatial autocorrelations --------------------------------------

feat_gb_l15_moran = calculate_spatial_autocorrelations(feat_gb_l15, 27700, 'GB-l15')
feat_it_l15_moran = calculate_spatial_autocorrelations(feat_it_l15, 25832, 'IT-l15')
feat_ny_l15_moran = calculate_spatial_autocorrelations(feat_ny_l15, 32618, 'NYmetro-l15')

# Combine all Moran's I results
all_moran_df = pd.concat([
    feat_gb_l15_moran,
    feat_it_l15_moran,
    feat_ny_l15_moran,
], ignore_index=True)

# Save combined results
all_moran_df.to_csv(    f'{results_dir}/all_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_spatial-autocorrelations_df.csv', index=False)
all_moran_df.to_pickle( f'{results_dir}/all_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_spatial-autocorrelations_df.pkl')
all_moran_df.to_parquet(f'{results_dir}/all_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_spatial-autocorrelations_df.parquet')
