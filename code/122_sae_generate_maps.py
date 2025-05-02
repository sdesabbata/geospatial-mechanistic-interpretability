import pandas as pd
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap


import geopandas as gpd
from shapely.geometry import Point

from pysal.explore import esda
from pysal.lib import weights
import mapclassify
from splot.esda import plot_moran

from tqdm import tqdm


def create_geodataframe(features_df, features_crs):
    # Create geopandas dataframe
    features_df['geometry'] = gpd.points_from_xy(features_df['longitude'], features_df['latitude'], crs="EPSG:4326")
    features_gdf = gpd.GeoDataFrame(features_df, geometry='geometry')
    
    # Project to target CRS
    features_gdf = features_gdf.to_crs(f'EPSG:{features_crs}')
    
    return features_gdf


def plot_features_map(features_df, features_crs, layer, feature_name, gsa_df, cmap, norm, breaks):
    # Create geodataframe from activations dataframe
    features_gdf = create_geodataframe(features_df, features_crs)

    if gsa_df.loc[
        (gsa_df['layer']           == layer) & 
        (gsa_df['sae_feat_name'] == feature_name ), 
        'moran_i'
    ].empty:
        moran_i_value = ''
        moran_p_value = ''
    else:
        moran_i_value = gsa_df.loc[
            (gsa_df['layer']           == layer) & 
            (gsa_df['sae_feat_name'] == feature_name ), 
            'moran_i'
        ].values[0]
        moran_i_value = f'{moran_i_value:.2f}'

        moran_p_value = gsa_df.loc[
            (gsa_df['layer']           == layer) & 
            (gsa_df['sae_feat_name'] == feature_name ), 
            'moran_p_sim'
        ].values[0]
        moran_p_value = f'{moran_p_value}'


    features_code = feature_name.replace('saef', '')
    layer_title   = layer.replace('-l', ' ')
    layer_area    = layer.split('-')[0]
    layer_num     = layer.split('-')[1].replace('l', '')

    map_title     = f'{layer_title} {features_code} i={moran_i_value} p={moran_p_value}'
    map_filename  = f'results/spatial-autocorrelation/sae_feature_maps_l{layer_num}/sae_feature_map_{feature_name}_l{layer_num}_{layer_area}.png'
    
    if not os.path.exists(f'results/spatial-autocorrelation/sae_feature_maps_l{layer_num}'):
        os.makedirs(      f'results/spatial-autocorrelation/sae_feature_maps_l{layer_num}')


    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    features_gdf.plot(
        ax=ax, 
        column=feature_name, 
        cmap=cmap, 
        norm=norm, 
        markersize=1
        )
    ax.set_axis_off()
    ax.set_title(map_title, fontsize=15)
    cbar = fig.colorbar(sm, ax=ax, boundaries=breaks, ticks=breaks, orientation='horizontal', spacing='proportional', drawedges=True, pad=0.1, aspect=40, shrink=0.75)
    cbar.ax.tick_params(labelsize=10)
    plt.savefig(map_filename, dpi=300, bbox_inches='tight')
    plt.close()



# Load data ---------------------------------------------------------------

feat_gb_l15 = pd.read_pickle('storage/GB_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')
feat_it_l15 = pd.read_pickle('storage/IT_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')
feat_ny_l15 = pd.read_pickle('storage/NYmetro_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')

all_moran_df = pd.read_pickle('results/spatial-autocorrelation/all_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_spatial-autocorrelations_df.pkl')
all_moran_df['layer_area'] = all_moran_df['layer'].apply(lambda x: x.split('-')[0])
all_moran_df['layer_num']  = all_moran_df['layer'].apply(lambda x: x.split('-')[1].replace('l', ''))

print(all_moran_df.groupby(['layer', 'layer_area', 'layer_num']).size().reset_index(name='count'))

feature_names_07 = all_moran_df[all_moran_df['layer_num'] == '07']['sae_feat_name'].unique()
feature_names_15 = all_moran_df[all_moran_df['layer_num'] == '15']['sae_feat_name'].unique()
feature_names_31 = all_moran_df[all_moran_df['layer_num'] == '31']['sae_feat_name'].unique()

# Maps --------------------------------------------------------------------

print('\nCreate maps:')
cmap = plt.get_cmap('viridis')

for feature_name in tqdm(feature_names_15, 'Layer 15'):

    this_feat_gb_l15 = feat_gb_l15[feature_name].values.tolist()
    this_feat_it_l15 = feat_it_l15[feature_name].values.tolist()
    this_feat_ny_l15 = feat_ny_l15[feature_name].values.tolist()

    combined_values = np.array(this_feat_gb_l15 + this_feat_it_l15 + this_feat_ny_l15)

    jenks = mapclassify.NaturalBreaks(combined_values, k=9)
    breaks = jenks.bins
    # print("Jenks Natural Breaks:", breaks)
    norm = BoundaryNorm(boundaries=breaks, ncolors=cmap.N)

    plot_features_map(feat_gb_l15, 27700,      'GB-l15', feature_name, all_moran_df, cmap, norm, breaks)
    plot_features_map(feat_it_l15, 25832,      'IT-l15', feature_name, all_moran_df, cmap, norm, breaks)
    plot_features_map(feat_ny_l15, 32618, 'NYmetro-l15', feature_name, all_moran_df, cmap, norm, breaks)

    del this_feat_gb_l15, this_feat_it_l15, this_feat_ny_l15, combined_values, jenks, breaks, norm
