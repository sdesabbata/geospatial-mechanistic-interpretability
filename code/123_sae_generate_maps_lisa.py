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
import warnings


def create_geodataframe(features_df, features_crs):
    # Create geopandas dataframe
    features_df['geometry'] = gpd.points_from_xy(features_df['longitude'], features_df['latitude'], crs="EPSG:4326")
    features_gdf = gpd.GeoDataFrame(features_df, geometry='geometry')
    
    # Project to target CRS
    features_gdf = features_gdf.to_crs(f'EPSG:{features_crs}')
    
    return features_gdf


# LISA cluster types and colours
# https://github.com/pysal/splot/blob/4f770241299792e495b6542d28c4c324ec1709d5/splot/_viz_utils.py#L23
# def moran_hot_cold_spots(q, p_sim, p=0.05):
#     sig = 1 * (p_sim < p)
#     HH = 1 * (sig * q == 1)
#     LL = 3 * (sig * q == 3)
#     LH = 2 * (sig * q == 2)
#     HL = 4 * (sig * q == 4)
#     cluster = HH + LL + LH + HL
#     return cluster
def lisa_cluster_type(q, p_sim, p=0.05):
    if p_sim >= p:
        return 'ns'
    else:
        if q == 1:
            return 'HH'
        elif q == 3:
            return 'LL'
        elif q == 2:
            return 'LH'
        elif q == 4:
            return 'HL'
# LISA cluster colours
lisa_colors = {
    "HH": "#d7191c",
    "HL": "#fdae61",
    "LH": "#abd9e9",
    "LL": "#2c7bb6",
    "ns": "#cccccc",
}


def plot_lisa_maps(gb_df, gb_crs, it_df, it_crs, ny_df, ny_crs, layer_num, all_moran_df):
    # Create geodataframe from features dataframe
    gb_gdf = create_geodataframe(gb_df, gb_crs)
    it_gdf = create_geodataframe(it_df, it_crs)
    ny_gdf = create_geodataframe(ny_df, ny_crs)

    # Create weights for combined LISA
    gb_weights = weights.KNN.from_dataframe(gb_gdf, k=8)
    it_weights = weights.KNN.from_dataframe(it_gdf, k=8)
    ny_weights = weights.KNN.from_dataframe(ny_gdf, k=8)

    # Transform weights to adjlist
    gb_adjlist = gb_weights.to_adjlist()
    it_adjlist = it_weights.to_adjlist()
    ny_adjlist = ny_weights.to_adjlist()

    # Adjust indices
    # shift indices for IT based on GB size
    # shift indices for NY based on GB and IT sizes
    it_adjlist['focal']    = it_adjlist['focal'].apply(   lambda x: x + gb_gdf.shape[0])
    it_adjlist['neighbor'] = it_adjlist['neighbor'].apply(lambda x: x + gb_gdf.shape[0])
    ny_adjlist['focal']    = ny_adjlist['focal'].apply(   lambda x: x + gb_gdf.shape[0] + it_gdf.shape[0])
    ny_adjlist['neighbor'] = ny_adjlist['neighbor'].apply(lambda x: x + gb_gdf.shape[0] + it_gdf.shape[0])

    # Combine weights
    weights_adjlist = pd.concat([
        gb_adjlist,
        it_adjlist,
        ny_adjlist
    ], ignore_index=True)
    features_w = weights.W.from_adjlist(weights_adjlist)
    features_w.transform = 'R'

    # Get activation names to plot
    all_moran_df_layer = all_moran_df[all_moran_df['layer_num'] == layer_num]
    all_moran_df_layer = all_moran_df_layer[(all_moran_df_layer['moran_i'] >= 0.3) & (all_moran_df_layer['moran_p_sim'] < 0.01)]
    feature_names   = all_moran_df_layer['sae_feat_name'].unique()

    # Create directories
    if not os.path.exists(f'results/spatial-autocorrelation/sae_feature_lisa-maps_l{layer_num}'):
        os.makedirs(      f'results/spatial-autocorrelation/sae_feature_lisa-maps_l{layer_num}')
    
    for feature_name in tqdm(feature_names, f'Layer {layer_num}'):

        # Get features
        gb_saef_gdf = gb_gdf[['geonameid', feature_name, gb_gdf.geometry.name]].copy()
        it_saef_gdf = it_gdf[['geonameid', feature_name, it_gdf.geometry.name]].copy()
        ny_saef_gdf = ny_gdf[['geonameid', feature_name, ny_gdf.geometry.name]].copy()

        # Combine features
        features = pd.concat([
            gb_saef_gdf[['geonameid', feature_name]],
            it_saef_gdf[['geonameid', feature_name]],
            ny_saef_gdf[['geonameid', feature_name]]
        ], ignore_index=True)
        features[feature_name] = features[feature_name].astype(np.float64)

        # Calculate LISA        
        lisa = esda.moran.Moran_Local(
            features[feature_name], 
            features_w
        )

        # Assign LISA results back to respective geodataframes
        gb_saef_gdf['lisa_q']       = lisa.q[    0:gb_saef_gdf.shape[0]]
        gb_saef_gdf['lisa_p_sim']   = lisa.p_sim[0:gb_saef_gdf.shape[0]]
        gb_saef_gdf['lisa_cluster'] = gb_saef_gdf.apply(lambda x: lisa_cluster_type(x['lisa_q'], x['lisa_p_sim'], p=0.01), axis=1)
        gb_saef_gdf['lisa_colour']  = gb_saef_gdf['lisa_cluster'].apply(lambda x: lisa_colors[x])

        it_saef_gdf['lisa_q']       = lisa.q[    gb_saef_gdf.shape[0]:gb_saef_gdf.shape[0]+it_saef_gdf.shape[0]]
        it_saef_gdf['lisa_p_sim']   = lisa.p_sim[gb_saef_gdf.shape[0]:gb_saef_gdf.shape[0]+it_saef_gdf.shape[0]]
        it_saef_gdf['lisa_cluster'] = it_saef_gdf.apply(lambda x: lisa_cluster_type(x['lisa_q'], x['lisa_p_sim'], p=0.01), axis=1)
        it_saef_gdf['lisa_colour']  = it_saef_gdf['lisa_cluster'].apply(lambda x: lisa_colors[x])

        ny_saef_gdf['lisa_q']       = lisa.q[    gb_saef_gdf.shape[0]+it_saef_gdf.shape[0]:]
        ny_saef_gdf['lisa_p_sim']   = lisa.p_sim[gb_saef_gdf.shape[0]+it_saef_gdf.shape[0]:]
        ny_saef_gdf['lisa_cluster'] = ny_saef_gdf.apply(lambda x: lisa_cluster_type(x['lisa_q'], x['lisa_p_sim'], p=0.01), axis=1)
        ny_saef_gdf['lisa_colour']  = ny_saef_gdf['lisa_cluster'].apply(lambda x: lisa_colors[x])

        # Plot LISA maps
        features_code = feature_name.replace('saef', '')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            # GB --------------------------------------------------------------

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gb_saef_gdf.plot(
                column="lisa_cluster",
                categorical=True,
                color=gb_saef_gdf["lisa_colour"],
                legend=False,
                ax=ax,
                markersize=1
            )
            ax.set_axis_off()
            ax.set_aspect("equal")
            ax.set_title(f'GB {layer_num} feature {features_code} Local Moran\'s I clusters', fontsize=15)
            plt.savefig(f'results/spatial-autocorrelation/sae_feature_lisa-maps_l{layer_num}/sae_feature_lisa-map_{feature_name}_l{layer_num}_GB.png', dpi=300, bbox_inches='tight')
            plt.close()

            del fig, ax

            # IT --------------------------------------------------------------

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            it_saef_gdf.plot(
                column="lisa_cluster",
                categorical=True,
                color=it_saef_gdf["lisa_colour"],
                legend=False,
                ax=ax,
                markersize=1
            )
            ax.set_axis_off()
            ax.set_aspect("equal")
            ax.set_title(f'IT {layer_num} feature {features_code} Local Moran\'s I clusters', fontsize=15)
            plt.savefig(f'results/spatial-autocorrelation/sae_feature_lisa-maps_l{layer_num}/sae_feature_lisa-map_{feature_name}_l{layer_num}_IT.png', dpi=300, bbox_inches='tight')
            plt.close()

            del fig, ax

            # NYm -------------------------------------------------------------

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ny_saef_gdf.plot(
                column="lisa_cluster",
                categorical=True,
                color=ny_saef_gdf["lisa_colour"],
                legend=False,
                ax=ax,
                markersize=1
            )
            ax.set_axis_off()
            ax.set_aspect("equal")
            ax.set_title(f'NYm {layer_num} feature {features_code} Local Moran\'s I clusters', fontsize=15)
            plt.savefig(f'results/spatial-autocorrelation/sae_feature_lisa-maps_l{layer_num}/sae_feature_lisa-map_{feature_name}_l{layer_num}_NYmetro.png', dpi=300, bbox_inches='tight')
            plt.close()

            del fig, ax
            del features, features_code, gb_saef_gdf, it_saef_gdf, ny_saef_gdf, lisa
    del features_w



# Load data ---------------------------------------------------------------

feat_gb_l15 = pd.read_pickle('storage/GB_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')
feat_it_l15 = pd.read_pickle('storage/IT_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')
feat_ny_l15 = pd.read_pickle('storage/NYmetro_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_nonzero.pkl')

all_moran_df = pd.read_pickle('results/spatial-autocorrelation/all_features_sae_l15-GB-IT-NY_to32768top2048_300epochs_spatial-autocorrelations_df.pkl')
all_moran_df['layer_area'] = all_moran_df['layer'].apply(lambda x: x.split('-')[0])
all_moran_df['layer_num']  = all_moran_df['layer'].apply(lambda x: x.split('-')[1].replace('l', ''))

print(all_moran_df.groupby(['layer', 'layer_area', 'layer_num']).size().reset_index(name='count'))


# LISA maps ----------------------------------------------------------------


user_input = input(f'Do you want to continue processing layer 15? (y/N): ')
if user_input.lower() != 'y':
    print(          'Exiting...')
    sys.exit(1)
plot_lisa_maps(feat_gb_l15, 27700, feat_it_l15, 25832, feat_ny_l15, 32618, '15', all_moran_df)
