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


def create_geodataframe(activations_df, activations_crs):
    # Create geopandas dataframe
    activations_df['geometry'] = gpd.points_from_xy(activations_df['longitude'], activations_df['latitude'], crs="EPSG:4326")
    activations_gdf = gpd.GeoDataFrame(activations_df, geometry='geometry')
    
    # Project to target CRS
    activations_gdf = activations_gdf.to_crs(f'EPSG:{activations_crs}')
    
    return activations_gdf


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
    # Create geodataframe from activations dataframe
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
    activations_w = weights.W.from_adjlist(weights_adjlist)
    activations_w.transform = 'R'

    # Get activation names to plot
    all_moran_df_layer = all_moran_df[all_moran_df['layer_num'] == layer_num]
    all_moran_df_layer = all_moran_df_layer[(all_moran_df_layer['moran_i'] >= 0.3) & (all_moran_df_layer['moran_p_sim'] < 0.01)]
    activation_names   = all_moran_df_layer['activation_name'].unique()

    # Create directories
    if not os.path.exists(f'results/spatial-autocorrelation/lisa-maps_l{layer_num}'):
        os.makedirs(      f'results/spatial-autocorrelation/lisa-maps_l{layer_num}')
    
    for activation_name in tqdm(activation_names, f'Layer {layer_num}'):

        # Get activations
        gb_act_gdf = gb_gdf[['geonameid', activation_name, gb_gdf.geometry.name]].copy()
        it_act_gdf = it_gdf[['geonameid', activation_name, it_gdf.geometry.name]].copy()
        ny_act_gdf = ny_gdf[['geonameid', activation_name, ny_gdf.geometry.name]].copy()

        # Combine activations
        activations = pd.concat([
            gb_act_gdf[['geonameid', activation_name]],
            it_act_gdf[['geonameid', activation_name]],
            ny_act_gdf[['geonameid', activation_name]]
        ], ignore_index=True)
        activations[activation_name] = activations[activation_name].astype(np.float64)

        # Calculate LISA        
        lisa = esda.moran.Moran_Local(
            activations[activation_name], 
            activations_w
        )

        # Assign LISA results back to respective geodataframes
        gb_act_gdf['lisa_q']       = lisa.q[    0:gb_act_gdf.shape[0]]
        gb_act_gdf['lisa_p_sim']   = lisa.p_sim[0:gb_act_gdf.shape[0]]
        gb_act_gdf['lisa_cluster'] = gb_act_gdf.apply(lambda x: lisa_cluster_type(x['lisa_q'], x['lisa_p_sim'], p=0.01), axis=1)
        gb_act_gdf['lisa_colour']  = gb_act_gdf['lisa_cluster'].apply(lambda x: lisa_colors[x])

        it_act_gdf['lisa_q']       = lisa.q[    gb_act_gdf.shape[0]:gb_act_gdf.shape[0]+it_act_gdf.shape[0]]
        it_act_gdf['lisa_p_sim']   = lisa.p_sim[gb_act_gdf.shape[0]:gb_act_gdf.shape[0]+it_act_gdf.shape[0]]
        it_act_gdf['lisa_cluster'] = it_act_gdf.apply(lambda x: lisa_cluster_type(x['lisa_q'], x['lisa_p_sim'], p=0.01), axis=1)
        it_act_gdf['lisa_colour']  = it_act_gdf['lisa_cluster'].apply(lambda x: lisa_colors[x])

        ny_act_gdf['lisa_q']       = lisa.q[    gb_act_gdf.shape[0]+it_act_gdf.shape[0]:]
        ny_act_gdf['lisa_p_sim']   = lisa.p_sim[gb_act_gdf.shape[0]+it_act_gdf.shape[0]:]
        ny_act_gdf['lisa_cluster'] = ny_act_gdf.apply(lambda x: lisa_cluster_type(x['lisa_q'], x['lisa_p_sim'], p=0.01), axis=1)
        ny_act_gdf['lisa_colour']  = ny_act_gdf['lisa_cluster'].apply(lambda x: lisa_colors[x])

        # Plot LISA maps
        activations_code = activation_name.replace('act', '')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            # GB --------------------------------------------------------------

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gb_act_gdf.plot(
                column="lisa_cluster",
                categorical=True,
                color=gb_act_gdf["lisa_colour"],
                legend=False,
                ax=ax,
                markersize=1
            )
            ax.set_axis_off()
            ax.set_aspect("equal")
            ax.set_title(f'GB {layer_num} {activations_code} Local Moran\'s I clusters', fontsize=15)
            plt.savefig(f'results/spatial-autocorrelation/lisa-maps_l{layer_num}/lisa-map_{activation_name}_l{layer_num}_GB.png', dpi=300, bbox_inches='tight')
            plt.close()

            del fig, ax

            # IT --------------------------------------------------------------

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            it_act_gdf.plot(
                column="lisa_cluster",
                categorical=True,
                color=it_act_gdf["lisa_colour"],
                legend=False,
                ax=ax,
                markersize=1
            )
            ax.set_axis_off()
            ax.set_aspect("equal")
            ax.set_title(f'IT {layer_num} {activations_code} Local Moran\'s I clusters', fontsize=15)
            plt.savefig(f'results/spatial-autocorrelation/lisa-maps_l{layer_num}/lisa-map_{activation_name}_l{layer_num}_IT.png', dpi=300, bbox_inches='tight')
            plt.close()

            del fig, ax

            # NYm -------------------------------------------------------------

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ny_act_gdf.plot(
                column="lisa_cluster",
                categorical=True,
                color=ny_act_gdf["lisa_colour"],
                legend=False,
                ax=ax,
                markersize=1
            )
            ax.set_axis_off()
            ax.set_aspect("equal")
            ax.set_title(f'NYm {layer_num} {activations_code} Local Moran\'s I clusters', fontsize=15)
            plt.savefig(f'results/spatial-autocorrelation/lisa-maps_l{layer_num}/lisa-map_{activation_name}_l{layer_num}_NYmetro.png', dpi=300, bbox_inches='tight')
            plt.close()

            del fig, ax
            del activations, activations_code, gb_act_gdf, it_act_gdf, ny_act_gdf, lisa
    del activations_w



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

all_moran_df = pd.read_pickle('results/spatial-autocorrelation/all_activations-mean-pooling_spatial-autocorrelations_df.pkl')
all_moran_df['layer_area'] = all_moran_df['layer'].apply(lambda x: x.split('-')[0])
all_moran_df['layer_num']  = all_moran_df['layer'].apply(lambda x: x.split('-')[1].replace('l', ''))

print(all_moran_df.groupby(['layer', 'layer_area', 'layer_num']).size().reset_index(name='count'))


# LISA maps ----------------------------------------------------------------


user_input = input(f'Do you want to continue processing layer 07? (y/N): ')
if user_input.lower() != 'y':
    print(          'Exiting...')
    sys.exit(1)
plot_lisa_maps(amp_gb_l07, 27700, amp_it_l07, 25832, amp_ny_l07, 32618, '07', all_moran_df)


user_input = input(f'Do you want to continue processing layer 15? (y/N): ')
if user_input.lower() != 'y':
    print(          'Exiting...')
    sys.exit(1)
plot_lisa_maps(amp_gb_l15, 27700, amp_it_l15, 25832, amp_ny_l15, 32618, '15', all_moran_df)


user_input = input(f'Do you want to continue processing layer 31? (y/N): ')
if user_input.lower() != 'y':
    print(          'Exiting...')
    sys.exit(1)
plot_lisa_maps(amp_gb_l31, 27700, amp_it_l31, 25832, amp_ny_l31, 32618, '31', all_moran_df)