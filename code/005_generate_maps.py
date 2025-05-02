import pandas as pd
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm

import geopandas as gpd
import mapclassify

from tqdm import tqdm


def create_geodataframe(activations_df, activations_crs):
    # Create geopandas dataframe
    activations_df['geometry'] = gpd.points_from_xy(activations_df['longitude'], activations_df['latitude'], crs="EPSG:4326")
    activations_gdf = gpd.GeoDataFrame(activations_df, geometry='geometry')
    
    # Project to target CRS
    activations_gdf = activations_gdf.to_crs(f'EPSG:{activations_crs}')
    
    return activations_gdf


def plot_activations_map(activations_df, activations_crs, layer, activation_name, gsa_df, cmap, norm, breaks):
    # Create geodataframe from activations dataframe
    activations_gdf = create_geodataframe(activations_df, activations_crs)


    moran_i_value = gsa_df.loc[
        (gsa_df['layer']           == layer) & 
        (gsa_df['activation_name'] == activation_name ), 
        'moran_i'
    ].values[0]

    moran_p_value = gsa_df.loc[
        (gsa_df['layer']           == layer) & 
        (gsa_df['activation_name'] == activation_name ), 
        'moran_p_sim'
    ].values[0]


    activations_code = activation_name.replace('act', '')
    layer_title      = layer.replace('-l', ' ')
    layer_area       = layer.split('-')[0]
    layer_num        = layer.split('-')[1].replace('l', '')

    map_title        = f'{layer_title} {activations_code} i={moran_i_value:.2f} p={moran_p_value}'
    map_filename     =    f'results/spatial-autocorrelation/maps_l{layer_num}/map_{activation_name}_l{layer_num}_{layer_area}.png'

    if not os.path.exists(f'results/spatial-autocorrelation/maps_l{layer_num}'):
        os.makedirs(      f'results/spatial-autocorrelation/maps_l{layer_num}')


    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    activations_gdf.plot(
        ax=ax, 
        column=activation_name, 
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

activation_names_07 = all_moran_df[all_moran_df['layer_num'] == '07']['activation_name'].unique()
activation_names_15 = all_moran_df[all_moran_df['layer_num'] == '15']['activation_name'].unique()
activation_names_31 = all_moran_df[all_moran_df['layer_num'] == '31']['activation_name'].unique()

# Maps --------------------------------------------------------------------

print('\nCreate maps:')
cmap = plt.get_cmap('inferno')


user_input = input(f'Do you want to continue processing layer 07? (y/N): ')
if user_input.lower() != 'y':
    print(          'Exiting...')
    sys.exit(1)

for activation_name in tqdm(activation_names_07, 'Layer 07'):

    this_amp_gb_l07 = amp_gb_l07[activation_name].values.tolist()
    this_amp_it_l07 = amp_it_l07[activation_name].values.tolist()
    this_amp_ny_l07 = amp_ny_l07[activation_name].values.tolist()

    combined_values = np.array(this_amp_gb_l07 + this_amp_it_l07 + this_amp_ny_l07)

    jenks = mapclassify.NaturalBreaks(combined_values, k=9)
    breaks = jenks.bins
    # print("Jenks Natural Breaks:", breaks)
    norm = BoundaryNorm(boundaries=breaks, ncolors=cmap.N)

    plot_activations_map(amp_gb_l07, 27700,      'GB-l07', activation_name, all_moran_df, cmap, norm, breaks)
    plot_activations_map(amp_it_l07, 25832,      'IT-l07', activation_name, all_moran_df, cmap, norm, breaks)
    plot_activations_map(amp_ny_l07, 32618, 'NYmetro-l07', activation_name, all_moran_df, cmap, norm, breaks)

    del this_amp_gb_l07, this_amp_it_l07, this_amp_ny_l07, combined_values, jenks, breaks, norm


user_input = input(f'Do you want to continue processing layer 15? (y/N): ')
if user_input.lower() != 'y':
    print(          'Exiting...')
    sys.exit(1)

for activation_name in tqdm(activation_names_15, 'Layer 15'):

    this_amp_gb_l15 = amp_gb_l15[activation_name].values.tolist()
    this_amp_it_l15 = amp_it_l15[activation_name].values.tolist()
    this_amp_ny_l15 = amp_ny_l15[activation_name].values.tolist()

    combined_values = np.array(this_amp_gb_l15 + this_amp_it_l15 + this_amp_ny_l15)

    jenks = mapclassify.NaturalBreaks(combined_values, k=9)
    breaks = jenks.bins
    # print("Jenks Natural Breaks:", breaks)
    norm = BoundaryNorm(boundaries=breaks, ncolors=cmap.N)

    plot_activations_map(amp_gb_l15, 27700,      'GB-l15', activation_name, all_moran_df, cmap, norm, breaks)
    plot_activations_map(amp_it_l15, 25832,      'IT-l15', activation_name, all_moran_df, cmap, norm, breaks)
    plot_activations_map(amp_ny_l15, 32618, 'NYmetro-l15', activation_name, all_moran_df, cmap, norm, breaks)

    del this_amp_gb_l15, this_amp_it_l15, this_amp_ny_l15, combined_values, jenks, breaks, norm


user_input = input(f'Do you want to continue processing layer 31? (y/N): ')
if user_input.lower() != 'y':
    print(          'Exiting...')
    sys.exit(1)

for activation_name in tqdm(activation_names_31, 'Layer 31'):
    
    this_amp_gb_l31 = amp_gb_l31[activation_name].values.tolist()
    this_amp_it_l31 = amp_it_l31[activation_name].values.tolist()
    this_amp_ny_l31 = amp_ny_l31[activation_name].values.tolist()

    combined_values = np.array(this_amp_gb_l31 + this_amp_it_l31 + this_amp_ny_l31)

    jenks = mapclassify.NaturalBreaks(combined_values, k=9)
    breaks = jenks.bins
    # print("Jenks Natural Breaks:", breaks)
    norm = BoundaryNorm(boundaries=breaks, ncolors=cmap.N)

    plot_activations_map(amp_gb_l31, 27700,      'GB-l31', activation_name, all_moran_df, cmap, norm, breaks)
    plot_activations_map(amp_it_l31, 25832,      'IT-l31', activation_name, all_moran_df, cmap, norm, breaks)
    plot_activations_map(amp_ny_l31, 32618, 'NYmetro-l31', activation_name, all_moran_df, cmap, norm, breaks)

    del this_amp_gb_l31, this_amp_it_l31, this_amp_ny_l31, combined_values, jenks, breaks, norm
