import os
import statistics
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import sys

# Set up
log_file_name = 'code/003_create_tables_log.txt'
storage_by_id = 'storage/activations-mean-pooling_by-id'
# Check if log file exists
if os.path.exists(log_file_name):
    print('Log file already exists. Exiting...')
    sys.exit(1)
# Create required directories
os.makedirs(storage_by_id, exist_ok=True)


# Log the info about a dataframe
def log_info(df_to_log, df_name, log_file_name):

    with open(log_file_name, 'a') as log_file:
        # Check dataframe
        log_file.write(f'# {df_name}\n\n')
        log_file.write(f'## shape \n{  df_to_log.shape}\n\n')
        log_file.write(f'## columns \n{df_to_log.columns}\n\n')
        log_file.write(f'## head \n{   df_to_log.head()}\n\n')
        log_file.write('\n\n')
        # Check activations
        # First prompt all layers, second prompt second layer, third prompt third layer
        log_file.write('## activations \n\n')
        log_file.write(f'{df_to_log['activations'].shape=}\n')
        log_file.write(f'{df_to_log['activations']=}\n\n')
        log_file.write(f'{df_to_log['prompt'][0]=} {df_to_log['layer'][0]=}\n')
        log_file.write(f'{df_to_log['activations'][0].shape=}\n')
        log_file.write(f'{df_to_log['activations'][0]=}\n\n')
        log_file.write(f'{df_to_log['prompt'][1]=} {df_to_log['layer'][1]=}\n')
        log_file.write(f'{df_to_log['activations'][1].shape=}\n')
        log_file.write(f'{df_to_log['activations'][1]=}\n\n')
        log_file.write(f'{df_to_log['prompt'][2]=} {df_to_log['layer'][2]=}\n')
        log_file.write(f'{df_to_log['activations'][2].shape=}\n')
        log_file.write(f'{df_to_log['activations'][2]=}\n\n')
        log_file.write(f'{df_to_log['prompt'][3]=} {df_to_log['layer'][3]=}\n')
        log_file.write(f'{df_to_log['activations'][3].shape=}\n')
        log_file.write(f'{df_to_log['activations'][3]=}\n\n')
        log_file.write(f'{df_to_log['prompt'][6]=} {df_to_log['layer'][6]=}\n')
        log_file.write(f'{df_to_log['activations'][6].shape=}\n')
        log_file.write(f'{df_to_log['activations'][6]=}\n\n')
        log_file.write('\n\n')
        # Check mean_pooling
        # First prompt all layers, second prompt second layer, third prompt third layer
        log_file.write('## mean_pooling \n\n')
        log_file.write(f'{df_to_log['mean_pooling'].shape=}\n')
        log_file.write(f'{df_to_log['mean_pooling']=}\n\n')
        log_file.write(f'{df_to_log['prompt'][0]=} {df_to_log['layer'][0]=}\n')
        log_file.write(f'{df_to_log['mean_pooling'][0].shape=}\n')
        log_file.write(f'{df_to_log['mean_pooling'][0]=}\n\n')
        log_file.write(f'{df_to_log['prompt'][1]=} {df_to_log['layer'][1]=}\n')
        log_file.write(f'{df_to_log['mean_pooling'][1].shape=}\n')
        log_file.write(f'{df_to_log['mean_pooling'][1]=}\n\n')
        log_file.write(f'{df_to_log['prompt'][2]=} {df_to_log['layer'][2]=}\n')
        log_file.write(f'{df_to_log['mean_pooling'][2].shape=}\n')
        log_file.write(f'{df_to_log['mean_pooling'][2]=}\n\n')
        log_file.write(f'{df_to_log['prompt'][3]=} {df_to_log['layer'][3]=}\n')
        log_file.write(f'{df_to_log['mean_pooling'][3].shape=}\n')
        log_file.write(f'{df_to_log['mean_pooling'][3]=}\n\n')
        log_file.write(f'{df_to_log['prompt'][6]=}\n {df_to_log['layer'][6]=}')
        log_file.write(f'{df_to_log['mean_pooling'][6].shape=}\n')
        log_file.write(f'{df_to_log['mean_pooling'][6]=}\n\n')
        log_file.write('\n\n')
        # Test get activation values function
        log_file.write('## mean_pooling values \n\n')
        test_get_activation_values = df_to_log.apply(
            get_activation_values, 
            axis=1, 
            column_name='mean_pooling', 
            activation_id=0
            )
        log_file.write(f'{test_get_activation_values[0]=}\n')
        log_file.write(f'{test_get_activation_values[1]=}\n')
        log_file.write(f'{test_get_activation_values[2]=}\n')
        log_file.write(f'{test_get_activation_values[3]=}\n')
        log_file.write(f'{test_get_activation_values[6]=}\n')
        # End
        log_file.write('\n\n\n\n')


# Extract a single activation value 
# from the list in the activations column
def get_activation_values(df_row, column_name, activation_id):
    return df_row[column_name][0][activation_id]


# Slice the dataset by layer 
# and save the activations
def save_activations(full_df, layer, activations_colname, activation_ids, output_filename, log_file_name):

    # Ask the user whether to continue
    print(             f'\n\n\n')
    print(             f'Layer {layer} - {activations_colname} - {output_filename}')
    print(             f'This will create {(len(activation_ids)*4)+3} files for layer {layer}.\n')
    user_input = input(f'Do you want to continue processing layer {layer}? (y/N): ')
    if user_input.lower() != 'y':
        print(          'Exiting...')
        return
    
    with open(log_file_name, 'a') as log_file:
        log_file.write(f'\n\n\nLayer {layer} - {activations_colname} - {output_filename}\n')


    # Slice the dataset by layer
    layer_df = full_df[(full_df['layer']==layer)].copy()
    layer_df = layer_df.reset_index(drop=True)

    assert len(layer_df) == (len(full_df) // 3), f'The dataset includes 3 layers, {len(layer_df)=} should be a third of {len(full_df)=}'

    # List of all neuros in the layer
    activations_columns = {}

    
    # Retrieve the activations
    for i in tqdm (activation_ids, desc=f'{output_filename} - Layer {layer:02d}'):
        this_colname = f'act{i:06d}'
    
        # Get the activation values
        # for a single neuron
        activations_columns[this_colname] = layer_df.apply(
            get_activation_values, 
            axis=1, 
            column_name=activations_colname, 
            activation_id=i
            )
        with open(log_file_name, 'a') as log_file:
            log_file.write(f'{activations_colname} - {i} - {this_colname}: {min(activations_columns[this_colname])}, {statistics.mean(activations_columns[this_colname])}, {max(activations_columns[this_colname])}\n')
        
    
        # Save the activations
        # for a single neuron
        activation_to_save               = layer_df[['geonameid', 'feature class', 'feature code', 'latitude', 'longitude', 'prompt']].copy()
        activation_to_save               = activation_to_save.reset_index(drop=True)
        activation_to_save[this_colname] = activations_columns[this_colname]
        activation_to_save               = activation_to_save.sort_values(by='geonameid')

        assert len(activation_to_save)         == len(layer_df), f'Number of rows is incorrect ({len(activation_to_save)=} should be equal to {len(layer_df)=})'
        assert len(activation_to_save.columns) == 7,             f'Number of columns is incorrect ({len(activation_to_save.columns)=} should be equal to 7)'
        
        activation_to_save.to_csv(     f'{storage_by_id}/{output_filename}-l{layer:02d}_{this_colname}_mean-pooling_df.csv', index=False)
        activation_to_save.to_pickle(  f'{storage_by_id}/{output_filename}-l{layer:02d}_{this_colname}_mean-pooling_df.pkl')
        activation_to_save.to_parquet( f'{storage_by_id}/{output_filename}-l{layer:02d}_{this_colname}_mean-pooling_df.parquet')

    
        # Save the activations
        # as geojson
        activation_to_save['geometry'] = activation_to_save.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
        activation_gdf_to_save         = gpd.GeoDataFrame(activation_to_save, geometry='geometry')
        activation_gdf_to_save.crs     = 'EPSG:4326'
        
        assert len(activation_gdf_to_save) == len(layer_df), f'Number of rows is incorrect ({len(activation_to_save)=} should be equal to {len(layer_df)=})'
        
        activation_gdf_to_save.to_file(f'{storage_by_id}/{output_filename}-l{layer:02d}_{this_colname}_mean-pooling_df.geojson', driver='GeoJSON')

        del activation_to_save, activation_gdf_to_save

    
    assert len(activations_columns) == len(activation_ids), f'Incorrect number of columns ({len(activations_columns)=} should be equal to {len(activation_ids)=}'

    
    # Save the activations
    # for all neurons in the layer
    layer_df_info          = layer_df[['geonameid', 'feature class', 'feature code', 'latitude', 'longitude', 'prompt']].copy()
    layer_df_info          = layer_df_info.reset_index(drop=True)
    activations_columns_df = pd.DataFrame(activations_columns)
    activations_columns_df = activations_columns_df.reset_index(drop=True)
    # Combine
    activations_to_save = pd.concat([
        layer_df_info, 
        activations_columns_df
        ], axis=1)
    activations_to_save = activations_to_save.reset_index(drop=True)
    # Sort
    activations_to_save = activations_to_save.sort_values(by='geonameid')
    
    assert len(activations_to_save)         == len(layer_df),           f'Number of rows is incorrect ({len(activations_to_save)=} should be equal to {len(layer_df)=})'
    assert len(activations_to_save.columns) == len(activation_ids) + 6, f'Number of columns is incorrect ({len(activation_to_save.columns)=} should be equal to {len(activation_ids)+6})'
        
    
    activations_to_save.to_csv(    f'storage/{output_filename}-l{layer:02d}_activations-mean-pooling_df.csv', index=False)
    activations_to_save.to_pickle( f'storage/{output_filename}-l{layer:02d}_activations-mean-pooling_df.pkl')
    activations_to_save.to_parquet(f'storage/{output_filename}-l{layer:02d}_activations-mean-pooling_df.parquet')
    
    
    del activations_to_save


# Load data ---------------------------------------------------------------

gb_places_df  = pd.read_csv(      'storage/geonames_GB.csv')
gb_results_df = pd.read_pickle('storage/activations_GB.pkl')
gb_results_df = gb_results_df.merge(gb_places_df, on='geonameid', how='inner')

log_info(gb_results_df, 'gb_results_df', log_file_name)


it_places_df  = pd.read_csv(      'storage/geonames_IT.csv')
it_results_df = pd.read_pickle('storage/activations_IT.pkl')
it_results_df = it_results_df.merge(it_places_df, on='geonameid', how='inner')

log_info(it_results_df, 'it_results_df', log_file_name)


ny_places_df  = pd.read_csv(      'storage/geonames_NYmetro.csv')
ny_results_df = pd.read_pickle('storage/activations_NYmetro.pkl')
ny_results_df = ny_results_df.merge(ny_places_df, on='geonameid', how='inner')

log_info(ny_results_df, 'ny_results_df', log_file_name)


# Ask the user whether to continue
user_input = input(f'\n\nDo you want to continue to saving the activations? (y/N): ')
if user_input.lower() != 'y':
    print('Exiting...')
    sys.exit()

# Save activations --------------------------------------------------------

print('\n\n\nSave activations...')

save_activations(gb_results_df,  7, 'mean_pooling', range(4096), 'GB', log_file_name)
save_activations(gb_results_df, 15, 'mean_pooling', range(4096), 'GB', log_file_name)
save_activations(gb_results_df, 31, 'mean_pooling', range(4096), 'GB', log_file_name)

save_activations(it_results_df,  7, 'mean_pooling', range(4096), 'IT', log_file_name)
save_activations(it_results_df, 15, 'mean_pooling', range(4096), 'IT', log_file_name)
save_activations(it_results_df, 31, 'mean_pooling', range(4096), 'IT', log_file_name)

save_activations(ny_results_df,  7, 'mean_pooling', range(4096), 'NYmetro', log_file_name)
save_activations(ny_results_df, 15, 'mean_pooling', range(4096), 'NYmetro', log_file_name)
save_activations(ny_results_df, 31, 'mean_pooling', range(4096), 'NYmetro', log_file_name)

print('done.')
