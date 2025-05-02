import requests
import zipfile
import io
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

### UK

# Retrieve file from GeoNames
geonames_GB_url = 'https://download.geonames.org/export/dump/GB.zip'
response = requests.get(geonames_GB_url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
zip_file.extractall(path='storage/')

# Load data from txt file and convert to geodataframe
df_geonames_GB = pd.read_csv(
    'storage/GB.txt', 
    sep='\t', header=None, 
    names=[
        'geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
        'feature class', 'feature code', 'country code', 'cc2', 'admin1 code',
        'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation',
        'dem', 'timezone', 'modification date'
    ])
gdf_geonames_GB = gpd.GeoDataFrame(
    df_geonames_GB, 
    geometry=gpd.points_from_xy(
        df_geonames_GB.longitude, 
        df_geonames_GB.latitude
        ), 
    crs='EPSG:4326'
    )


# Save both as GeoJSON
gdf_geonames_GB.to_file('storage/geonames_GB.geojson', driver='GeoJSON')
# and as CSV
gdf_geonames_GB.drop(columns="geometry").to_csv('storage/geonames_GB.csv', index=False)


### Italy

# Retrieve file from GeoNames
geonames_IT_url = 'https://download.geonames.org/export/dump/IT.zip'
response = requests.get(geonames_IT_url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
zip_file.extractall(path='storage/')

# Load data from txt file and convert to geodataframe
df_geonames_IT = pd.read_csv(
    'storage/IT.txt', 
    sep='\t', header=None, 
    names=[
        'geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
        'feature class', 'feature code', 'country code', 'cc2', 'admin1 code',
        'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation',
        'dem', 'timezone', 'modification date'
    ])
gdf_geonames_IT = gpd.GeoDataFrame(
    df_geonames_IT, 
    geometry=gpd.points_from_xy(
        df_geonames_IT.longitude, 
        df_geonames_IT.latitude
        ), 
    crs='EPSG:4326'
    )


# Save both as GeoJSON
gdf_geonames_IT.to_file('storage/geonames_IT.geojson', driver='GeoJSON')
# and as CSV
gdf_geonames_IT.drop(columns="geometry").to_csv('storage/geonames_IT.csv', index=False)


### New York

# Retrieve file from GeoNames
geonames_US_url = 'https://download.geonames.org/export/dump/US.zip'
response = requests.get(geonames_US_url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
zip_file.extractall(path='storage/')

# Load data from txt file and convert to geodataframe
df_geonames_US = pd.read_csv(
    'storage/US.txt', 
    sep='\t', header=None, 
    names=[
        'geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
        'feature class', 'feature code', 'country code', 'cc2', 'admin1 code',
        'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation',
        'dem', 'timezone', 'modification date'
    ])
df_geonames_NY = df_geonames_US[df_geonames_US['admin1 code'].isin(['NY', 'NJ', 'CT', 'PA'])]
gdf_geonames_NY = gpd.GeoDataFrame(
    df_geonames_NY, 
    geometry=gpd.points_from_xy(
        df_geonames_NY.longitude, 
        df_geonames_NY.latitude
        ), 
    crs='EPSG:4326'
    )


# Save both as GeoJSON
gdf_geonames_NY.to_file('storage/geonames_NYmetro.geojson', driver='GeoJSON')
# and as CSV
gdf_geonames_NY.drop(columns="geometry").to_csv('storage/geonames_NYmetro.csv', index=False)
