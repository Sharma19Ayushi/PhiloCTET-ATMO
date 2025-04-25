import os
import numpy as np
import pandas as pd
import geopandas as gpd
from netCDF4 import Dataset
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
from multiprocessing import cpu_count
import concurrent.futures

# Function that returns a list of points in France and their concentrations from INERIS (cartography)
def coordo_ineris(pol):
    print("Starting coordo_ineris function")
    # Standardize pollutant names
    pol = {
        "ug_PM25_RH50_high": "ug_PM25_RH50",
        "ug_PM25_RH50_low": "ug_PM25_RH50",
        "ug_NO2_high": "ug_NO2",
        "ug_NO2_low": "ug_NO2"
    }.get(pol, pol)

    # Define paths to the data files
    path_INERIS_2019 = {
        "ug_PM25_RH50": "data/1-processed-data/SHERPA/conc-2019/Reanalysed_FRA_2019_PM25_avgannual_Ineris_v.Jan2024.nc",
        "ug_NO2": "data/1-processed-data/SHERPA/conc-2019/Reanalysed_FRA_2019_NO2_avgannual_Ineris_v.Jan2024.nc"
    }.get(pol)

    var = "PM25" if pol == "ug_PM25_RH50" else "NO2"

    # Check if the file exists
    if not os.path.exists(path_INERIS_2019):
        raise FileNotFoundError(f"File not found: {path_INERIS_2019}")

    # Load the data file
    print(f"Loading data from {path_INERIS_2019}")
    with Dataset(path_INERIS_2019) as nc_file_conc:
        latitude = nc_file_conc.variables['lat'][:]
        longitude = nc_file_conc.variables['lon'][:]
        conc19 = nc_file_conc.variables[var][:]

    # Create a DataFrame with latitude, longitude, and concentration values
    lon, lat = np.meshgrid(longitude, latitude)
    df = pd.DataFrame({
        'x': lon.ravel(),
        'y': lat.ravel(),
        'conc19': conc19.ravel()
    }).dropna()

    # Convert to a GeoDataFrame
    conc_ineris = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326")
    print("Finished processing coordo_ineris function")

    return conc_ineris


import unicodedata
def normalize_column_to_utf8(df, column_name):
    """Normalize a specific column to ensure proper UTF-8 encoding."""
    if column_name in df.columns:
        df[column_name] = df[column_name].apply(
            lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode('utf-8')
        )
    return df
