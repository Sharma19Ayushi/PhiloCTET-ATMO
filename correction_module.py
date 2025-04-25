import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from multiprocessing import Pool, cpu_count
from sklearn.neighbors import NearestNeighbors
import unicodedata
import logging


# Normalize strings to UTF-8 for columns
def normalize_column_to_utf8(df, column_name):
    """Normalize a specific column to ensure proper UTF-8 encoding."""
    if column_name in df.columns:
        df[column_name] = df[column_name].apply(
            lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode('utf-8')
        )
    return df


# ✅ Correct Subset Function
def correct_subset(args):
    """Function to correct concentration values for a subset of points."""
    subset_indices, conc_points, conc_ineris, conc_ineris_vals, nn_model = args

    # Ensure 'x' and 'y' are available
    if 'x' not in conc_points.columns or 'y' not in conc_points.columns:
        conc_points['x'] = conc_points.geometry.x
        conc_points['y'] = conc_points.geometry.y

    subset = conc_points.iloc[subset_indices].copy()

    # Find nearest neighbor indices
    distances, nearest_indices = nn_model.kneighbors(subset[['x', 'y']])
    nearest_indices = nearest_indices.flatten()

    # Get corresponding 'conc19' values
    nearest_conc19 = conc_ineris_vals[nearest_indices]

    # Compute new concentrations
    new_delta_conc = subset['delta_conc'].values * nearest_conc19 / subset['conc'].values
    new_conc = nearest_conc19 * (1 - subset['delta_conc'].values / subset['conc'].values)

    subset['conc'] = new_conc
    subset['delta_conc'] = new_delta_conc

    return subset


# ✅ Correction Function in Parallel
def correction(conc_points, conc_ineris):
    """Function to apply correction in parallel."""
    try:
        logging.info("Starting correction function")
        nb_cores = min(cpu_count() - 1, 4)  # Limit to 4 cores to prevent overhead

        # Convert conc_ineris concentration to NumPy array
        logging.info("Normalizing and preparing data for correction...")
        conc_ineris_vals = conc_ineris['conc19'].values

        # Normalize French-specific columns in both dataframes
        conc_points = normalize_column_to_utf8(conc_points, 'region_name')  # Example column
        conc_ineris = normalize_column_to_utf8(conc_ineris, 'region_name')
        print("Bounds of conc_points:", conc_points.total_bounds)
        print("Bounds of conc_ineris:", conc_ineris.total_bounds)

        # Prepare nearest neighbor model
        nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(conc_ineris[['x', 'y']])
        # Clip conc_ineris to the bounds of conc_points if needed
        conc_ineris = conc_ineris.cx[
                      conc_points.total_bounds[0]:conc_points.total_bounds[2],  # xmin, xmax
                      conc_points.total_bounds[1]:conc_points.total_bounds[3]  # ymin, ymax
                      ]
        # Split indices for parallel processing
        split_indices = np.array_split(np.arange(len(conc_points)), nb_cores)

        with Pool(processes=nb_cores) as pool:
            results = pool.map(correct_subset, [(idx, conc_points, conc_ineris, conc_ineris_vals, nn_model)
                                                for idx in split_indices])

        corrected_data = pd.concat(results)
        logging.info(f"Corrected data: {corrected_data.head()}")
        return corrected_data

    except Exception as e:
        logging.error(f"Error in correction function: {e}")
        return None
