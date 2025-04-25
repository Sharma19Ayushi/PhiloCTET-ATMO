import pandas as pd
import geopandas as gpd
import numpy as np
from multiprocessing import Pool, cpu_count
from shapely.geometry import Point
from scipy.spatial import cKDTree
import logging


def expo(donnees_exportees_transformed, conc_corrigee, grille_combinee):
    """
    Parallelized and optimized function for processing exposure metrics.
    """
    logging.info("Starting optimized expo function")

    try:
        # Validate input data
        required_columns = ['geometry', 'conc', 'delta_conc']
        for col in required_columns:
            if col not in conc_corrigee.columns:
                raise ValueError(f"Missing column '{col}' in conc_corrigee GeoDataFrame")

        # Build spatial index (cKDTree) once for efficiency
        coords = np.array(list(zip(conc_corrigee.geometry.x, conc_corrigee.geometry.y)))
        tree = cKDTree(coords)

        # Limit number of cores for multiprocessing
        num_cores = min(cpu_count() - 1, 4)  # Max 4 cores for stability
        subsets = np.array_split(donnees_exportees_transformed, num_cores)

        # Prepare arguments for parallel processing
        args = [(subset, conc_corrigee, grille_combinee, tree) for subset in subsets]

        # Use multiprocessing pool to process subsets
        with Pool(num_cores) as pool:
            results = pool.map(process_expo_subset, args)

        # Combine processed results into a single DataFrame
        processed_results = pd.concat(results, ignore_index=True)

        logging.info("Expo processing completed successfully")
        return processed_results

    except Exception as e:
        logging.error(f"Error in expo function: {e}")
        return pd.DataFrame()  # Ensure a DataFrame, even if empty, is returned


def process_expo_subset(args):
    """
    Optimized subset processing for computing meanconc and meandelta using spatial data.
    """
    donnees, conc_corrigee, grille_combinee, tree = args
    logging.info(f"Processing {len(donnees)} rows in subset")

    # Work on a copy of the data
    donnees = donnees.copy()

    # Group grille_combinee by 'iriscod' (optimize filtering)
    grille_grouped = grille_combinee[grille_combinee['perc'] > 0].groupby('iriscod')

    # Initialize results storage
    meanconc_results = []
    meandelta_results = []
    donnees_indices = []

    # Default values if no valid neighbors are found
    default_meanconc = 0  # Default mean concentration
    default_meandelta = 0  # Default mean delta concentration

    # Process in bulk
    for iriscode, points in grille_grouped:
        if iriscode not in donnees['iriscod'].values:
            continue  # Skip processing if no matching data

        # Extract relevant rows for the current iriscod
        donnees_subset = donnees[donnees['iriscod'] == iriscode]
        point_coords = np.array(list(zip(points.geometry.x, points.geometry.y)))

        # Query nearest neighbors in bulk
        distances, indices = tree.query(point_coords, k=1)

        # Filter valid data based on threshold
        max_distance_threshold = 5000  # Threshold based on CRS units
        valid_mask = distances <= max_distance_threshold
        if not valid_mask.any():
            logging.warning(f"No valid neighbors found for IRIS code: {iriscode}")
            # Assign default values
            meanconc_results.extend([default_meanconc] * len(donnees_subset))
            meandelta_results.extend([default_meandelta] * len(donnees_subset))
            donnees_indices.extend(donnees_subset.index)
            continue

        # Retrieve valid points and associated values
        indices = indices[valid_mask]
        points = points.iloc[valid_mask]
        nearest_points = conc_corrigee.iloc[indices]

        # Retrieve concentration and delta values
        conc_values = nearest_points['conc'].values
        delta_values = nearest_points['delta_conc'].values
        weights = points['perc'].values

        # Normalize weights
        weights = np.clip(weights, 0, 1)  # Ensure weights are fractional
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum  # Normalize weights
        else:
            logging.warning(f"Zero weight sum for IRIS code: {iriscode}")
            # Assign default values
            meanconc_results.extend([default_meanconc] * len(donnees_subset))
            meandelta_results.extend([default_meandelta] * len(donnees_subset))
            donnees_indices.extend(donnees_subset.index)
            continue

        # Perform vectorized weighted average
        meanconc = np.dot(conc_values, weights)
        meandelta = np.dot(delta_values, weights)

        # Store results
        meanconc_results.extend([meanconc] * len(donnees_subset))
        meandelta_results.extend([meandelta] * len(donnees_subset))
        donnees_indices.extend(donnees_subset.index)

    # Apply results to the DataFrame (update in bulk)
    donnees.loc[donnees_indices, 'meanconc'] = meanconc_results
    donnees.loc[donnees_indices, 'meandelta'] = meandelta_results

    # Fill missing values (if any) with default values
    donnees['meanconc'] = donnees['meanconc'].fillna(default_meanconc)
    donnees['meandelta'] = donnees['meandelta'].fillna(default_meandelta)

    return donnees

