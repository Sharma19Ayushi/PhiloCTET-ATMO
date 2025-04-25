import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import pandas as pd
import numpy as np
from functools import partial


def generate_points(polygone, conc_points):
    """
    Generate grid points for a given polygon and associate them with concentration points.
    """
    if polygone.geometry.is_empty or not polygone.geometry.is_valid:
        return gpd.GeoDataFrame()  # Return empty GeoDataFrame if invalid

    if conc_points is None or conc_points.empty:
        return gpd.GeoDataFrame()  # Return empty GeoDataFrame if conc_points is empty

    # Filter concentration points within the polygon
    points_within = conc_points[conc_points.geometry.within(polygone.geometry)].copy()
    points_within['iriscod'] = polygone['iriscod']

    return points_within

def calculate_intersection_percentages(grille_combinee, donnees_exportees_transformed):
    """
    Iteratively calculate the intersection percentages for each grid point's rectangle and associated polygon.
    Returns:
        - GeoDataFrame: grille_combinee with calculated 'perc' values.
    """
    # Initialize 'perc' column with zeros
    grille_combinee['perc'] = 0

    # Iterate through each row in grille_combinee
    for idx, row in grille_combinee.iterrows():
        # Extract coordinates of the point
        x, y = row.geometry.x, row.geometry.y

        # Create a rectangle (polygon) around the point
        rectangle = Polygon([
            (round(x - 0.05, 3), round(y - 0.025, 3)),
            (round(x + 0.05, 3), round(y - 0.025, 3)),
            (round(x + 0.05, 3), round(y + 0.025, 3)),
            (round(x - 0.05, 3), round(y + 0.025, 3)),
            (round(x - 0.05, 3), round(y - 0.025, 3))
        ])

        # Filter the polygon corresponding to the current grid point's 'iriscod'
        iriscod = row['iriscod']
        polygon_row = donnees_exportees_transformed[donnees_exportees_transformed['iriscod'] == iriscod]

        if polygon_row.empty or polygon_row.geometry.is_empty.any():
            continue  # Skip if no matching polygon or geometry is invalid

        # Extract the polygon geometry
        polygon_geom = polygon_row.geometry.iloc[0]

        # Intersection between the rectangle and the polygon
        intersection = rectangle.intersection(polygon_geom)

        if not intersection.is_empty:
            area_intersection = intersection.area
            area_polygon = polygon_geom.area

            # Calculate the percentage of intersection
            perc = round(area_intersection / area_polygon, 3)
            grille_combinee.at[idx, 'perc'] = perc

    # Filter rows where 'perc' is greater than 0
    grille_combinee = grille_combinee[grille_combinee['perc'] > 0]

    return grille_combinee

from geopandas import GeoDataFrame  # Ensure correct imports
def process_subset(subset, conc_points):
    # Make sure subset and conc_points are correct types
    if not isinstance(subset, GeoDataFrame):
        raise TypeError(f"Expected 'subset' to be a GeoDataFrame, got {type(subset)}")
    if not isinstance(conc_points, GeoDataFrame):
        raise TypeError(f"Expected 'conc_points' to be a GeoDataFrame, got {type(conc_points)}")

    # Perform operations on the 'subset' that involve iterating over rows
    # Here, I'm assuming 'conc_points' needs to be used within the process
    result_rows = []
    for idx, row in subset.iterrows():
        # Perform some operation with `row` and `conc_points`
        result_rows.append(row)  # This is just a placeholder for actual logic

    # Convert the result back to a GeoDataFrame
    return GeoDataFrame(result_rows)



def worker_function(donnees_subset, conc_points):
    """
    Wrapper to process subsets of polygons using multiprocessing.
    """
    try:
        return process_subset(donnees_subset, conc_points)
    except Exception as e:
        print(f"Worker function error: {e}")
        return gpd.GeoDataFrame()

import multiprocessing
import geopandas as gpd
from functools import partial
from geopandas import GeoDataFrame
import pandas as pd

def association(donnees_exportees_transformed, conc_points):
    # Ensure input GeoDataFrames are valid
    if not isinstance(donnees_exportees_transformed, GeoDataFrame):
        raise TypeError("donnees_exportees_transformed must be a GeoDataFrame")
    if not isinstance(conc_points, GeoDataFrame):
        raise TypeError("conc_points must be a GeoDataFrame")

    # Split the data into smaller subsets for multiprocessing
    subsets = []  # Replace with logic to split donnees_exportees_transformed into chunks
    num_cores = 4  # Example number of cores; modify as needed

    # Use multiprocessing to process the subsets
    with multiprocessing.Pool(num_cores) as pool:
        worker_func = partial(process_subset, conc_points=conc_points)
        grids = pool.map(worker_func, subsets)  # Map the function across subsets

    # Combine results into a single GeoDataFrame
    result = GeoDataFrame(pd.concat(grids, ignore_index=True))

    return result


