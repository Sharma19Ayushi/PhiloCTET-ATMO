
import os
import seaborn as sns
from shapely.geometry import Point
import numpy as np
from shapely.ops import transform
from functools import partial
import pyproj
import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import geopandas as gpd
import contextily as ctx


# Function to read shapefiles
def read_shapefile(path, title):
    return gpd.read_file(os.path.join(path, f"{title}.shp"))

# Function to align CRS of spatial objects
def align_crs(data, target_crs):
    if data.crs != target_crs:
        data = data.to_crs(target_crs)
    return data

# Function to export data to shapefile
def export_data_shp(data, path, title):
    shp_path = os.path.join(path, f"{title}.shp")
    data.to_file(shp_path, driver='ESRI Shapefile')

# Function to plot the number of IRIS polygons based on the characteristic distance
def plot_distance(donnees_exportees):
    donnees_exportees['area'] = donnees_exportees.geometry.area
    donnees_exportees['sqrt_area'] = donnees_exportees['area'].apply(np.sqrt)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    sns.histplot(donnees_exportees['sqrt_area'], bins=30, ax=ax[0], color='blue')
    ax[0].axvline(1200, color='red', linestyle='--', linewidth=1)
    ax[0].set_title('Number of IRIS polygons based on the square root of the polygon area')
    ax[0].set_xlabel('Square root of the polygon area (m)')
    ax[0].set_ylabel('Number of IRIS polygons')
    
    donnees_exportees.plot(column='sqrt_area', ax=ax[1], legend=True, cmap='viridis')
    ctx.add_basemap(ax[1], crs=donnees_exportees.crs.to_string())
    ax[1].set_title('Map of IRIS polygons colored by the square root of the polygon area')
    
    plt.tight_layout()
    plt.show()

# Function to plot population map of over 30s for a year
def plot_carte_iris(donnees_mixtes, annee):
    pop = f"pop{annee}"
    donnees_mixtes.plot(column=pop, legend=True)
    plt.title(f"Population map of over 30s for {annee}")
    plt.show()

# Function to save population map of over 30s for a year
def save_carte_iris(donnees_mixtes, annee, path):
    pop = f"pop{annee}"
    donnees_mixtes.plot(column=pop, legend=True)
    plt.title(f"Population map of over 30s for {annee}")
    plt.savefig(path)
    plt.close()

# Function to plot exposure map using concentration data
def plot_carte_expo(result, col):
    try:
        # Validate input
        if result is None or result.empty:
            raise ValueError("Input GeoDataFrame is empty or None.")
        if col not in result.columns:
            raise KeyError(f"Column '{col}' does not exist in the data.")

        # Ensure geometries have the correct CRS
        if result.crs.to_string() != "EPSG:3857":
            result = result.to_crs(epsg=3857)

        # Plot the data
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        result.plot(column=col, ax=ax, legend=True, cmap='viridis')
        ctx.add_basemap(ax, crs=result.crs.to_string())
        ax.set_title(f"Exposure in {col} (µg/m³)")
        plt.show()
    except Exception as e:
        logging.error(f"An error occurred in plot_carte_expo: {e}")


# Function to plot exposure maps using concentration data
def plot_carte_expo(result, col, n):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # Plot exposure data from the GeoDataFrame
    result.plot(column=col, cmap="viridis", linewidth=0, ax=ax, legend=True,
                norm=Normalize(vmin=0, vmax=n))
    # Add gridlines and labels for lat/lon
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color="gray")  # Enable grid
    ax.set_axisbelow(True)  # Place grid below the polygons
    ax.tick_params(labelsize=12)  # Adjust tick label size
    ax.set_xlabel("Longitude", fontsize=12)  # X-axis label
    ax.set_ylabel("Latitude", fontsize=12)  # Y-axis label
    # Set the title
    ax.set_title("Exposure in μg/m³", fontsize=14)
    # Display the plot
    plt.tight_layout()
    plt.show()


# Function to save exposure map
def save_carte_expo(result, path, col, n):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # Plot exposure data from the GeoDataFrame
    result.plot(column=col, cmap="viridis", linewidth=0, ax=ax, legend=True,
                norm=Normalize(vmin=0, vmax=n))
    # Add gridlines and labels for lat/lon
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color="gray")  # Enable grid
    ax.set_axisbelow(True)  # Place grid below the polygons
    ax.tick_params(labelsize=12)  # Adjust tick label size
    ax.set_xlabel("Longitude", fontsize=12)  # X-axis label
    ax.set_ylabel("Latitude", fontsize=12)  # Y-axis label
    # Set the title
    ax.set_title("Exposure in μg/m³", fontsize=14)
    # Save the plot to the specified path
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


# Function to get scale value n1 based on pollutant
def echelle_n1(pol):
    if pol in ["ug_PM25_RH50_high", "ug_PM25_RH50_low"]:
        pol = "ug_PM25_RH50"
    elif pol in ["ug_NO2_high", "ug_NO2_low"]:
        pol = "ug_NO2"
    return 13 if pol == "ug_PM25_RH50" else 34

# Function to get scale value n2 based on pollutant
def echelle_n2(pol):
    if pol in ["ug_PM25_RH50_high", "ug_PM25_RH50_low"]:
        pol = "ug_PM25_RH50"
    elif pol in ["ug_NO2_high", "ug_NO2_low"]:
        pol = "ug_NO2"
    return 9 if pol == "ug_PM25_RH50" else 25

# Function to create map
def create_map(data, scenario, variable, n, titles_colors):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    data.plot(column=variable, ax=ax, legend=True, cmap='viridis')
    ax.set_title(titles_colors[scenario]["title"])
    plt.savefig(f"{scenario}_{variable}.png")  # Save map instead of displaying

print('Successfully loaded plotting command')

# Main execution
if __name__ == "__main__":
    donnees_exportees = gpd.read_file("path_to_donnees_exportees_shapefile.shp")
    grille_combinee = gpd.read_file("path_to_grille_combinee_shapefile.shp")
    donnees_merged = pd.read_csv("path_to_donnees_merged.csv")

    # Ensure proper CRS alignment
    target_crs = "EPSG:4326"
    donnees_exportees = align_crs(donnees_exportees, target_crs)
    grille_combinee = align_crs(grille_combinee, target_crs)

    # Example call to plot_distance function
    plot_distance(donnees_exportees)

    
