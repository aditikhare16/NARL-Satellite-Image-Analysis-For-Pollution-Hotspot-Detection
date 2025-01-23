import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Paths
RAW_DATA_FOLDER = "./raw_data/"
OUTPUT_FOLDER = "./output/"

def visualize_aod_distribution(file_path):
    """
    Visualize the distribution of Aerosol Optical Depth (AOD) values for a single HDF file.
    """
    try:
        # Open the HDF file
        with Dataset(file_path, mode="r") as hdf:
            print(f"Visualizing AOD distribution for file: {file_path}")

            # Extract AOD data
            aod_data = hdf.variables["Optical_Depth_Land_And_Ocean"][:]
            aod_data = np.ma.masked_invalid(aod_data)  # Mask invalid values

            # Flatten the data for histogram
            flattened_data = aod_data.compressed()

            # Plot Histogram
            plt.figure(figsize=(10, 6))
            plt.hist(flattened_data, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
            plt.title(f"AOD Value Distribution: {os.path.basename(file_path)}")
            plt.xlabel("AOD Values")
            plt.ylabel("Frequency")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.savefig(os.path.join(OUTPUT_FOLDER, f"aod_distribution_{os.path.basename(file_path)}.png"))
            plt.close()

            print(f"Saved AOD distribution plot for {file_path}")

    except Exception as e:
        print(f"Error visualizing AOD distribution for file {file_path}: {e}")

def visualize_comparison(files):
    """
    Visualize a comparison of AOD value averages across multiple files.
    """
    try:
        file_names = []
        avg_aod_values = []

        for file_path in files:
            with Dataset(file_path, mode="r") as hdf:
                # Extract AOD data
                aod_data = hdf.variables["Optical_Depth_Land_And_Ocean"][:]
                aod_data = np.ma.masked_invalid(aod_data)
                avg_aod = np.mean(aod_data)  # Calculate average AOD

                file_names.append(os.path.basename(file_path))
                avg_aod_values.append(avg_aod)

        # Plot Bar Chart
        plt.figure(figsize=(12, 6))
        plt.bar(file_names, avg_aod_values, color="orange", alpha=0.8)
        plt.title("Average AOD Values Across Files")
        plt.xlabel("File Name")
        plt.ylabel("Average AOD Value")
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "average_aod_comparison.png"))
        plt.close()

        print("Saved average AOD comparison plot")

    except Exception as e:
        print(f"Error visualizing comparison: {e}")

if __name__ == "__main__":
    # Load all HDF files in the raw data folder
    files = [os.path.join(RAW_DATA_FOLDER, f) for f in os.listdir(RAW_DATA_FOLDER) if f.endswith(".hdf")]

    # Visualize AOD distribution for the first file
    if files:
        visualize_aod_distribution(files[0])  # Visualize for one file
        visualize_comparison(files[:10])      # Compare averages for the first 10 files
    else:
        print("No HDF files found in the raw_data folder.")
