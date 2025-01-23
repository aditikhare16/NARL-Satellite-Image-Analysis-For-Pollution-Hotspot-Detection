import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from netCDF4 import Dataset
from sklearn.cluster import KMeans


 #####
#import pandas as pd

# Ensure the folder exists
#if not os.path.exists('./processed_data'):
   # os.makedirs('./processed_data')

# Save the processed DataFrame as a CSV file
#processed_df.to_csv(f'./processed_data/processed_{os.path.basename(file_path)}.csv', index=False)

#######


# Paths
RAW_DATA_FOLDER = "./raw_data/"
OUTPUT_FOLDER = "./output/"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_hdf_file(file_path):
    """
    Process a single HDF file to extract relevant data, detect pollution, and save outputs.
    """
    try:
        # Open the HDF file
        with Dataset(file_path, mode="r") as hdf:
            print(f"Processing file: {file_path}")

            # Print available variables to identify the desired dataset
            # Uncomment this line to explore dataset structure
            # print(hdf.variables.keys())

            # Example: Extract AOD (Aerosol Optical Depth) data
            # Replace 'Optical_Depth_Land_And_Ocean' with the actual variable name in your dataset
            aod_data = hdf.variables["Optical_Depth_Land_And_Ocean"][:]
            
            # Mask invalid data (e.g., negative or NaN values)
            aod_data = np.ma.masked_invalid(aod_data)

            # Thresholding: Set high pollution values
            high_pollution = np.where(aod_data > 0.8, 255, 0).astype(np.uint8)  # Adjust threshold as needed

            # Edge detection (Canny)
            edges = cv2.Canny(high_pollution, 100, 200)

            # Contour detection
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = np.zeros_like(high_pollution)
            cv2.drawContours(contour_image, contours, -1, (255), thickness=1)

            # K-Means Clustering (Group regions into pollution levels)
            flattened_data = aod_data.compressed().reshape(-1, 1)  # Flatten and remove masked data
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(flattened_data)
            clustered_image = np.zeros_like(aod_data, dtype=np.uint8)
            clustered_image[aod_data.mask == False] = labels  # Map labels back to image

            # Visualization: Heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(clustered_image, cmap="coolwarm", interpolation="nearest")
            plt.colorbar(label="Pollution Levels")
            plt.title(f"Pollution Map: {os.path.basename(file_path)}")
            plt.savefig(os.path.join(OUTPUT_FOLDER, f"pollution_map_{os.path.basename(file_path)}.png"))
            plt.close()

            print(f"Saved heatmap for {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_all_files():
    """
    Process all HDF files in the raw_data folder.
    """
    files = [os.path.join(RAW_DATA_FOLDER, f) for f in os.listdir(RAW_DATA_FOLDER) if f.endswith(".hdf")]

    print(f"Found {len(files)} files to process.")
    for file_path in files:
        process_hdf_file(file_path)

if __name__ == "__main__":
    process_all_files()
