import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

# Folder paths
RAW_DATA_FOLDER = "./raw_data"
PROCESSED_DATA_FOLDER = "./processed_data"

# Ensure the processed data folder exists
os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

# Function to process and save images from HDF files
def process_and_save_images():
    files = [f for f in os.listdir(RAW_DATA_FOLDER) if f.endswith('.hdf')]

    if not files:
        print("No HDF files found in the raw_data folder.")
        return

    for file in files:
        file_path = os.path.join(RAW_DATA_FOLDER, file)

        try:
            # Open the HDF file using netCDF4
            dataset = netCDF4.Dataset(file_path, mode='r')

            # Extract latitude, longitude, and relevant data variables
            if 'Latitude' in dataset.variables and 'Longitude' in dataset.variables:
                latitudes = dataset.variables['Latitude'][:]
                longitudes = dataset.variables['Longitude'][:]
            else:
                print(f"Skipping {file}: Latitude or Longitude variables not found.")
                continue

            # Assuming 'Optical_Depth_Land_And_Ocean' is the key for pollution data
            if 'Optical_Depth_Land_And_Ocean' in dataset.variables:
                pollution_data = dataset.variables['Optical_Depth_Land_And_Ocean'][:]
            else:
                print(f"Skipping {file}: Pollution data not found.")
                continue

            # Replace fill values with NaN for clarity
            pollution_data = np.array(pollution_data)
            pollution_data = np.where(pollution_data == dataset.variables['Optical_Depth_Land_And_Ocean']._FillValue, np.nan, pollution_data)

            # Create a plot for the processed data
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(longitudes, latitudes, pollution_data, cmap='jet')
            plt.colorbar(label='Pollution Level')
            plt.title(f"Pollution Visualization: {file}")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")

            # Save the plot as an image
            image_path = os.path.join(PROCESSED_DATA_FOLDER, f"{os.path.splitext(file)[0]}.png")
            plt.savefig(image_path)
            plt.close()

            print(f"Processed and saved: {image_path}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    print("Starting to process raw HDF files...")
    process_and_save_images()
    print("Processing complete. Images saved in the processed_data folder.")
