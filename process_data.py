import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os

# Configuration for file paths
OUTPUT_DIR = 'site/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_band(dataset, variable_name, output_filename, colormap, vmin=None, vmax=None):
    """
    Decodes a variable and saves it as a transparent PNG.
    """
    try:
        if variable_name not in dataset:
            print(f"Skipping {output_filename}: Variable '{variable_name}' not found.")
            return

        data = dataset[variable_name]
        
        # Setup the plot
        fig = plt.figure(figsize=(10, 10), frameon=False)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-125, -65, 25, 50]) # Continental US View
        
        # Remove axes and borders
        ax.outline_patch.set_visible(False)
        ax.background_patch.set_visible(False)
        plt.axis('off')

        # Plot data
        data.plot.pcolormesh(
            ax=ax, 
            transform=ccrs.PlateCarree(), 
            cmap=colormap, 
            vmin=vmin, 
            vmax=vmax,
            add_colorbar=False
        )

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        print(f"Generated {output_filename}")
        
    except Exception as e:
        print(f"Error generating {output_filename}: {e}")

def main():
    grib_file = 'latest_goes.grib2'
    
    if not os.path.exists(grib_file):
        print("No GRIB2 file found.")
        return

    # --- 1. LOAD SURFACE DATA ---
    # We explicitly ask for 'surface' to fix the "multiple values" error
    try:
        print("Loading Surface Level...")
        ds_surface = xr.open_dataset(
            grib_file, 
            engine='cfgrib', 
            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}}
        )

        # 'vis' = Visibility (Proxy for Visible Satellite)
        process_band(ds_surface, 'vis', 'blue_visible.png', 'Blues_r')
        
        # 't' = Temperature (Proxy for Infrared)
        process_band(ds_surface, 't', 'infrared.png', 'gray_r', vmin=240, vmax=310)
    except Exception as e:
        print(f"Could not process Surface level: {e}")

    # --- 2. LOAD ATMOSPHERE DATA ---
    # We open the file AGAIN to get the 'atmosphere' level (for clouds)
    try:
        print("Loading Atmosphere Level...")
        ds_atmos = xr.open_dataset(
            grib_file, 
            engine='cfgrib', 
            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'atmosphere'}}
        )
        
        # 'tcc' = Total Cloud Cover (Proxy for GeoColor/Clouds)
        process_band(ds_atmos, 'tcc', 'geocolor.png', 'gray')
    except Exception as e:
        print(f"Could not process Atmosphere level: {e}")

if __name__ == "__main__":
    main()
