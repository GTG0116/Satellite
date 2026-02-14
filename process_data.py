import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os

# Configuration for file paths
OUTPUT_DIR = 'site/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_band(dataset, variable_name, output_filename, colormap, vmin=None, vmax=None):
    """
    Decodes a GRIB2 variable and saves it as a transparent PNG for the web map.
    """
    try:
        # Select the variable
        data = dataset[variable_name]
        
        # Setup the plot with a transparent background for the web map
        fig = plt.figure(figsize=(10, 10), frameon=False)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-100, -60, 20, 50]) # Focus on US East Coast/Atlantic (Adjust as needed)
        
        # Remove axes and borders so we just get the data overlay
        ax.outline_patch.set_visible(False)
        ax.background_patch.set_visible(False)
        plt.axis('off')

        # Plot the data
        # Note: GRIB2 coordinates can vary. Ensure lat/lon are correctly mapped.
        data.plot.pcolormesh(
            ax=ax, 
            transform=ccrs.PlateCarree(), 
            cmap=colormap, 
            vmin=vmin, 
            vmax=vmax,
            add_colorbar=False
        )

        # Save as PNG with transparent background
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        print(f"Generated {output_filename}")
        
    except KeyError:
        print(f"Variable {variable_name} not found in GRIB2 file.")

def main():
    # 1. DOWNLOAD GRIB2 DATA
    # For this example, we assume a file 'latest_goes.grib2' exists.
    # In production, use boto3 or requests to fetch from NOAA/NCEP.
    # Example: specific forecast model or satellite derived product in GRIB2
    grib_file = 'latest_goes.grib2' 
    
    if not os.path.exists(grib_file):
        print("No GRIB2 file found. Skipping processing (Add download logic here).")
        return

    # 2. DECODE DATA
    # We use cfgrib engine which relies on eccodes
    ds = xr.open_dataset(grib_file, engine='cfgrib')

    # 3. GENERATE IMAGES
    
    # INFRARED (Band 13 equivalent)
    # Variable names depend on your specific GRIB2 source. 
    # Common names: 'tcc' (Cloud Cover), 't' (Temperature), or specific GOES codes.
    # Here we assume a Brightness Temperature variable exists.
    process_band(ds, 't', 'infrared.png', 'gray_r', vmin=200, vmax=300)

    # BLUE VISIBLE (Band 1 equivalent)
    # Often 'vis' or similar in GRIB2
    process_band(ds, 'vis', 'blue_visible.png', 'Blues_r')

    # GEOCOLOR (Synthetic)
    # Real GeoColor is complex. We approximate by overlaying visible on IR or using a custom cmap.
    # If your GRIB contains specific RGB composite channels, use them here.
    process_band(ds, 'r', 'geocolor.png', 'terrain') 

if __name__ == "__main__":
    main()
