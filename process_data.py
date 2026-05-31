import boto3
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import numpy as np
import os
from datetime import datetime, timezone, timedelta
from botocore import UNSIGNED
from botocore.config import Config
from scipy.ndimage import zoom

OUTPUT_DIR = 'site/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET = 'noaa-goes19'
MAX_FRAMES = 10  # rolling frame buffer per product

# Geographic extent: [west_lon, east_lon, south_lat, north_lat]
# This MUST match the imageBounds in site/index.html
# GOES-19 CONUS sector extends to ~135.7°W; use -135 to capture the full west coast.
EXTENT = [-135, -60, 20, 55]


# ---------------------------------------------------------------------------
# Custom colour maps
# ---------------------------------------------------------------------------

def _ir_colormap():
    """NWS-style rainbow IR enhancement.

    Maps brightness temperature (190 K → 310 K):
      Cold cloud tops  (190–220 K) → white / magenta / red / orange
      Moderate clouds  (230–260 K) → orange / green / cyan
      Warm clear sky   (270–310 K) → blue → dark blue → near-black
    """
    return LinearSegmentedColormap.from_list('ir_enhancement', [
        (0.00, '#ffffff'),  # 190 K  – white  (extreme cold tops)
        (0.07, '#dd00dd'),  # 199 K  – magenta
        (0.17, '#ff0000'),  # 210 K  – red
        (0.27, '#ff5500'),  # 222 K  – orange-red
        (0.37, '#ff8800'),  # 233 K  – orange (was amber, removed yellow cast)
        (0.45, '#44cc00'),  # 244 K  – green  (was yellow #ffff00 – removed)
        (0.53, '#00cc00'),  # 254 K  – green
        (0.62, '#00cccc'),  # 264 K  – cyan
        (0.72, '#0066ff'),  # 276 K  – blue
        (0.87, '#001177'),  # 294 K  – dark blue
        (1.00, '#060606'),  # 310 K  – near-black
    ])


def _wv_colormap():
    """Water-vapour enhancement colormap.

    Maps brightness temperature (195 K → 280 K):
      Cold / moist upper troposphere (195–225 K) → deep navy → royal blue
      Moderate moisture              (225–250 K) → medium blue → teal
      Warm / dry troposphere         (250–280 K) → green → orange → red
    """
    return LinearSegmentedColormap.from_list('wv_enhancement', [
        (0.00, '#00003c'),  # 195 K  – deep navy
        (0.18, '#0000cc'),  # 209 K  – royal blue
        (0.35, '#0066ee'),  # 222 K  – medium blue
        (0.50, '#00bbdd'),  # 233 K  – light blue / cyan
        (0.63, '#00bb66'),  # 242 K  – teal-green
        (0.74, '#22cc00'),  # 250 K  – green (was yellow-green #aadd00 – removed)
        (0.84, '#ff8800'),  # 258 K  – orange (was yellow #ffcc00 – removed)
        (0.92, '#ff5500'),  # 265 K  – deep orange
        (1.00, '#cc1100'),  # 280 K  – red-orange (warm / dry)
    ])


# ---------------------------------------------------------------------------
# Frame management
# ---------------------------------------------------------------------------

def shift_frames(product_base):
    """Shift existing frames back one slot to make room for a new _00 frame.

    _00 is always the newest frame; _{MAX_FRAMES-1} is the oldest.
    When the buffer is full the oldest frame is deleted before shifting.
    A legacy single-file (product.png) is migrated to the oldest slot on
    the first call so no historical imagery is lost.
    """
    legacy   = os.path.join(OUTPUT_DIR, f'{product_base}.png')
    frame_00 = os.path.join(OUTPUT_DIR, f'{product_base}_00.png')

    # One-time migration: seed the oldest slot with the pre-frame-buffer image
    if os.path.exists(legacy) and not os.path.exists(frame_00):
        seed = os.path.join(OUTPUT_DIR, f'{product_base}_{MAX_FRAMES - 1:02d}.png')
        os.rename(legacy, seed)
        print(f"  Migrated legacy {product_base}.png → {os.path.basename(seed)}")

    # Count how many frame files currently exist
    n_existing = sum(
        1 for i in range(MAX_FRAMES)
        if os.path.exists(os.path.join(OUTPUT_DIR, f'{product_base}_{i:02d}.png'))
    )

    # Drop the oldest frame only when the buffer is already at capacity
    if n_existing >= MAX_FRAMES:
        oldest = os.path.join(OUTPUT_DIR, f'{product_base}_{MAX_FRAMES - 1:02d}.png')
        if os.path.exists(oldest):
            os.remove(oldest)

    # Shift _08→_09, _07→_08, …, _00→_01
    for i in range(MAX_FRAMES - 2, -1, -1):
        src = os.path.join(OUTPUT_DIR, f'{product_base}_{i:02d}.png')
        dst = os.path.join(OUTPUT_DIR, f'{product_base}_{i + 1:02d}.png')
        if os.path.exists(src):
            os.rename(src, dst)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_latest_goes_file(s3_client, band, domain='C'):
    """Find the most recent GOES-19 ABI CMIP file for a given band.

    Searches backwards up to 6 hours to find the latest available file.
    Domain 'C' = CONUS, 'F' = Full Disk, 'M' = Mesoscale.
    """
    now = datetime.now(timezone.utc)

    for hour_offset in range(6):
        t = now - timedelta(hours=hour_offset)
        year = t.strftime('%Y')
        doy  = t.strftime('%j')
        hour = t.strftime('%H')

        prefix   = f'ABI-L2-CMIP{domain}/{year}/{doy}/{hour}/'
        band_str = f'C{band:02d}_G19'

        try:
            resp  = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
            files = [
                obj['Key'] for obj in resp.get('Contents', [])
                if band_str in obj['Key']
            ]
            if files:
                latest = sorted(files)[-1]
                print(f"  Found: {os.path.basename(latest)}")
                return latest
        except Exception as e:
            print(f"  Warning: could not list {prefix}: {e}")

    return None


def _make_figure():
    """Create a matplotlib figure sized to match the geographic extent."""
    lon_range = EXTENT[1] - EXTENT[0]
    lat_range = EXTENT[3] - EXTENT[2]
    mid_lat   = (EXTENT[2] + EXTENT[3]) / 2.0
    fig_width = 12.0
    fig_height = fig_width * lat_range / (lon_range * np.cos(np.radians(mid_lat)))

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax  = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_aspect('auto')  # prevent Cartopy equal-aspect padding; image must fill extent exactly
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig, ax


def _download_band(s3_client, band_num):
    """Download a GOES-19 ABI band.

    Returns (data_array, x_metres, y_metres, goes_proj).
    data_array contains raw float values with NaNs intact (no fill applied).
    Returns (None, None, None, None) on any failure.
    """
    print(f"  Downloading Band {band_num}...")
    key = get_latest_goes_file(s3_client, band_num)
    if key is None:
        print(f"  ERROR: No Band {band_num} data found in the last 6 hours.")
        return None, None, None, None

    local_file = f'/tmp/goes_band{band_num}.nc'
    try:
        s3_client.download_file(BUCKET, key, local_file)
        ds = xr.open_dataset(local_file, engine='netcdf4')

        cmi = ds['CMI'].values.astype(np.float32)

        proj_var  = ds['goes_imager_projection']
        sat_h     = float(proj_var.attrs['perspective_point_height'])
        sat_lon   = float(proj_var.attrs['longitude_of_projection_origin'])
        sat_sweep = str(proj_var.attrs['sweep_angle_axis'])
        x = ds['x'].values * sat_h
        y = ds['y'].values * sat_h
        goes_proj = ccrs.Geostationary(
            central_longitude=sat_lon,
            satellite_height=sat_h,
            sweep_axis=sat_sweep,
        )
        ds.close()
        return cmi, x, y, goes_proj

    except Exception as e:
        print(f"  ERROR loading Band {band_num}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    finally:
        if os.path.exists(local_file):
            os.remove(local_file)


# ---------------------------------------------------------------------------
# Single-band renderer
# ---------------------------------------------------------------------------

def process_goes_band(s3_client, band, output_filename, colormap, vmin, vmax, gamma=1.0):
    """Download and render a single GOES-19 ABI band as a transparent PNG.

    gamma – optional power-law correction applied after normalising to [0, 1].
            gamma < 1 brightens the image (e.g. 0.5 = square-root stretch).
    """
    print(f"\n--- Band {band}: {output_filename} ---")

    key = get_latest_goes_file(s3_client, band)
    if key is None:
        print(f"  ERROR: No Band {band} data found in the last 6 hours. Skipping.")
        return

    local_file = f'/tmp/goes_band{band}.nc'
    try:
        print(f"  Downloading...")
        s3_client.download_file(BUCKET, key, local_file)

        ds       = xr.open_dataset(local_file, engine='netcdf4')
        cmi_data = ds['CMI'].values  # Reflectance [0-1] for visible; BT [K] for IR/WV

        # Apply gamma correction if requested (normalise → gamma → restore range)
        if gamma != 1.0:
            normed   = np.clip((cmi_data - vmin) / (vmax - vmin), 0.0, 1.0)
            cmi_data = np.power(normed, gamma) * (vmax - vmin) + vmin

        # --- Projection parameters from the file ---
        proj_var  = ds['goes_imager_projection']
        sat_h     = float(proj_var.attrs['perspective_point_height'])
        sat_lon   = float(proj_var.attrs['longitude_of_projection_origin'])
        sat_sweep = str(proj_var.attrs['sweep_angle_axis'])

        # Convert scan angles (radians) → projection coordinates (meters)
        x = ds['x'].values * sat_h
        y = ds['y'].values * sat_h
        X, Y = np.meshgrid(x, y)

        goes_proj = ccrs.Geostationary(
            central_longitude=sat_lon,
            satellite_height=sat_h,
            sweep_axis=sat_sweep
        )

        fig, ax = _make_figure()

        ax.pcolormesh(
            X, Y, cmi_data,
            transform=goes_proj,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )

        product_base = output_filename.replace('.png', '')
        shift_frames(product_base)
        output_path = os.path.join(OUTPUT_DIR, f'{product_base}_00.png')
        plt.savefig(output_path, dpi=300, transparent=True)
        plt.close()
        print(f"  Saved: {output_path}")

        ds.close()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if os.path.exists(local_file):
            os.remove(local_file)


# ---------------------------------------------------------------------------
# GeoColor composite (day / night)
# ---------------------------------------------------------------------------

def process_geocolor(s3_client):
    """GeoColor RGB composite.

    Daytime  – pseudo-natural colour from Bands 1 and 2 with gamma correction.
    Nighttime – IR cloud layer (Band 13) blended with city-lights proxy
                (Band 7 minus thermal background) on a transparent background.
    """
    print(f"\n--- GeoColor RGB Composite ---")

    # Fetch the two visible bands needed for the RGB composite
    b1, x1, y1, goes_proj = _download_band(s3_client, 1)
    if b1 is None:
        print("  ERROR: Missing Band 1. Skipping GeoColor.")
        return

    b2, x2, y2, _ = _download_band(s3_client, 2)
    if b2 is None:
        print("  ERROR: Missing Band 2. Skipping GeoColor.")
        return

    # Band 2 is 0.5 km (2× resolution) – downsample to match Band 1 (1 km)
    b2 = np.nan_to_num(b2, nan=0.0)
    if b2.shape != b1.shape:
        zy = b1.shape[0] / b2.shape[0]
        zx = b1.shape[1] / b2.shape[1]
        b2 = zoom(b2, (zy, zx), order=1)

    b1 = np.nan_to_num(b1, nan=0.0)

    # Determine day vs night: at night Band 2 visible reflectance ≈ 0
    mean_ref = float(np.nanmean(b2))
    is_daytime = mean_ref > 0.05
    print(f"  Band 2 mean reflectance: {mean_ref:.4f}  →  {'DAYTIME' if is_daytime else 'NIGHTTIME'}")

    if is_daytime:
        # Fetch Band 13 for cloud-top enhancement (failure is non-fatal)
        bt13, *_ = _download_band(s3_client, 13)
        _render_geocolor_day(b1, b2, x1, y1, goes_proj, bt13)
    else:
        _render_geocolor_night(s3_client)


def _render_geocolor_day(b1, b2, x, y, goes_proj, bt13=None):
    """Pseudo-natural colour composite for daytime.

    bt13 is an optional Band 13 brightness-temperature array (same spatial
    footprint after resampling).  When provided, very cold cloud tops are
    blended towards bright white so deep convective anvils are clearly
    visible against land/ocean backgrounds.
    """
    R = np.clip(b2, 0, 1)
    # Synthetic green: average of red and blue channels only.
    # Omitting NIR (Band 3 / 0.86 µm) prevents the yellow cast that NIR
    # introduces over vegetated and arid land surfaces.
    G = np.clip(0.5 * b2 + 0.5 * b1, 0, 1)
    B = np.clip(b1, 0, 1)

    # Gamma correction for natural brightness
    gamma = 0.5
    R = np.power(R, gamma)
    G = np.power(G, gamma)
    B = np.power(B, gamma)

    # --- Cloud enhancement via Band 13 IR ---
    # Pixels colder than ~255 K (high cloud tops) are blended towards bright
    # white, making anvils and deep convection clearly pop out.
    if bt13 is not None:
        bt13_f = np.where(np.isnan(bt13), 320.0, bt13)
        # Band 13 is 2 km; Band 1/2/3 composite is at 1 km – upsample to match
        if bt13_f.shape != R.shape:
            zy = R.shape[0] / bt13_f.shape[0]
            zx = R.shape[1] / bt13_f.shape[1]
            bt13_f = zoom(bt13_f, (zy, zx), order=1)
        # 255 K → 0 (no enhancement); 200 K → 1 (pure-white cloud tops)
        cloud_enhance = np.clip((255.0 - bt13_f) / 55.0, 0.0, 1.0)
        strength = 0.85
        R = np.clip(R + cloud_enhance * (1.0 - R) * strength, 0.0, 1.0)
        G = np.clip(G + cloud_enhance * (1.0 - G) * strength, 0.0, 1.0)
        B = np.clip(B + cloud_enhance * (1.0 - B) * (strength + 0.05), 0.0, 1.0)

    rgb = np.dstack([R, G, B])
    fig, ax = _make_figure()
    img_extent = (x[0], x[-1], y[-1], y[0])
    ax.imshow(rgb, origin='upper', extent=img_extent,
              transform=goes_proj, aspect='auto', interpolation='none')

    shift_frames('geocolor')
    output_path = os.path.join(OUTPUT_DIR, 'geocolor_00.png')
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
    print(f"  Saved (daytime): {output_path}")


def _render_geocolor_night(s3_client):
    """Nighttime GeoColor composite.

    Cloud layer  – derived from Band 13 (10.35 µm clean IR window).
                   Cold temperatures → bright blue-white clouds.
    City lights  – derived from Band 7 (3.9 µm shortwave IR) minus the
                   thermal background estimated from Band 13.  At night,
                   cities, fires, and industrial heat sources emit
                   anomalously in 3.9 µm.
    Background   – fully transparent so the dark basemap shows through.
    """
    # Band 13: brightness temperature (K), same 2 km resolution as Band 7
    bt13, x13, y13, goes_proj = _download_band(s3_client, 13)
    if bt13 is None:
        print("  ERROR: Missing Band 13 for nighttime GeoColor. Skipping.")
        return

    # Fill off-earth NaNs with a warm value so they don't look like cloud tops
    bt13 = np.where(np.isnan(bt13), 320.0, bt13)

    # Band 7: 3.9 µm shortwave IR — optional; gracefully absent
    bt7, x7, y7, _ = _download_band(s3_client, 7)

    h, w = bt13.shape

    # ------------------------------------------------------------------
    # Cloud layer
    # ------------------------------------------------------------------
    # 275 K → no cloud (opacity 0); 220 K → deep convection (opacity 1)
    cloud_opacity = np.clip((275.0 - bt13) / 55.0, 0.0, 1.0)

    # Clouds rendered as cool blue-white (scattered light / natural night look)
    cloud_R = cloud_opacity * 0.80
    cloud_G = cloud_opacity * 0.88
    cloud_B = cloud_opacity * 1.00

    # ------------------------------------------------------------------
    # City lights layer
    # ------------------------------------------------------------------
    if bt7 is not None:
        bt7 = np.where(np.isnan(bt7), 0.0, bt7)

        # Match Band 7 resolution to Band 13 if needed
        if bt7.shape != (h, w):
            zy = h / bt7.shape[0]
            zx = w / bt7.shape[1]
            bt7 = zoom(bt7, (zy, zx), order=1)

        # At typical surface temperatures Band 7 BT runs ~12 K cooler than
        # Band 13 due to Planck function differences.  Where Band 7 exceeds
        # this offset (i.e. Band7 > Band13 − 12) there is anomalous emission
        # from city lights, fires, or industrial heat.
        city_raw = np.clip((bt7 - (bt13 - 12.0)) / 25.0, 0.0, 1.0)

        # Only paint city lights over clear, warm surface pixels
        surface_clear = (bt13 > 265.0).astype(np.float32)
        city_lights   = city_raw * surface_clear
    else:
        city_lights = np.zeros((h, w), dtype=np.float32)
        print("  WARNING: Band 7 unavailable; city lights layer disabled.")

    # ------------------------------------------------------------------
    # Compose RGBA image
    # ------------------------------------------------------------------
    # Clouds: blue-white  |  City lights: warm yellow-orange
    R = np.clip(cloud_R + city_lights * 1.00, 0.0, 1.0)
    G = np.clip(cloud_G + city_lights * 0.75, 0.0, 1.0)
    B = np.clip(cloud_B + city_lights * 0.10, 0.0, 1.0)

    # Alpha: transparent where there is nothing to show
    A = np.clip(cloud_opacity + city_lights * 2.0, 0.0, 1.0)

    rgba = np.dstack([R, G, B, A]).astype(np.float32)

    fig, ax = _make_figure()
    img_extent = (x13[0], x13[-1], y13[-1], y13[0])
    ax.imshow(rgba, origin='upper', extent=img_extent,
              transform=goes_proj, aspect='auto', interpolation='none')

    shift_frames('geocolor')
    output_path = os.path.join(OUTPUT_DIR, 'geocolor_00.png')
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
    print(f"  Saved (nighttime): {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("GOES-19 Satellite Image Processor")
    print("=" * 40)
    print(f"Extent: {EXTENT}")
    print(f"Bucket: s3://{BUCKET}")

    # Anonymous access — GOES-19 bucket is publicly readable
    s3 = boto3.client(
        's3',
        region_name='us-east-1',
        config=Config(signature_version=UNSIGNED)
    )

    # Band 2  — Visible (0.64 µm)              reflectance [0.0 – 1.0]
    # 'gray' maps 0→black (clear sky) and 1→white (bright cloud).
    # gamma=0.5 (square-root stretch) matches conventional satellite display.
    process_goes_band(s3, 2,  'visible.png',  'gray',        vmin=0.0, vmax=1.0, gamma=0.5)

    # Band 13 — Clean IR Longwave (10.35 µm)   brightness temp [K]
    # Custom NWS-style rainbow: cold tops → red/orange, warm surface → dark blue/black
    process_goes_band(s3, 13, 'infrared.png', _ir_colormap(), vmin=190, vmax=310)

    # Band 9  — Mid-Level Water Vapor (6.95 µm) brightness temp [K]
    # Custom enhancement: cold/moist → navy/blue, warm/dry → orange/red
    process_goes_band(s3, 9,  'water_vapor.png', _wv_colormap(), vmin=195, vmax=280)

    # GeoColor — natural colour (day) or IR+city-lights composite (night)
    process_geocolor(s3)

    # Write a plain-text timestamp so the website can show freshness
    ts_path = os.path.join(OUTPUT_DIR, 'last_updated.txt')
    with open(ts_path, 'w') as f:
        f.write(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))
    print(f"\nTimestamp written: {ts_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
