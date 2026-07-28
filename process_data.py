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
from scipy.ndimage import zoom, uniform_filter, gaussian_filter

OUTPUT_DIR = 'site/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET = 'noaa-goes19'
MAX_FRAMES = 10  # rolling frame buffer per product

# Geographic extent: [west_lon, east_lon, south_lat, north_lat]
# This MUST match the imageBounds in site/index.html
# GOES-19 CONUS sector extends to ~135.7°W; use -135 to capture the full west coast.
EXTENT = [-135, -60, 20, 55]


# ---------------------------------------------------------------------------
# Experimental precipitation-rate estimator configuration
# ---------------------------------------------------------------------------
# Look this far back for a previous Band 13 scan to derive a cloud-top cooling
# rate.  The GOES-19 CONUS sector refreshes every 5 minutes, so 30 minutes is
# ~6 scans — long enough for a measurable temperature trend, short enough that
# the cloud field hasn't advected far.  Set to 0 to disable the trend term.
PRECIP_COOLING_MINUTES = 30

# Rain rates are reported in inches per hour.  The underlying regression is
# published in mm/hr, so it is converted on the way out.
MM_PER_INCH = 25.4

# Rain rates below this (in/hr) are treated as no rain and drawn transparent.
# 0.01 in/hr is the conventional "trace" threshold.
PRECIP_MIN_RATE = 0.01
# Hard ceiling (in/hr ≈ 102 mm/hr).  The bare regression curve runs away below
# ~195 K (hundreds of mm/hr), which is physically implausible; the operational
# Hydro-Estimator applies a similar cap.
PRECIP_MAX_RATE = 4.0
# Cloud-top temperatures warmer than this are assumed non-precipitating.
# Raised from 260 K: comparison against live MRMS showed the old cut removing
# most of the stratiform rain shield, whose tops sit in the 255–268 K range.
PRECIP_CLOUD_MAX_K = 270.0
# Diameter (km) of the neighbourhood used for the cloud-top texture test.
PRECIP_TEXTURE_KM = 100.0
# Minutes of GLM lightning accumulated for the deep-convection confirmation.
PRECIP_GLM_MINUTES = 15

# Gaussian smoothing (km, 1-sigma) applied to the finished rate field.  The
# per-pixel screens are noisy at the 2 km sampling of Band 13, and against a
# discrete colour scale that noise renders as salt-and-pepper speckle that no
# real rain field exhibits.  Rain rate is coherent on ~10 km scales, so
# smoothing at this length recovers the texture of a radar mosaic without
# blurring away convective cores.
PRECIP_SMOOTH_KM = 6.0

# Correct the apparent position of cloud tops for viewing parallax.  A cold top
# 12 km up is displaced ~15–25 km away from the satellite sub-point over the
# mid-latitude CONUS, which shifts the estimate north of the rain it implies.
PRECIP_PARALLAX = True
# Surface temperature and lapse rate used to turn cloud-top brightness
# temperature into a cloud-top height for the parallax shift.  A standard
# tropospheric lapse rate is accurate enough: an error of 1 km in height moves
# the correction by only ~1 km on the ground.
PRECIP_SFC_TEMP_K   = 288.0
PRECIP_LAPSE_K_PER_KM = 6.5
PRECIP_MAX_CLOUD_KM = 16.0


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
# Precipitation rate scale
# ---------------------------------------------------------------------------
# Discrete radar-style rain-rate bins (in/hr).  N+1 edges → N colour bins.
# Edges 0.01, 0.05, 0.20, 0.75, 2.00 and 4.00 land on the legend ticks in
# site/index.html.
PRECIP_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.75, 1.25, 2.00, 3.00, 4.00]

# One colour per bin, ramping cyan → blue → green → yellow → red → magenta.
# Unlike the IR/WV enhancements, yellow is kept here: on a rain-rate scale it
# is a meaningful step between green and orange rather than a colour cast.
PRECIP_COLORS = [
    '#9fe8ff',  # 0.01 – 0.02  trace
    '#4fc0f0',  # 0.02 – 0.05  very light
    '#1878d0',  # 0.05 – 0.10  light
    '#17a92a',  # 0.10 – 0.20  light-moderate
    '#4fd12a',  # 0.20 – 0.40  moderate
    '#ffe000',  # 0.40 – 0.75  moderate-heavy
    '#ff9000',  # 0.75 – 1.25  heavy
    '#ff3000',  # 1.25 – 2.00  very heavy
    '#c00040',  # 2.00 – 3.00  extreme
    '#ff40ff',  # 3.00 – 4.00+ exceptional
]

# Opacity ramp — trace rain stays translucent so the basemap reads through,
# heavy cores are nearly opaque.
PRECIP_ALPHAS = np.linspace(0.45, 0.95, len(PRECIP_COLORS))


def _hex_to_rgb(h):
    """'#rrggbb' → (r, g, b) floats in [0, 1]."""
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Experimental IR → precipitation rate estimator
# ---------------------------------------------------------------------------
# The core curve is the GOES Auto-Estimator power-law regression of Vicente,
# Scofield & Menzel (1998), which fitted 10.7 µm cloud-top brightness
# temperature against radar-derived rain rates:
#
#     R = 1.1183e11 · exp(-3.6382e-2 · T^1.2)      [mm/hr, T in K]
#
# Band 13 (10.35 µm) is the modern ABI equivalent of the GOES-8 10.7 µm
# channel the regression was built on.
AE_A = 1.1183e11
AE_B = 3.6382e-2
AE_P = 1.2

# Fixed GOES-R series geostationary geometry, used for the view-angle taper.
# Every ABI file carries these same values in goes_imager_projection.
GOES_SAT_HEIGHT = 35786023.0  # perspective point height, m
EARTH_RADIUS    = 6378137.0   # GRS80 semi-major axis, m


def _auto_estimator_rate(bt):
    """Bare Vicente et al. (1998) Auto-Estimator curve, mm/hr."""
    # Clamp the input so the exponential can't overflow on bad/fill pixels.
    bt = np.clip(bt, 180.0, 330.0)
    return AE_A * np.exp(-AE_B * np.power(bt, AE_P))


def _cloud_anomaly(bt13, km_per_px):
    """Cloud-top temperature anomaly relative to the surrounding cloud, K.

    Positive = this pixel is *colder* than the cloudy pixels around it, i.e. a
    locally towering top.  Negative = warmer than its surroundings, the
    signature of anvil that has spread downwind of the core that produced it.
    """
    size = int(round(PRECIP_TEXTURE_KM / max(km_per_px, 0.5)))
    size = max(3, size | 1)  # odd window, at least 3 px

    cloud = (bt13 < PRECIP_CLOUD_MAX_K).astype(np.float32)
    # Mean over cloudy pixels only — clear-sky warmth must not dilute the mean.
    bt_sum = uniform_filter(np.where(cloud > 0, bt13, 0.0).astype(np.float32), size=size)
    bt_cnt = uniform_filter(cloud, size=size)
    local_mean = np.where(bt_cnt > 1e-6, bt_sum / np.maximum(bt_cnt, 1e-6), bt13)

    return (local_mean - bt13).astype(np.float32)


def _texture_factor(anomaly):
    """Convective-core vs. cirrus-anvil weighting from cloud-top texture.

    The raw regression assigns rain to any cold pixel, so a wide cirrus anvil
    blooms into a huge false rain shield.  The Hydro-Estimator's fix — adopted
    here — compares each pixel to the mean temperature of the cloudy pixels
    around it.  Actively growing convective cores are locally *colder* than
    their surroundings; anvil that has spread downwind is locally *warmer*.

    The multiplier only ever suppresses (max 1.0): a colder-than-average pixel
    is already assigned more rain by the temperature curve itself, so boosting
    it again for being cold relative to its neighbours would double-count the
    same signal.  Returns ~0.1 for warm-relative anvil rising to 1.0 at and
    below the local cloud mean.
    """
    return np.interp(anomaly, [-6.0, -2.0, 0.0],
                              [0.10, 0.55, 1.00]).astype(np.float32)


def _wv_factor(bt13, bt_wv):
    """Deep-convection screen from the water-vapour minus IR difference.

    When a cloud top reaches the upper troposphere both Band 9 and Band 13 see
    the same surface, so the difference approaches zero (and goes slightly
    positive for overshooting tops that punch into the stratospheric
    inversion).  Either side of that narrow window means no deep convection:

      • Strongly negative — Band 9 still senses cold upper-tropospheric
        moisture well above a mid-level or thin cloud, so it reads far colder
        than the window channel.  Cold-but-dry cirrus lands here.
      • Strongly positive — the window channel is seeing something colder than
        the mid-troposphere itself, which no cloud top does.  That is a cold
        *surface*: snow pack or high terrain.  Without this branch a winter
        Rockies snowfield scores as extreme convection.
    """
    diff = bt_wv - bt13
    return np.interp(diff, [-35.0, -20.0, -8.0, 0.0, 5.0, 18.0],
                           [0.05, 0.25, 0.85, 1.00, 1.00, 0.00]).astype(np.float32)


def _stratiform_rate(bt13, bt15=None):
    """Light stratiform rain rate, mm/hr, for thick non-convective cloud.

    The Auto-Estimator regression is calibrated on convective rainfall and
    falls off so steeply with temperature that it is already below the display
    threshold by ~248 K — a 250 K top scores 0.14 mm/hr, which rounds to no
    rain at all.  Side-by-side with a live MRMS mosaic that is the dominant
    error: a broad synoptic rain shield producing a solid 20–35 dBZ echo over
    several states has tops in the 240–265 K band and disappears entirely,
    leaving only the scattered cold cores behind and a rain *area* several
    times too small.

    This adds a second, much flatter branch for that regime — a nimbostratus /
    thick altostratus deck raining lightly but steadily.  The rates are the
    light half of the scale by construction (roughly 0.01–0.20 in/hr from 265 K
    down to 220 K) so that the branch fills in coverage without competing with
    the convective curve where that curve is actually skilful.

    The one screen that matters here is optical thickness.  Cloud that lets the
    12.3 µm channel see part of the warm surface below is too thin to be
    raining, and thin cirrus is exactly the population this branch would
    otherwise turn into a huge false rain shield.
    """
    # Anchored against Marshall-Palmer (Z = 200·R^1.6) rates for the
    # reflectivities a synoptic rain shield actually produces: 20 dBZ ≈ 0.6,
    # 25 dBZ ≈ 1.3 and 30 dBZ ≈ 2.7 mm/hr.
    rate = np.interp(bt13, [215.0, 230.0, 240.0, 250.0, 260.0, 268.0, 272.0],
                           [  7.0,   4.5,   2.8,   1.5,   0.6,   0.12,  0.0])

    if bt15 is not None:
        # Same discriminator as _split_window_factor but applied harder: an
        # opaque deck (BTD ≈ 0–2 K) rains, a semi-transparent one does not.
        btd = bt13 - bt15
        rate = rate * np.interp(btd, [1.5, 3.0, 5.0],
                                     [1.0, 0.45, 0.03])

    return rate.astype(np.float32)


def _stratiform_wv_factor(bt13, bt_wv):
    """Water-vapour screen sized for the stratiform branch.

    _wv_factor() is built to isolate deep convection, so it penalises any cloud
    whose top sits well below the level Band 9 senses.  A mid-level raining
    deck legitimately reads 15–25 K colder in the water-vapour channel than in
    the window channel, and the convective screen would throw most of it away.

    This keeps the two genuine rejections and drops the deep-convection
    preference in between:

      • Very negative — Band 9 is sensing cold upper-tropospheric moisture far
        above a thin or broken cloud, so there is no thick deck here.
      • Positive — the window channel is colder than the mid-troposphere, which
        is a cold *surface* (snow pack, high terrain), not a cloud at all.
    """
    diff = bt_wv - bt13
    return np.interp(diff, [-45.0, -30.0, -18.0, -2.0, 2.0, 8.0],
                           [  0.0,   0.35,   0.9,  1.0, 1.0, 0.0]).astype(np.float32)


def _stratiform_texture_factor(anomaly):
    """Mild anvil penalty for the stratiform branch.

    A stratiform deck is by definition smooth, so the convective texture test
    would score all of it as spreading anvil and suppress it by 10×.  Only the
    strongly warm-relative tail — the thin outflow cirrus streaming off the
    downwind edge of a storm — is penalised here, and never below a third.
    """
    return np.interp(anomaly, [-12.0, -5.0, 0.0],
                              [  0.30,  0.70, 1.00]).astype(np.float32)


def _split_window_factor(bt13, bt15):
    """Thin-cirrus screen from the split-window difference (Band 13 − Band 15).

    The 12.3 µm window channel is more strongly absorbing than the 10.35 µm
    one, so a semi-transparent cloud — which lets both channels see some of
    the warm surface below — produces a clearly positive difference.  An
    optically thick raining cloud is a near-blackbody in both channels, so the
    difference collapses towards zero.

    This is the classic discriminator for exactly the population the estimator
    is most prone to over-reading: cold, high, thin cirrus blown downwind of a
    storm that looks identical to the raining core in a single IR channel.

    Thresholds follow the observed distribution, where the median BTD rises
    from ~1.4 K for opaque tops colder than 200 K to ~2.9 K at 250–260 K, with
    a p90 tail reaching 7–8 K in the thin-cloud range.  The knee therefore
    starts above the deep-convective median so genuine cores are not penalised.
    """
    btd = bt13 - bt15
    return np.interp(btd, [1.5, 3.0, 6.5],
                          [1.00, 0.70, 0.10]).astype(np.float32)


def _view_geometry(x, y):
    """Per-pixel viewing geometry for the geostationary grid.

    x, y are projection coordinates in metres (scan angle × satellite height).
    Broadcasts the 1-D axes rather than meshgrid, which would allocate two full
    copies of the grid.

    Returns (scan, zenith_rad, slant_m):
      scan       – angular distance from nadir along the line of sight, rad
      zenith_rad – satellite zenith angle at the surface point, rad
      slant_m    – satellite-to-surface distance, m
    """
    xs = np.asarray(x, dtype=np.float32)[None, :] / GOES_SAT_HEIGHT
    ys = np.asarray(y, dtype=np.float32)[:, None] / GOES_SAT_HEIGHT
    scan = np.sqrt(xs * xs + ys * ys)

    # Law of sines on the satellite–Earth-centre–pixel triangle:
    #   sin(scan) = (R_e / (R_e + h)) · sin(zenith)
    ratio = (EARTH_RADIUS + GOES_SAT_HEIGHT) / EARTH_RADIUS
    s = np.clip(np.sin(scan) * ratio, -1.0, 1.0)  # >1 would be off-earth
    zenith = np.arcsin(s)

    # Ray–sphere intersection for the slant range.  Off-earth pixels give a
    # negative discriminant; clamp so the result stays finite (those pixels are
    # masked out elsewhere anyway).
    r_sat = EARTH_RADIUS + GOES_SAT_HEIGHT
    disc  = EARTH_RADIUS ** 2 - (r_sat * np.sin(scan)) ** 2
    slant = r_sat * np.cos(scan) - np.sqrt(np.maximum(disc, 0.0))

    return scan.astype(np.float32), zenith.astype(np.float32), slant.astype(np.float32)


def _view_angle_factor(x, y):
    """Taper the estimate to zero towards the limb of the disk.

    Near the edge of the disk each pixel is smeared over a huge, highly
    oblique footprint and the line of sight passes through so much atmosphere
    that cloud-top temperatures are badly biased.  Beyond roughly 65° of
    satellite zenith angle the retrieval is not usable.  The CONUS sector sits
    well inside that limit, so in practice this is a no-op guard rather than a
    visible taper — it only matters if the domain is ever widened.
    """
    _, zenith, _ = _view_geometry(x, y)
    return np.interp(np.degrees(zenith), [65.0, 75.0],
                                         [1.0, 0.0]).astype(np.float32)


def _parallax_correct(rate, bt13, x, y, km_per_px):
    """Move the estimate from the cloud top down to the ground beneath it.

    Infrared assigns each pixel the position where the *cloud top* intersects
    the line of sight, not the point on the ground below it.  Seen from the
    equator at 75.2°W, a 12 km top over the mid-Atlantic is displaced roughly
    15–25 km away from the sub-satellite point — northward over the CONUS —
    so an uncorrected estimate lands consistently north of the rain it is
    predicting.  That offset is a large fraction of the width of a squall line
    and is plainly visible when the product is overlaid on a radar mosaic.

    The displacement is radial in the projection plane (the sub-satellite point
    sits at the origin) with a ground magnitude of h·tan(zenith), so the
    correction is a move along that radius.  Cloud-top height comes from
    brightness temperature via a standard lapse rate.

    The move is a forward scatter rather than a backward resample: how far a
    pixel travels depends on the height of the cloud *at that pixel*, which a
    backward pass cannot know — it would read the height at the destination,
    which for the leading edge of a storm is clear sky, and leave the edge
    behind.  Colliding pixels keep the larger rate, and the caller smooths
    afterwards, which closes the sub-pixel gaps a scatter opens up where the
    shift field stretches.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.size < 2 or ys.size < 2:
        return rate

    _, zenith, slant = _view_geometry(xs, ys)

    # Cloud-top height from brightness temperature, km → m.
    h = np.clip((PRECIP_SFC_TEMP_K - bt13) / PRECIP_LAPSE_K_PER_KM,
                0.0, PRECIP_MAX_CLOUD_KM) * 1000.0

    # Ground displacement of the apparent position, away from nadir.
    shift_m = h * np.tan(np.minimum(zenith, np.radians(80.0)))

    # Ground sample distance along the radial direction.  A pixel subtends a
    # fixed scan angle, so its footprint grows both with slant range and with
    # the obliquity of the surface: gsd = nadir_gsd · (slant/H) / cos(zenith).
    gsd_m = (km_per_px * 1000.0) * (slant / GOES_SAT_HEIGHT) / \
            np.maximum(np.cos(zenith), 0.2)

    n_px = shift_m / np.maximum(gsd_m, 1.0)

    # Unit vector pointing away from nadir in the projection plane.
    r = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)
    r = np.maximum(r, 1.0)
    ux = xs[None, :] / r
    uy = ys[:, None] / r

    # Only the raining pixels need moving, which keeps the scatter cheap.
    wet = rate > 0
    if not wet.any():
        return rate

    ny, nx = rate.shape
    rows, cols = np.nonzero(wet)

    # Column index increases with x, so a step towards nadir (−x) is a negative
    # column step.  The y axis runs top-down (y[0] is the northern edge), so a
    # −y step is a *positive* row step — carry the sign of the y spacing so the
    # code stays correct if a grid is ever stored the other way up.
    row_sign = np.sign(float(ys[1] - ys[0]))
    n = n_px[rows, cols]
    dst_col = np.rint(cols - n * ux[0, cols]).astype(np.int64)
    dst_row = np.rint(rows - n * uy[rows, 0] * row_sign).astype(np.int64)

    keep = (dst_row >= 0) & (dst_row < ny) & (dst_col >= 0) & (dst_col < nx)
    out = np.zeros_like(rate)
    np.maximum.at(out.reshape(-1),
                  dst_row[keep] * nx + dst_col[keep],
                  rate[rows[keep], cols[keep]])
    return out


def fetch_glm_flashes(s3_client, minutes=PRECIP_GLM_MINUTES, max_files=60):
    """Collect GLM lightning flash positions from the last `minutes`.

    GOES-19 carries the Geostationary Lightning Mapper alongside ABI, and its
    L2 product lands in the same bucket as one small file every 20 seconds.
    Returns (lats, lons) as float arrays, empty on any failure — lightning is
    strictly a bonus input, so nothing here is allowed to break the product.
    """
    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=minutes)

    keys = []
    for hour_offset in (1, 0):  # older hour first so keys stay time-ordered
        t = now - timedelta(hours=hour_offset)
        prefix = (f'GLM-L2-LCFA/{t.strftime("%Y")}/{t.strftime("%j")}/'
                  f'{t.strftime("%H")}/')
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
                for obj in page.get('Contents', []):
                    stamp = _parse_goes_start(obj['Key'])
                    if stamp is not None and stamp >= cutoff:
                        keys.append((stamp, obj['Key']))
        except Exception as e:
            print(f"  Warning: could not list {prefix}: {e}")

    if not keys:
        print("  Note: no GLM files found; lightning confirmation disabled.")
        return np.array([]), np.array([])

    # Keep the most recent files if the window somehow yields more than
    # expected, so a listing hiccup can't blow up the runtime.
    keys.sort()
    keys = keys[-max_files:]

    lats, lons = [], []
    local_file = '/tmp/goes_glm.nc'
    for _, key in keys:
        try:
            s3_client.download_file(BUCKET, key, local_file)
            ds = xr.open_dataset(local_file, engine='netcdf4')
            if 'flash_lat' in ds.variables and ds['flash_lat'].size:
                lats.append(ds['flash_lat'].values.astype(np.float32))
                lons.append(ds['flash_lon'].values.astype(np.float32))
            ds.close()
        except Exception:
            continue  # a single bad 20-second granule is not worth failing over
        finally:
            if os.path.exists(local_file):
                os.remove(local_file)

    if not lats:
        print(f"  GLM: {len(keys)} file(s) scanned, no flashes.")
        return np.array([]), np.array([])

    lat_arr = np.concatenate(lats)
    lon_arr = np.concatenate(lons)
    print(f"  GLM: {len(lat_arr)} flashes from {len(keys)} file(s) "
          f"over the last {minutes} min.")
    return lat_arr, lon_arr


def glm_confidence(lats, lons, x, y, goes_proj, km_per_px):
    """Grid lightning flashes onto the image grid as a 0–1 confidence field.

    Lightning is proof of a vigorous ice-phase updraught, which is exactly
    what the infrared screens are trying to infer indirectly.  Where flashes
    are present the pixel is deep convection regardless of what the cloud-top
    texture or split-window tests think.
    """
    if lats.size == 0 or len(x) < 2 or len(y) < 2:
        return None

    # Flash lat/lon → the same geostationary projection metres as the imagery
    pts = goes_proj.transform_points(ccrs.PlateCarree(), lons, lats)
    fx, fy = pts[:, 0], pts[:, 1]
    good = np.isfinite(fx) & np.isfinite(fy)
    fx, fy = fx[good], fy[good]
    if fx.size == 0:
        return None

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    ix = np.round((fx - float(x[0])) / dx).astype(np.int64)
    iy = np.round((fy - float(y[0])) / dy).astype(np.int64)

    h, w = len(y), len(x)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    if not inside.any():
        return None

    counts = np.zeros((h, w), dtype=np.float32)
    np.add.at(counts, (iy[inside], ix[inside]), 1.0)

    # Spread each flash over a ~50 km neighbourhood: GLM locates a flash to
    # roughly 10 km, and the rain it implies covers a broader area than the
    # single pixel the flash was fixed to.
    size = max(3, int(round(50.0 / max(km_per_px, 0.5))) | 1)
    density = uniform_filter(counts, size=size)          # mean flashes / pixel
    density = density * 1000.0 / (km_per_px ** 2)        # flashes / 1000 km²

    # A single isolated flash smeared over the neighbourhood lands near 0.4
    # flashes/1000 km²/15 min, so the curve stays at zero there and only
    # reaches full confidence for a genuinely electrified storm.
    return np.interp(density, [1.0, 6.0, 25.0],
                              [0.0, 0.50, 1.00]).astype(np.float32)


def _cooling_factor(bt13, bt13_prev, minutes, km_per_px=2.0):
    """Growth-rate weighting from the cloud-top temperature trend.

    A cooling top is a rising, still-building updraught and rains hardest; a
    warming top is a collapsing or spreading anvil where rainfall is already
    tapering off.  A single IR image cannot tell those two apart even though
    they look identical in temperature — comparing two scans can, and it is
    the largest single skill gain available from infrared alone.

    Weighted towards growth rather than symmetric about zero: sustained deep
    convection cools its top continuously while it builds, so a top that has
    stopped cooling is usually already past its peak.

    The two scans are half an hour apart, over which the cloud field advects
    tens of kilometres, so a raw pixel-by-pixel difference is dominated by the
    edge of the cloud sweeping across the pixel rather than by the top actually
    changing temperature.  That produces a sharp dipole of large positive and
    negative differences along every cloud boundary, and — because the factor
    used to reach as low as 0.2 — punched holes straight through otherwise
    solid rain areas.  Smoothing the trend over roughly the distance the field
    advects, and limiting how far the factor can suppress, keeps the real
    growth/decay signal while removing that artefact.
    """
    if bt13_prev is None or minutes <= 0 or bt13_prev.shape != bt13.shape:
        return 1.0

    # Normalise the trend to K per 15 minutes so the thresholds below are
    # independent of the actual gap between the two scans.
    trend = (bt13 - bt13_prev) * (15.0 / minutes)
    trend = gaussian_filter(trend.astype(np.float32),
                            sigma=max(1.0, 15.0 / max(km_per_px, 0.5)))
    return np.interp(trend, [-8.0, -2.0, 0.0, 2.0, 6.0],
                            [1.20, 1.05, 0.90, 0.70, 0.50]).astype(np.float32)


def estimate_precip_rate(bt13, bt_wv=None, bt13_prev=None,
                         prev_minutes=0, km_per_px=2.0, x=None, y=None,
                         bt_split=None, glm_conf=None):
    """Estimate surface rain rate (in/hr) from ABI infrared brightness temps.

    EXPERIMENTAL.  Infrared sees cloud *tops*, not raindrops, so this is an
    inference chain rather than a measurement.

    Two branches are summed:

      • Convective — the Auto-Estimator curve gated hard by the deep-convection
        screens (texture, water vapour, split window, cooling trend, GLM).
        Skilful for cores, produces nothing at all outside them.
      • Stratiform — a flat, light curve for optically thick cloud in the
        240–268 K range, gated mainly by optical thickness.  This is what fills
        in the broad synoptic rain shield that the convective branch cannot
        see, and it is the difference between a scatter of isolated blobs and
        a continuous rain area.

    Known limitations:

      • Warm rain from shallow maritime clouds is invisible — those tops are
        warmer than PRECIP_CLOUD_MAX_K and are scored as zero.
      • Cold cirrus that is not raining can still be scored as rain when the
        texture, water-vapour and split-window screens all fail to catch it.
      • The stratiform branch keys on cold, optically thick cloud, and a deep
        cold-season snow pack under clear skies can mimic that: the surface is
        cold enough to pass the temperature test and there is no thin-cloud
        signature to reject.  Treat cold-season output over snow-covered
        terrain with suspicion.
      • The parallax correction assumes a fixed lapse rate, so cloud-top height
        — and therefore the size of the shift — is only good to a few km.
      • The Auto-Estimator regression was fitted over the continental US in
        summer; it over-predicts in dry environments and under-predicts in
        moist tropical ones.  The operational Hydro-Estimator corrects this
        with model precipitable water and RH, which are not used here.

    bt13        – Band 13 (10.35 µm) brightness temperature, K
    bt_wv       – optional Band 9 (6.95 µm) brightness temperature, K
    bt13_prev   – optional earlier Band 13 scan on the same grid, K
    prev_minutes– minutes between bt13_prev and bt13
    km_per_px   – ground sampling distance, used to size the texture window
    x, y        – optional projection coordinates (m) enabling the limb taper
    bt_split    – optional Band 15 (12.3 µm) brightness temperature, K
    glm_conf    – optional 0–1 GLM lightning confidence field
    """
    bt13 = np.asarray(bt13, dtype=np.float32)
    # Off-earth / fill pixels become "warm" so they score as no rain.
    valid = np.isfinite(bt13)
    bt13_f = np.where(valid, bt13, 330.0)

    def _match(arr, fill=330.0):
        """Coerce a companion band onto the Band 13 grid."""
        a = np.asarray(arr, dtype=np.float32)
        a = np.where(np.isfinite(a), a, fill)
        if a.shape != bt13_f.shape:
            a = zoom(a, (bt13_f.shape[0] / a.shape[0],
                         bt13_f.shape[1] / a.shape[1]), order=1)
        return a

    wv_bt    = _match(bt_wv) if bt_wv is not None else None
    split_bt = _match(bt_split) if bt_split is not None else None
    anomaly  = _cloud_anomaly(bt13_f, km_per_px)

    # --- Convective branch -------------------------------------------------
    # The screens are accumulated separately from the rate curve so that
    # lightning can raise the floor on all of them at once (below).
    screen = _texture_factor(anomaly)

    if wv_bt is not None:
        screen = screen * _wv_factor(bt13_f, wv_bt)

    if split_bt is not None:
        screen = screen * _split_window_factor(bt13_f, split_bt)

    if bt13_prev is not None:
        screen = screen * _cooling_factor(bt13_f, _match(bt13_prev),
                                          prev_minutes, km_per_px)

    if glm_conf is not None:
        # Lightning is direct evidence of deep convection, so it overrides the
        # indirect screens rather than multiplying with them — a flashing core
        # must never be screened away as cirrus.  It only ever raises the
        # weighting, so lightning-free heavy rain is not penalised.
        screen = np.maximum(screen, _match(glm_conf, fill=0.0))

    rate = _auto_estimator_rate(bt13_f) * screen

    # --- Stratiform branch -------------------------------------------------
    strat = _stratiform_rate(bt13_f, split_bt) * _stratiform_texture_factor(anomaly)
    if wv_bt is not None:
        strat = strat * _stratiform_wv_factor(bt13_f, wv_bt)

    # Summed, not maximised: a convective core is embedded in the same deck
    # that the stratiform branch is describing, and both are falling on the
    # ground below it.  The clip at PRECIP_MAX_RATE keeps the sum bounded.
    rate = rate + strat

    # Applied last: this is a data-quality mask, not a convection screen, so
    # not even lightning should reinstate rain in unusable limb geometry.
    if x is not None and y is not None:
        rate = rate * _view_angle_factor(x, y)

    # Nothing warmer than the cloud threshold precipitates in this scheme.
    rate = np.where(bt13_f > PRECIP_CLOUD_MAX_K, 0.0, rate)
    rate = np.where(valid, rate, 0.0)

    # The regression is published in mm/hr; the product reports inches.
    rate = rate / MM_PER_INCH

    rate = np.clip(np.nan_to_num(rate, nan=0.0, posinf=PRECIP_MAX_RATE),
                   0.0, PRECIP_MAX_RATE)

    # Shift the estimate from the cloud top to the ground beneath it.
    if PRECIP_PARALLAX and x is not None and y is not None:
        rate = _parallax_correct(rate.astype(np.float32), bt13_f, x, y, km_per_px)

    # Smooth after the shift and before thresholding: it closes the sub-pixel
    # gaps the scatter opens, and it means the cut to zero follows a coherent
    # rain edge rather than carving speckle out of the interior of the field.
    if PRECIP_SMOOTH_KM > 0:
        sigma = PRECIP_SMOOTH_KM / max(km_per_px, 0.5)
        if sigma > 0.5:
            rate = gaussian_filter(rate.astype(np.float32), sigma=sigma)

    rate = np.where(rate < PRECIP_MIN_RATE, 0.0, rate)
    return rate.astype(np.float32)


def _precip_rgba(rate):
    """Map a rain-rate field to an RGBA image using the discrete scale."""
    colors = np.array([_hex_to_rgb(c) for c in PRECIP_COLORS], dtype=np.float32)
    alphas = PRECIP_ALPHAS.astype(np.float32)

    # digitize returns 0 below the first edge, so subtract 1 to index bins.
    idx = np.digitize(rate, PRECIP_LEVELS) - 1
    has_rain = idx >= 0
    idx = np.clip(idx, 0, len(PRECIP_COLORS) - 1)

    rgb = colors[idx]
    a   = np.where(has_rain, alphas[idx], 0.0)
    return np.dstack([rgb, a[..., None]]).astype(np.float32)


def _precip_stats(rate):
    """One-line coverage summary, useful for spotting a broken run in the log."""
    raining = rate >= PRECIP_MIN_RATE
    n = int(raining.sum())
    if n == 0:
        return "no precipitating pixels"
    return (f"{100.0 * n / rate.size:.2f}% of pixels raining, "
            f"max {float(rate.max()):.2f} in/hr, "
            f"mean-where-raining {float(rate[raining].mean()):.3f} in/hr")


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


def _parse_goes_start(key):
    """Extract the scan start time from a GOES filename.

    Keys look like  OR_ABI-L2-CMIPC-M6C13_G19_s20262081900208_e..._c....nc
    where the _s field is sYYYYDDDHHMMSSt (t = tenths of a second).
    Returns a timezone-aware datetime, or None if the field can't be parsed.
    """
    for part in os.path.basename(key).split('_'):
        if part.startswith('s') and len(part) >= 14 and part[1:14].isdigit():
            try:
                return datetime.strptime(part[1:14], '%Y%j%H%M%S').replace(
                    tzinfo=timezone.utc)
            except ValueError:
                return None
    return None


def get_goes_file_near(s3_client, band, minutes_back, domain='C'):
    """Find the GOES-19 file whose scan time is closest to `minutes_back` ago.

    Lists the target hour and the hour before it (the target may sit near an
    hour boundary) and picks the newest scan at or before the target time.
    Returns (key, scan_start_datetime), or (None, None) if nothing suitable
    was found.
    """
    now    = datetime.now(timezone.utc)
    target = now - timedelta(minutes=minutes_back)

    band_str   = f'C{band:02d}_G19'
    candidates = []
    for hour_offset in (0, 1):
        t = target - timedelta(hours=hour_offset)
        prefix = (f'ABI-L2-CMIP{domain}/{t.strftime("%Y")}/{t.strftime("%j")}/'
                  f'{t.strftime("%H")}/')
        try:
            resp = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
            for obj in resp.get('Contents', []):
                if band_str not in obj['Key']:
                    continue
                stamp = _parse_goes_start(obj['Key'])
                if stamp is not None:
                    candidates.append((stamp, obj['Key']))
        except Exception as e:
            print(f"  Warning: could not list {prefix}: {e}")

    # Only accept scans at or before the target time — a later one would
    # shrink the gap and could even be the current scan itself.
    candidates = [c for c in candidates if c[0] <= target]
    if not candidates:
        return None, None

    stamp, key = max(candidates, key=lambda c: c[0])
    return key, stamp


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


def _download_band(s3_client, band_num, key=None):
    """Download a GOES-19 ABI band.

    key – optional explicit S3 key; when omitted the latest scan is used.

    Returns (data_array, x_metres, y_metres, goes_proj).
    data_array contains raw float values with NaNs intact (no fill applied).
    Returns (None, None, None, None) on any failure.
    """
    print(f"  Downloading Band {band_num}...")
    if key is None:
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
# Experimental precipitation rate product
# ---------------------------------------------------------------------------

def _km_per_pixel(x):
    """Ground sampling distance in km from the projection x coordinates.

    x is scan angle × satellite height, so the spacing equals the nadir
    ground resolution.  Pixels grow away from nadir; this is the nadir value.
    """
    if x is None or len(x) < 2:
        return 2.0
    return abs(float(x[1] - x[0])) / 1000.0


def _fetch_previous_ir(s3_client, current_time):
    """Fetch the Band 13 scan ~PRECIP_COOLING_MINUTES ago for the trend term.

    current_time – scan start time of the current Band 13 image, so the gap is
                   measured scan-to-scan rather than against the wall clock.
    Returns (bt_array, gap_minutes).  Returns (None, 0) if the lookback is
    disabled or the earlier scan can't be retrieved — the estimator then
    simply runs without the cooling-rate term.
    """
    if PRECIP_COOLING_MINUTES <= 0:
        return None, 0

    key, stamp = get_goes_file_near(s3_client, 13, PRECIP_COOLING_MINUTES)
    if key is None:
        print("  Note: no earlier Band 13 scan found; skipping cooling-rate term.")
        return None, 0

    ref = current_time if current_time is not None else datetime.now(timezone.utc)
    gap = (ref - stamp).total_seconds() / 60.0
    if gap <= 0:
        print("  Note: lookback scan is not older than the current scan; "
              "skipping cooling-rate term.")
        return None, 0
    print(f"  Lookback scan: {os.path.basename(key)}  ({gap:.0f} min before current)")

    prev, _, _, _ = _download_band(s3_client, 13, key=key)
    if prev is None:
        return None, 0
    return prev, gap


def process_precip(s3_client):
    """Render the experimental IR-derived precipitation rate."""
    print(f"\n--- Precipitation Rate (experimental, IR-derived) ---")

    key_now = get_latest_goes_file(s3_client, 13)
    if key_now is None:
        print("  ERROR: No Band 13 data found. Skipping precipitation.")
        return

    bt13, x, y, goes_proj = _download_band(s3_client, 13, key=key_now)
    if bt13 is None:
        print("  ERROR: Missing Band 13. Skipping precipitation.")
        return

    # Band 9 sharpens the deep-convection screen but is not required.
    bt9, _, _, _ = _download_band(s3_client, 9)
    if bt9 is None:
        print("  WARNING: Band 9 unavailable; water-vapour screen disabled.")

    # Band 15 gives the split-window thin-cirrus screen.
    bt15, _, _, _ = _download_band(s3_client, 15)
    if bt15 is None:
        print("  WARNING: Band 15 unavailable; split-window screen disabled.")

    # The earlier scan must be fetched last so its temporary file doesn't
    # collide with the current Band 13 download.
    prev, gap = _fetch_previous_ir(s3_client, _parse_goes_start(key_now))

    km = _km_per_pixel(x)
    lats, lons = fetch_glm_flashes(s3_client)
    glm = glm_confidence(lats, lons, x, y, goes_proj, km)

    rate = estimate_precip_rate(bt13, bt_wv=bt9, bt13_prev=prev,
                                prev_minutes=gap, km_per_px=km, x=x, y=y,
                                bt_split=bt15, glm_conf=glm)
    print(f"  Grid {rate.shape[0]}×{rate.shape[1]} @ {km:.1f} km/px — {_precip_stats(rate)}")

    fig, ax = _make_figure()
    ax.imshow(_precip_rgba(rate), origin='upper',
              extent=(x[0], x[-1], y[-1], y[0]), transform=goes_proj,
              aspect='auto', interpolation='nearest')

    shift_frames('precip')
    output_path = os.path.join(OUTPUT_DIR, 'precip_00.png')
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")


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

    # Precipitation Rate — EXPERIMENTAL.  Two branches from Band 13 cloud-top
    # temperature: the Vicente et al. (1998) Auto-Estimator regression for
    # convective cores, screened with Bands 9 and 15, the 30-minute cooling
    # trend and GLM lightning; plus a light stratiform branch for optically
    # thick mid-level cloud, which is what the rain shield around those cores
    # is made of.  Corrected for view parallax.  Inference from cloud tops,
    # not a radar measurement.
    process_precip(s3)

    # Write a plain-text timestamp so the website can show freshness
    ts_path = os.path.join(OUTPUT_DIR, 'last_updated.txt')
    with open(ts_path, 'w') as f:
        f.write(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))
    print(f"\nTimestamp written: {ts_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
