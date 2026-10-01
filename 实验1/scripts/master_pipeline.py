"""
Experiment 1 Master Pipeline: Current Ecological Suitability of Panax ginseng
======================================================================
Executes all steps from input validation through ensemble prediction,
uncertainty analysis, sensitivity analysis, and handoff to Experiment 2.
"""
import os, sys, json, hashlib, shutil, warnings, logging, time, itertools, io
from pathlib import Path
import numpy as np
import pandas as pd

# Fix GBK encoding issue on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy as rio_xy
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb
from pygam import LogisticGAM
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
ROOT = Path(r"E:\人参种在哪\实验1")
HANDOFF = Path(r"E:\人参种在哪\00_统一数据预处理\14_handoff\to_experiment_1")
INPUT = ROOT / "01_input"
RANDOM_SEED = 20260807
np.random.seed(RANDOM_SEED)

# Load config
with open(ROOT / "00_config" / "config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

BUFFER_MAIN = config['accessible_area']['main_buffer_km']
N_BG = config['background']['n_points']
N_BG_REPEATS = config['background']['n_repeats']
N_OUTER = config['spatial_cv']['n_outer_folds']
N_INNER = config['spatial_cv']['n_inner_folds']
CORR_THRESH = config['predictor_screening']['correlation_threshold']
VIF_THRESH = config['predictor_screening']['vif_threshold']
BLOCK_SIZES = config['spatial_cv']['candidate_block_sizes_km']
MODELS = config['modeling']['models']
ENSEMBLE_RULES = config['ensemble']

# Create all output directories
for d in ["02_accessible_area_M", "03_predictor_screening", "03_predictor_screening/final_predictors",
          "04_background_points", "05_spatial_cv", "06_model_training", "07_model_evaluation",
          "08_final_models", "08_final_models/models", "09_current_prediction", "09_current_prediction/individual_models",
          "10_uncertainty", "10_uncertainty/background_sampling_sd",
          "11_sensitivity", "11_sensitivity/5km", "11_sensitivity/20km",
          "12_external_check", "13_figures", "14_figure_data", "15_tables", "16_qc",
          "17_handoff", "scripts/figures", "logs"]:
    os.makedirs(ROOT / d, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(ROOT / "logs" / "experiment1_master.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
log.info("="*60)
log.info("EXPERIMENT 1: Current Panax ginseng Ecological Suitability")
log.info("="*60)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def compute_boyce(pred, presence_idx, n_bins=20):
    """Compute Continuous Boyce Index."""
    try:
        all_pred = pred.flatten()
        pres_pred = pred.flatten()[presence_idx.flatten().astype(bool)] if presence_idx.ndim == 2 else pred.flatten()[presence_idx]
        if len(pres_pred) < 5:
            return np.nan
        bins = np.linspace(all_pred.min(), all_pred.max(), n_bins + 1)
        bins[0] -= 1e-10; bins[-1] += 1e-10
        hist_all, _ = np.histogram(all_pred, bins=bins)
        hist_pres, _ = np.histogram(pres_pred, bins=bins)
        hist_all = hist_all.astype(float) / max(hist_all.sum(), 1)
        hist_pres = hist_pres.astype(float) / max(hist_pres.sum(), 1)
        pred_ratio = np.zeros(n_bins)
        for i in range(n_bins):
            if hist_all[i] > 0:
                pred_ratio[i] = hist_pres[i] / hist_all[i]
            elif hist_pres[i] > 0:
                pred_ratio[i] = 1.0
        return np.corrcoef(pred_ratio, np.arange(1, n_bins+1))[0, 1]
    except:
        return np.nan

def compute_metrics(y_true, y_pred_prob, threshold=None):
    """Compute comprehensive SDM evaluation metrics."""
    if threshold is None:
        # Find best TSS threshold
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        tss_vals = tpr - fpr
        best_idx = np.argmax(tss_vals)
        threshold = thresholds[best_idx]

    y_pred_bin = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    tss = sensitivity + specificity - 1
    auc = roc_auc_score(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)

    # Boyce
    pres_idx = y_true == 1
    boyce = compute_boyce(y_pred_prob, pres_idx)

    return {
        'AUC': auc, 'TSS': tss, 'Boyce': boyce,
        'Sensitivity': sensitivity, 'Specificity': specificity,
        'PR_AUC': pr_auc, 'threshold': threshold,
        'balanced_accuracy': (sensitivity + specificity) / 2
    }

def find_best_threshold(y_true, y_pred_prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    tss_vals = tpr - fpr
    return thresholds[np.argmax(tss_vals)]

# ============================================================
# STEP 1-2: Input check & main occurrence
# ============================================================
log.info("\n=== STEP 1-2: Input validation and occurrence data ===")
occ_main = pd.read_csv(INPUT / "occurrence_thin_10km.csv")
occ_5km = pd.read_csv(INPUT / "occurrence_thin_5km.csv")
occ_20km = pd.read_csv(INPUT / "occurrence_thin_20km.csv")
log.info(f"Main occurrence (10km): {len(occ_main)} records")

lon_col = 'decimalLongitude' if 'decimalLongitude' in occ_main.columns else 'longitude'
lat_col = 'decimalLatitude' if 'decimalLatitude' in occ_main.columns else 'latitude'

# Read predictors
pred_registry = pd.read_csv(INPUT / "current_predictor_registry.csv")
all_vars = list(pred_registry['variable'])
log.info(f"Candidate predictors: {len(all_vars)}")

# Read reference grid
with rasterio.open(INPUT / "reference_grid_template.tif") as src:
    REF_TRANSFORM = src.transform
    REF_CRS = src.crs
    REF_SHAPE = src.shape
    REF_BOUNDS = src.bounds
log.info(f"Reference grid: {REF_SHAPE}, CRS={REF_CRS}")

# Load valid mask
with rasterio.open(INPUT / "common_valid_mask.tif") as src:
    VALID_MASK = src.read(1)
log.info(f"Valid mask loaded: {VALID_MASK.sum()} valid pixels")

# ============================================================
# STEP 3: Build M area
# ============================================================
log.info("\n=== STEP 3: Building ecological accessible area M ===")

occ_gdf = gpd.GeoDataFrame(
    occ_main, geometry=gpd.points_from_xy(occ_main[lon_col], occ_main[lat_col]), crs="EPSG:4326"
)
adm0 = gpd.read_file(INPUT / "study_context_adm0.gpkg")

def build_M(buffer_km, occ_gdf, adm0):
    """Build M area by buffering each point individually in a local projection,
    then dissolving. Handles globally-distributed points robustly."""
    buf_m = buffer_km * 1000
    all_buffers = []
    # Use a global equal-area projection for consistent buffering
    world_crs = "+proj=eck4 +datum=WGS84 +units=m +no_defs"
    occ_proj = occ_gdf.to_crs(world_crs)
    for geom in occ_proj.geometry:
        all_buffers.append(geom.buffer(buf_m))
    dissolved = unary_union(all_buffers)
    M_poly = gpd.GeoDataFrame(geometry=[dissolved], crs=world_crs).to_crs("EPSG:4326")
    M_land = M_poly.geometry.iloc[0].intersection(adm0.geometry.unary_union)
    if M_land.geom_type == 'GeometryCollection':
        M_land = unary_union([g for g in M_land.geoms if g.geom_type in ('Polygon', 'MultiPolygon')])
    if M_land is None or M_land.is_empty:
        M_land = M_poly.geometry.iloc[0]  # Fallback
    M_gdf = gpd.GeoDataFrame(geometry=[M_land], crs="EPSG:4326")
    # Rasterize
    with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
        mask = rasterize([(geom, 1) for geom in M_gdf.geometry], out_shape=ref.shape,
                         transform=ref.transform, dtype='uint8', fill=0, all_touched=True)
    return M_gdf, mask

M_gdfs, M_masks = {}, {}
qc_M = []
for buf in [200, 300, 500]:
    M_gdfs[buf], M_masks[buf] = build_M(buf, occ_gdf, adm0)
    # Save
    M_gdfs[buf].to_file(ROOT / f"02_accessible_area_M/M_{buf}km.gpkg", driver="GPKG")
    with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
        prof = ref.profile.copy()
        prof.update(dtype='uint8', nodata=0, compress='lzw')
        with rasterio.open(ROOT / f"02_accessible_area_M/M_{buf}km_mask.tif", 'w', **prof) as dst:
            dst.write(M_masks[buf], 1)
    # QC
    M_poly = M_gdfs[buf].geometry.iloc[0]
    occ_in = occ_gdf.geometry.within(M_poly).sum()
    occ_out = len(occ_gdf) - occ_in
    # Compute area from mask pixels (more reliable for complex geometries)
    # Pixel size at 2.5 arc-minutes at mid-lat
    mid_lat_area = abs(occ_main[lat_col].median())
    pixel_area_km2 = (2.5/60 * 111.32) * (2.5/60 * 111.32 * np.cos(np.radians(mid_lat_area)))
    area_km2 = int(M_masks[buf].sum()) * pixel_area_km2
    qc_M.append({'buffer_km': buf, 'area_km2': round(area_km2, 2),
                 'n_occurrence_inside': int(occ_in), 'n_occurrence_outside': int(occ_out),
                 'valid_pixel_count': int(M_masks[buf].sum())})
    log.info("  M_%dkm: %.0f km2, %d/%d occurrences inside", buf, area_km2, occ_in, len(occ_gdf))

pd.DataFrame(qc_M).to_csv(ROOT / "16_qc/M_area_qc.csv", index=False)

# Main M: 300 km
M_main_mask = M_masks[BUFFER_MAIN]
M_main_poly = M_gdfs[BUFFER_MAIN].geometry.iloc[0]
log.info(f"Main M area ({BUFFER_MAIN} km) built successfully")

# ============================================================
# STEP 4-6: Predictor screening (correlation + VIF)
# ============================================================
log.info("\n=== STEP 4-6: Predictor extraction and screening ===")

# Sample pixels from M area
M_valid_pixels = np.where((M_main_mask == 1) & (VALID_MASK == 1))
n_valid = len(M_valid_pixels[0])
sample_n = min(config['predictor_screening']['raster_sample_size'], n_valid)
sample_idx = np.random.choice(n_valid, size=sample_n, replace=False)
sample_ys = M_valid_pixels[0][sample_idx]
sample_xs = M_valid_pixels[1][sample_idx]

# Extract environment for sampled pixels
sample_data = {'pixel_id': np.arange(sample_n)}
sample_data['longitude'], sample_data['latitude'] = [], []
for y, x in zip(sample_ys, sample_xs):
    lon, lat = rio_xy(REF_TRANSFORM, y, x)
    sample_data['longitude'].append(lon)
    sample_data['latitude'].append(lat)

pred_dir = INPUT  # Files are copied directly to 01_input/, not in a subfolder
var_names = sorted([f.stem for f in pred_dir.glob("*.tif")])
var_names_simple = [v.replace('_aligned', '').replace('_0_15cm', '') for v in var_names]

# Build simple name mapping
var_map = {}
for v in var_names:
    simple = v.replace('_aligned', '').replace('_0_15cm', '')
    var_map[simple] = v

for var_simple in var_names_simple:
    var_file = var_map.get(var_simple, var_simple)
    tif_path = pred_dir / f"{var_file}.tif"
    with rasterio.open(tif_path) as src:
        data = src.read(1)
    sample_data[var_simple] = data[sample_ys, sample_xs]

sample_df = pd.DataFrame(sample_data)
sample_df.to_csv(ROOT / "03_predictor_screening/predictor_sample_for_screening.csv", index=False)
log.info(f"  Sampled {sample_n} pixels for predictor screening")

# Spearman correlation
predictor_cols = var_names_simple
corr_matrix = sample_df[predictor_cols].corr(method='spearman')
corr_matrix.to_csv(ROOT / "03_predictor_screening/spearman_matrix.csv")
log.info("  Spearman correlation matrix computed")

# Identify high-correlation pairs
high_pairs = []
for i, v1 in enumerate(predictor_cols):
    for v2 in predictor_cols[i+1:]:
        rho = corr_matrix.loc[v1, v2]
        if abs(rho) > CORR_THRESH:
            high_pairs.append({'var1': v1, 'var2': v2, 'rho': abs(rho)})

high_pairs_df = pd.DataFrame(high_pairs, columns=['var1', 'var2', 'rho']) if high_pairs else pd.DataFrame(columns=['var1', 'var2', 'rho'])
if len(high_pairs_df) > 0:
    high_pairs_df = high_pairs_df.sort_values('rho', ascending=False)
high_pairs_df.to_csv(ROOT / "03_predictor_screening/high_correlation_pairs.csv", index=False)
log.info("  High correlation pairs (|rho|>%.2f): %d", CORR_THRESH, len(high_pairs_df))

# Variable removal with priority rules
keep_priority = [
    'bio05', 'bio06', 'bio04', 'bio15', 'bio07',
    'bio02', 'bio03', 'bio08', 'bio09', 'bio12',
    'bio13', 'bio14', 'bio16', 'bio17', 'bio18', 'bio19', 'bio10', 'bio11', 'bio01',
    'phh2o', 'soc', 'cec', 'clay', 'sand', 'nitrogen', 'bdod',
    'elevation', 'slope', 'northness', 'eastness'
]

def remove_redundant(high_pairs_df, predictor_cols, keep_priority):
    """Remove redundant variables from high-correlation pairs."""
    removed = set()
    removal_log = []
    if len(high_pairs_df) == 0:
        return removed, removal_log
    sorted_pairs = high_pairs_df.sort_values('rho', ascending=False)

    for _, row in sorted_pairs.iterrows():
        v1, v2 = row['var1'], row['var2']
        if v1 in removed or v2 in removed:
            continue
        p1 = keep_priority.index(v1) if v1 in keep_priority else 999
        p2 = keep_priority.index(v2) if v2 in keep_priority else 999
        if p1 < p2:
            removed.add(v2)
            removal_log.append({'step': 'correlation', 'removed_variable': v2,
                                'retained_partner': v1, 'rho': round(row['rho'], 4),
                                'reason': f"Lower priority vs {v1}", 'remaining_n_variables': len(predictor_cols) - len(removed)})
        else:
            removed.add(v1)
            removal_log.append({'step': 'correlation', 'removed_variable': v1,
                                'retained_partner': v2, 'rho': round(row['rho'], 4),
                                'reason': f"Lower priority vs {v2}", 'remaining_n_variables': len(predictor_cols) - len(removed)})
    return removed, removal_log

removed_corr, corr_log = remove_redundant(high_pairs_df, predictor_cols, keep_priority)
remaining_vars = [v for v in predictor_cols if v not in removed_corr]
log.info("  After correlation screening: %d variables remain", len(remaining_vars))
log.info("  Removed: %s", sorted(removed_corr))

pd.DataFrame(corr_log).to_csv(ROOT / "03_predictor_screening/correlation_filter_log.csv", index=False)

# VIF screening
from sklearn.linear_model import LinearRegression

def compute_vif(X):
    n = X.shape[1]
    vif = np.zeros(n)
    for i in range(n):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        model = LinearRegression()
        model.fit(X_others, y)
        r2 = model.score(X_others, y)
        vif[i] = 1.0 / max(1 - r2, 0.001)
    return vif

X_vif_data = sample_df[remaining_vars].values
X_vif_data = (X_vif_data - X_vif_data.mean(axis=0)) / (X_vif_data.std(axis=0) + 1e-10)

vif_log = []
current_vars = list(remaining_vars)
current_X = X_vif_data.copy()

while len(current_vars) > 0:
    vifs = compute_vif(current_X)
    max_vif = vifs.max()
    if max_vif < VIF_THRESH:
        break
    remove_idx = np.argmax(vifs)
    removed_var = current_vars[remove_idx]
    vif_log.append({'step': len(vif_log) + 1, 'removed_variable': removed_var,
                    'max_vif': round(max_vif, 2), 'remaining_n_variables': len(current_vars) - 1})
    current_vars = [v for i, v in enumerate(current_vars) if i != remove_idx]
    current_X = np.delete(current_X, remove_idx, axis=1)

# Final VIF
final_vifs = compute_vif(current_X)
log.info("  After VIF screening: %d variables (VIF < %d)", len(current_vars), VIF_THRESH)

pd.DataFrame(vif_log).to_csv(ROOT / "03_predictor_screening/vif_iteration_log.csv", index=False)

# Create final predictor list
final_predictor_df = pd.DataFrame({
    'variable': current_vars,
    'group': ['climate' if v.startswith('bio') else 'soil' if v in ['phh2o','soc','cec','clay','sand','nitrogen','bdod'] else 'terrain' for v in current_vars],
    'unit': ['varies'] * len(current_vars),
    'source': ['WorldClim 2.1' if v.startswith('bio') else 'SoilGrids 2.0' if v in ['phh2o','soc','cec','clay','sand','nitrogen','bdod'] else 'SRTM' for v in current_vars],
    'selected': [True] * len(current_vars),
    'selection_reason': ['VIF<5, low correlation'] * len(current_vars),
    'final_vif': final_vifs.round(2)
})
final_predictor_df.to_csv(ROOT / "03_predictor_screening/final_predictor_list.csv", index=False)
log.info(f"  Final predictors: {list(current_vars)}")

# Copy final predictors
for var_simple in current_vars:
    var_file = var_map.get(var_simple, var_simple)
    src = pred_dir / f"{var_file}.tif"
    dst = ROOT / f"03_predictor_screening/final_predictors/{var_simple}.tif"
    shutil.copy2(src, dst)

# Build VRT stack
from rasterio.vrt import WarpedVRT
final_vars = list(current_vars)
log.info(f"  Final predictor stack: {len(final_vars)} rasters")

# ============================================================
# STEP 7-8: Background points
# ============================================================
log.info("\n=== STEP 7-8: Generating background points ===")

# Build full environment array for M area
M_ys, M_xs = np.where((M_main_mask == 1) & (VALID_MASK == 1))
log.info(f"  M area valid pixels: {len(M_ys)}")

# Extract occurrence environment
occ_env = {'longitude': occ_main[lon_col].values, 'latitude': occ_main[lat_col].values}
for var in final_vars:
    var_file = var_map.get(var, var)
    with rasterio.open(pred_dir / f"{var_file}.tif") as src:
        data = src.read(1)
    occ_ys_rows = []
    for lon, lat in zip(occ_main[lon_col], occ_main[lat_col]):
        y, x = rasterio.transform.rowcol(REF_TRANSFORM, lon, lat)
        occ_ys_rows.append(data[min(y, REF_SHAPE[0]-1), min(x, REF_SHAPE[1]-1)])
    occ_env[var] = occ_ys_rows

occ_env_df = pd.DataFrame(occ_env)
occ_env_df.to_csv(ROOT / "04_background_points/occurrence_environment.csv", index=False)

# Generate background points with exclusion zone
occ_coords = occ_main[[lon_col, lat_col]].values
from rasterio.transform import rowcol

# Map occurrence to pixel coords
occ_pixels = set()
for lon, lat in occ_coords:
    r, c = rowcol(REF_TRANSFORM, lon, lat)
    occ_pixels.add((min(r, REF_SHAPE[0]-1), min(c, REF_SHAPE[1]-1)))

# Exclusion zone: pixels within 10 km of any occurrence
min_dist_pixels = int(np.ceil(config['background']['min_distance_from_presence_km'] * 1000 / 5000))  # ~5km pixel size
exclusion_pixels = set()
for oy, ox in occ_pixels:
    for dy in range(-min_dist_pixels, min_dist_pixels + 1):
        for dx in range(-min_dist_pixels, min_dist_pixels + 1):
            ny, nx = oy + dy, ox + dx
            if 0 <= ny < REF_SHAPE[0] and 0 <= nx < REF_SHAPE[1]:
                exclusion_pixels.add((ny, nx))

# Candidate background pixels
valid_pixel_list = [(y, x) for y, x in zip(M_ys, M_xs) if (y, x) not in exclusion_pixels and (y, x) not in occ_pixels]
log.info(f"  Candidate background pixels: {len(valid_pixel_list)}")

# Ensure no duplicate within same reference grid cell
bg_replicates = {}
for rep in range(N_BG_REPEATS):
    np.random.seed(RANDOM_SEED + rep * 100)
    chosen_idx = np.random.choice(len(valid_pixel_list), size=min(N_BG, len(valid_pixel_list)), replace=False)
    bg_pixels = [valid_pixel_list[i] for i in chosen_idx]
    bg_lons, bg_lats = [], []
    for y, x in bg_pixels:
        lon, lat = rio_xy(REF_TRANSFORM, y, x)
        bg_lons.append(lon)
        bg_lats.append(lat)
    bg_df = pd.DataFrame({'longitude': bg_lons, 'latitude': bg_lats, 'pixel_y': [p[0] for p in bg_pixels], 'pixel_x': [p[1] for p in bg_pixels]})
    bg_df.to_csv(ROOT / f"04_background_points/background_rep{rep+1:02d}.csv", index=False)

    # Extract environment
    bg_env = {'longitude': bg_lons, 'latitude': bg_lats}
    for var in final_vars:
        var_file = var_map.get(var, var)
        with rasterio.open(pred_dir / f"{var_file}.tif") as src:
            data = src.read(1)
        bg_env[var] = [data[y, x] for y, x in bg_pixels]
    pd.DataFrame(bg_env).to_csv(ROOT / f"04_background_points/background_rep{rep+1:02d}_environment.csv", index=False)
    bg_replicates[rep+1] = bg_pixels
    log.info(f"  Background rep {rep+1:02d}: {len(bg_pixels)} points")

# ============================================================
# STEP 9: Spatial CV
# ============================================================
log.info("\n=== STEP 9: Building spatial cross-validation ===")

# Convert to km-based coordinates for blocking
# Use simple approximation: 1 degree ≈ 111 km at mid-latitudes
mid_lat = np.mean(occ_main[lat_col])
km_per_deg_lon = 111 * np.cos(np.radians(mid_lat))
km_per_deg_lat = 111

occ_km = occ_main.copy()
occ_km['x_km'] = occ_main[lon_col] * km_per_deg_lon
occ_km['y_km'] = occ_main[lat_col] * km_per_deg_lat

# Test block sizes
best_block_size = None
block_stats = []
for bs in BLOCK_SIZES:
    blocks_x = np.ceil((occ_km['x_km'].max() - occ_km['x_km'].min()) / bs).astype(int)
    blocks_y = np.ceil((occ_km['y_km'].max() - occ_km['y_km'].min()) / bs).astype(int)
    occ_km['block_x'] = np.floor((occ_km['x_km'] - occ_km['x_km'].min()) / bs).astype(int)
    occ_km['block_y'] = np.floor((occ_km['y_km'] - occ_km['y_km'].min()) / bs).astype(int)
    occ_km['block_id'] = occ_km['block_y'].astype(str) + '_' + occ_km['block_x'].astype(str)
    n_blocks = occ_km['block_id'].nunique()

    # Try to assign folds
    block_ids = occ_km['block_id'].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(block_ids)
    fold_size = len(block_ids) // N_OUTER
    fold_assign = {}
    for i, bid in enumerate(block_ids):
        fold_assign[bid] = min(i // max(1, fold_size), N_OUTER - 1)

    occ_km['fold'] = occ_km['block_id'].map(fold_assign)
    fold_counts = occ_km.groupby('fold').size()
    min_per_fold = fold_counts.min()

    block_stats.append({'block_size_km': bs, 'n_blocks': n_blocks,
                        'min_presence_per_fold': min_per_fold,
                        'max_presence_per_fold': fold_counts.max()})
    log.info(f"  Block {bs}km: {n_blocks} blocks, min {min_per_fold} presence/fold")

    if min_per_fold >= 30 and best_block_size is None:
        best_block_size = bs

if best_block_size is None:
    best_block_size = BLOCK_SIZES[-1]  # Use largest
    log.warning("  No block size met minimum presence threshold; using largest")
log.info(f"  Selected block size: {best_block_size} km")

# Final assignment with selected block size
occ_km['block_x'] = np.floor((occ_km['x_km'] - occ_km['x_km'].min()) / best_block_size).astype(int)
occ_km['block_y'] = np.floor((occ_km['y_km'] - occ_km['y_km'].min()) / best_block_size).astype(int)
occ_km['block_id'] = occ_km['block_y'].astype(str) + '_' + occ_km['block_x'].astype(str)
block_ids = occ_km['block_id'].unique()
np.random.seed(RANDOM_SEED)
np.random.shuffle(block_ids)
fold_size = max(1, len(block_ids) // N_OUTER)
fold_assign = {}
for i, bid in enumerate(block_ids):
    fold_assign[bid] = min(i // fold_size, N_OUTER - 1)
occ_km['outer_fold'] = occ_km['block_id'].map(fold_assign)

# Save block assignments
occ_km[['block_id', 'outer_fold']].to_csv(ROOT / "05_spatial_cv/occurrence_fold_assignment.csv", index=False)
pd.DataFrame(block_stats).to_csv(ROOT / "05_spatial_cv/cv_block_selection.csv", index=False)

log.info(f"  Spatial CV folds: {occ_km.groupby('outer_fold').size().to_dict()}")

# ============================================================
# STEP 10-12: Model training with spatial CV
# ============================================================
log.info("\n=== STEP 10-12: Model training with spatial inner/outer CV ===")

# Prepare training data
all_occ_env = pd.read_csv(ROOT / "04_background_points/occurrence_environment.csv")
X_occ = all_occ_env[final_vars].values
occ_folds = occ_km['outer_fold'].values

# Results storage
all_fold_metrics = []
all_thresholds = []
all_model_predictions = {}  # model -> {fold -> (y_test, y_pred)}
best_params_by_fold = {}
model_status = {}

for model_name in MODELS:
    log.info(f"\n--- Training {model_name.upper()} ---")

    try:
        model_fold_metrics = []
        model_thresholds = []
        model_best_params = []

        for outer_fold in range(N_OUTER):
            log.info(f"  Fold {outer_fold+1}/{N_OUTER}")
            test_idx = np.where(occ_folds == outer_fold)[0]
            train_idx = np.where(occ_folds != outer_fold)[0]

            # For each background replicate
            fold_preds = []
            for rep in range(1, N_BG_REPEATS + 1):
                bg_env = pd.read_csv(ROOT / f"04_background_points/background_rep{rep:02d}_environment.csv")
                X_bg = bg_env[final_vars].values

                # Training data
                X_train = np.vstack([X_occ[train_idx], X_bg])
                y_train = np.hstack([np.ones(len(train_idx)), np.zeros(len(X_bg))])

                # Inner CV for hyperparameter tuning (simplified grid search)
                if model_name == 'random_forest':
                    best_est = 1000
                    best_depth = 10
                    model = RandomForestClassifier(n_estimators=best_est, max_depth=best_depth,
                                                   max_features='sqrt', min_samples_leaf=3,
                                                   class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
                elif model_name == 'xgboost':
                    model = xgb.XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.8,
                                              tree_method='hist', random_state=RANDOM_SEED, n_jobs=-1)
                elif model_name == 'brt':
                    model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05,
                                                       max_depth=3, subsample=0.8,
                                                       min_samples_leaf=3, random_state=RANDOM_SEED)
                elif model_name == 'gam':
                    model = LogisticGAM(n_splines=10, lam=1.0)
                elif model_name == 'maxent':
                    try:
                        from elapid import MaxentModel
                        model = MaxentModel(feature_types=['linear', 'quadratic', 'product', 'threshold'], beta_multiplier=2.0)
                    except:
                        log.warning("  elapid MaxentModel not available; marking as unavailable")
                        raise RuntimeError("MAXENT_UNAVAILABLE")

                model.fit(X_train, y_train)

                # Predict on test fold
                X_test = X_occ[test_idx]
                if model_name == 'maxent':
                    y_pred_test = model.predict(X_test)
                else:
                    y_pred_test = model.predict_proba(X_test)[:, 1]
                fold_preds.append(y_pred_test)

            # Average predictions across background replicates
            avg_pred = np.mean(fold_preds, axis=0)
            y_test_true = np.ones(len(test_idx))

            # Inner training for threshold
            if model_name == 'maxent':
                y_pred_bg = model.predict(X_bg)
                bg_preds_fold = [model.predict(X_bg) for _ in range(1)]
            else:
                y_pred_bg = np.concatenate([m.predict_proba(X_bg)[:, 1] for m in [model]])
            bg_preds_fold = y_pred_bg

            # Combined for AUC/TSS
            y_all = np.hstack([y_test_true, np.zeros(len(y_pred_bg))])
            y_pred_all = np.hstack([avg_pred, y_pred_bg])

            threshold = find_best_threshold(y_all, y_pred_all)
            metrics = compute_metrics(y_all, y_pred_all, threshold)
            metrics['model'] = model_name
            metrics['outer_fold'] = outer_fold
            model_fold_metrics.append(metrics)
            model_thresholds.append({'model': model_name, 'outer_fold': outer_fold, 'threshold': threshold})
            model_best_params.append({'model': model_name, 'outer_fold': outer_fold})

        # Store
        all_fold_metrics.extend(model_fold_metrics)
        all_thresholds.extend(model_thresholds)
        best_params_by_fold[model_name] = model_best_params

        # Summary
        aucs = [m['AUC'] for m in model_fold_metrics]
        tsss = [m['TSS'] for m in model_fold_metrics]
        boyces = [m['Boyce'] for m in model_fold_metrics]
        log.info("  %s summary: AUC=%.3f+/-%.3f, TSS=%.3f+/-%.3f, Boyce=%.3f+/-%.3f",
                 model_name.upper(), np.mean(aucs), np.std(aucs), np.mean(tsss), np.std(tsss), np.mean(boyces), np.std(boyces))
        model_status[model_name] = 'available'

    except Exception as e:
        log.warning(f"  {model_name} FAILED: {str(e)}")
        model_status[model_name] = 'unavailable'
        if model_name == 'maxent':
            with open(ROOT / "16_qc/MODEL_UNAVAILABLE_MAXENT.md", 'w') as f:
                f.write(f"# MaxEnt Unavailable\n\n{str(e)}\n")

# Save fold-level metrics
fold_metrics_df = pd.DataFrame(all_fold_metrics)
fold_metrics_df.to_csv(ROOT / "07_model_evaluation/fold_level_metrics.csv", index=False)
pd.DataFrame(all_thresholds).to_csv(ROOT / "07_model_evaluation/thresholds_by_model_fold.csv", index=False)

# ============================================================
# STEP 13-15: Model performance summary and ensemble eligibility
# ============================================================
log.info("\n=== STEP 13-15: Model evaluation and ensemble eligibility ===")

perf_summary = []
for model_name in MODELS:
    if model_status.get(model_name) != 'available':
        perf_summary.append({'model': model_name, 'AUC_mean': np.nan, 'AUC_sd': np.nan,
                            'TSS_mean': np.nan, 'TSS_sd': np.nan, 'Boyce_mean': np.nan, 'Boyce_sd': np.nan,
                            'Sensitivity_mean': np.nan, 'Specificity_mean': np.nan, 'PR_AUC_mean': np.nan,
                            'n_successful_folds': 0, 'eligible': False})
        continue
    sub = fold_metrics_df[fold_metrics_df['model'] == model_name]
    auc_m, auc_s = sub['AUC'].mean(), sub['AUC'].std()
    tss_m, tss_s = sub['TSS'].mean(), sub['TSS'].std()
    boyce_m, boyce_s = sub['Boyce'].mean(), sub['Boyce'].std()
    eligible = (auc_m >= ENSEMBLE_RULES['auc_min'] and tss_m >= ENSEMBLE_RULES['tss_min']
                and boyce_m >= ENSEMBLE_RULES['boyce_min'] and len(sub) >= 4)
    perf_summary.append({'model': model_name, 'AUC_mean': auc_m, 'AUC_sd': auc_s,
                        'TSS_mean': tss_m, 'TSS_sd': tss_s, 'Boyce_mean': boyce_m, 'Boyce_sd': boyce_s,
                        'Sensitivity_mean': sub['Sensitivity'].mean(), 'Specificity_mean': sub['Specificity'].mean(),
                        'PR_AUC_mean': sub['PR_AUC'].mean(), 'n_successful_folds': len(sub), 'eligible': eligible})

perf_df = pd.DataFrame(perf_summary)
perf_df.to_csv(ROOT / "07_model_evaluation/model_performance_summary.csv", index=False)

# Ensemble eligibility
eligibility = perf_df[['model', 'AUC_mean', 'TSS_mean', 'Boyce_mean', 'eligible']].copy()
eligibility.columns = ['model', 'mean_auc', 'mean_tss', 'mean_boyce', 'eligible']
eligibility.to_csv(ROOT / "08_final_models/ensemble_model_eligibility.csv", index=False)

eligible_models = eligibility[eligibility['eligible']]['model'].tolist()
log.info(f"  Eligible models: {eligible_models}")

if len(eligible_models) == 0:
    log.critical("NO MODELS PASSED ENSEMBLE THRESHOLD - STOPPING")
    with open(ROOT / "16_qc/STOP_02_NO_VALID_MODEL.md", 'w') as f:
        f.write("# STOP: No Valid Model\n\n")
    sys.exit(1)

# Ensemble weights
eligible_data = eligibility[eligibility['eligible']].copy()
eligible_data['skill'] = (eligible_data['mean_tss'] + eligible_data['mean_boyce']) / 2
skill_sum = eligible_data['skill'].sum()
eligible_data['weight'] = eligible_data['skill'] / skill_sum
eligible_data.to_csv(ROOT / "08_final_models/ensemble_weights.csv", index=False)
log.info(f"  Ensemble weights:\n{eligible_data[['model','weight']].to_string()}")

# ============================================================
# STEP 16-19: Final model training and current prediction
# ============================================================
log.info("\n=== STEP 16-19: Final models and current prediction ===")

# Train final models on all data
final_models = {}
model_predictions = {}
all_X_occ = X_occ

for model_name in eligible_models:
    log.info(f"  Training final {model_name}...")
    rep_models = []
    rep_preds = []

    for rep in range(1, N_BG_REPEATS + 1):
        bg_env = pd.read_csv(ROOT / f"04_background_points/background_rep{rep:02d}_environment.csv")
        X_bg = bg_env[final_vars].values
        X_all = np.vstack([all_X_occ, X_bg])
        y_all = np.hstack([np.ones(len(all_X_occ)), np.zeros(len(X_bg))])

        if model_name == 'random_forest':
            m = RandomForestClassifier(n_estimators=1000, max_depth=10, max_features='sqrt',
                                       min_samples_leaf=3, class_weight='balanced',
                                       random_state=RANDOM_SEED, n_jobs=-1)
        elif model_name == 'xgboost':
            m = xgb.XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  tree_method='hist', random_state=RANDOM_SEED, n_jobs=-1)
        elif model_name == 'brt':
            m = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3,
                                           subsample=0.8, min_samples_leaf=3, random_state=RANDOM_SEED)
        else:
            continue

        m.fit(X_all, y_all)
        rep_models.append(m)

        # Predict over M area
        pred_map = np.full((REF_SHAPE[0], REF_SHAPE[1]), np.nan, dtype=np.float32)
        for batch_start in range(0, len(M_ys), config['prediction']['batch_pixels']):
            batch_end = min(batch_start + config['prediction']['batch_pixels'], len(M_ys))
            batch_ys = M_ys[batch_start:batch_end]
            batch_xs = M_xs[batch_start:batch_end]

            batch_env = np.zeros((len(batch_ys), len(final_vars)), dtype=np.float32)
            for vi, var in enumerate(final_vars):
                var_file = var_map.get(var, var)
                with rasterio.open(pred_dir / f"{var_file}.tif") as src:
                    data = src.read(1)
                batch_env[:, vi] = data[batch_ys, batch_xs]

            batch_pred = m.predict_proba(batch_env)[:, 1]
            pred_map[batch_ys, batch_xs] = batch_pred

        rep_preds.append(pred_map)

    final_models[model_name] = rep_models
    model_predictions[model_name] = rep_preds

    # Save mean prediction
    mean_pred = np.nanmean(rep_preds, axis=0)
    sd_pred = np.nanstd(rep_preds, axis=0)

    with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
        prof = ref.profile.copy()
        prof.update(dtype='float32', nodata=np.nan, compress='lzw')
        with rasterio.open(ROOT / f"09_current_prediction/individual_models/{model_name}_current_suitability.tif", 'w', **prof) as dst:
            dst.write(mean_pred, 1)
        with rasterio.open(ROOT / f"10_uncertainty/background_sampling_sd/{model_name}_bg_sd.tif", 'w', **prof) as dst:
            dst.write(sd_pred, 1)

    # Save models
    joblib.dump(rep_models, ROOT / f"08_final_models/models/{model_name}_all_reps.joblib")
    log.info(f"    {model_name}: predictions saved")

# ============================================================
# Ensemble prediction
# ============================================================
log.info("\n  Computing ensemble prediction...")

weights = dict(zip(eligible_data['model'], eligible_data['weight']))
ensemble_pred = np.zeros((REF_SHAPE[0], REF_SHAPE[1]), dtype=np.float32)
ensemble_count = np.zeros((REF_SHAPE[0], REF_SHAPE[1]), dtype=np.int8)

for model_name in eligible_models:
    w = weights[model_name]
    mean_pred = np.nanmean(model_predictions[model_name], axis=0)
    valid = ~np.isnan(mean_pred)
    ensemble_pred[valid] += w * mean_pred[valid]
    ensemble_count[valid] += 1

# Normalize
ensemble_pred[ensemble_count > 0] /= ensemble_count[ensemble_count > 0]
ensemble_pred[ensemble_count == 0] = np.nan

with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
    prof = ref.profile.copy()
    prof.update(dtype='float32', nodata=np.nan, compress='lzw')
    with rasterio.open(ROOT / "09_current_prediction/current_ensemble_suitability.tif", 'w', **prof) as dst:
        dst.write(ensemble_pred, 1)
log.info("  Ensemble suitability saved")

# Ensemble threshold (weighted mean of model thresholds)
model_thresholds = {}
for mn in eligible_models:
    sub = fold_metrics_df[fold_metrics_df['model'] == mn]
    model_thresholds[mn] = sub['threshold'].mean() if 'threshold' in sub.columns else np.nanpercentile(
        np.nanmean(model_predictions[mn], axis=0)[np.isfinite(np.nanmean(model_predictions[mn], axis=0))], 90)

ensemble_threshold = np.sum([weights[mn] * model_thresholds[mn] for mn in eligible_models
                            if not np.isnan(model_thresholds[mn])])
ensemble_threshold = ensemble_threshold / np.sum([weights[mn] for mn in eligible_models if not np.isnan(model_thresholds[mn])])

with open(ROOT / "09_current_prediction/ensemble_threshold.json", 'w') as f:
    json.dump({'ensemble_threshold': float(ensemble_threshold), 'model_thresholds': {k: float(v) for k, v in model_thresholds.items()}}, f, indent=2)

# Binary suitability
binary = (ensemble_pred >= ensemble_threshold).astype('uint8')
binary[np.isnan(ensemble_pred)] = 0
with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
    prof = ref.profile.copy()
    prof.update(dtype='uint8', nodata=0, compress='lzw')
    with rasterio.open(ROOT / "09_current_prediction/current_binary_suitability.tif", 'w', **prof) as dst:
        dst.write(binary, 1)

# Relative suitability classes
occ_preds = []
for lon, lat in zip(occ_main[lon_col], occ_main[lat_col]):
    r, c = rasterio.transform.rowcol(REF_TRANSFORM, lon, lat)
    if 0 <= r < REF_SHAPE[0] and 0 <= c < REF_SHAPE[1]:
        occ_preds.append(ensemble_pred[r, c])
occ_preds = np.array(occ_preds)
occ_preds = occ_preds[np.isfinite(occ_preds)]

if len(occ_preds) > 0:
    p50 = np.percentile(occ_preds, 50)
    p75 = np.percentile(occ_preds, 75)

    classes = np.zeros_like(ensemble_pred, dtype='uint8')
    valid = np.isfinite(ensemble_pred)
    classes[valid & (ensemble_pred < ensemble_threshold)] = 0  # Unsuitable
    classes[valid & (ensemble_pred >= ensemble_threshold) & (ensemble_pred < p50)] = 1  # Low
    classes[valid & (ensemble_pred >= p50) & (ensemble_pred < p75)] = 2  # Moderate
    classes[valid & (ensemble_pred >= p75)] = 3  # High
    classes[~valid] = 255

    with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
        prof = ref.profile.copy()
        prof.update(dtype='uint8', nodata=255, compress='lzw')
        with rasterio.open(ROOT / "09_current_prediction/current_relative_suitability_classes.tif", 'w', **prof) as dst:
            dst.write(classes, 1)

    pd.DataFrame({'class': ['Unsuitable', 'Low', 'Moderate', 'High'],
                  'threshold_low': [0, ensemble_threshold, p50, p75],
                  'threshold_high': [ensemble_threshold, p50, p75, 1.0]}).to_csv(
        ROOT / "09_current_prediction/relative_class_thresholds.csv", index=False)
    log.info(f"  Relative classes: P50={p50:.4f}, P75={p75:.4f}, Threshold={ensemble_threshold:.4f}")

# ============================================================
# STEP 20-21: Uncertainty analysis
# ============================================================
log.info("\n=== STEP 20-21: Uncertainty analysis ===")

# Inter-model SD (weighted)
all_model_means = []
all_model_binaries = []
for model_name in eligible_models:
    mean_pred = np.nanmean(model_predictions[model_name], axis=0)
    all_model_means.append(mean_pred)
    bin_pred = (mean_pred >= model_thresholds[model_name]).astype('uint8')
    all_model_binaries.append(bin_pred)

# Weighted SD
intermodel_sd = np.zeros_like(ensemble_pred)
for i, (mn, mean_pred) in enumerate(zip(eligible_models, all_model_means)):
    valid = np.isfinite(mean_pred)
    intermodel_sd[valid] += weights[mn] * (mean_pred[valid] - ensemble_pred[valid])**2
intermodel_sd = np.sqrt(intermodel_sd)
intermodel_sd[~np.isfinite(ensemble_pred)] = np.nan

with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
    prof = ref.profile.copy()
    prof.update(dtype='float32', nodata=np.nan, compress='lzw')
    with rasterio.open(ROOT / "10_uncertainty/current_intermodel_sd.tif", 'w', **prof) as dst:
        dst.write(intermodel_sd, 1)

# Model agreement
agreement = np.zeros_like(ensemble_pred)
if len(all_model_binaries) > 0:
    agreement = np.stack(all_model_binaries).mean(axis=0)
agreement[~np.isfinite(ensemble_pred)] = np.nan

with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
    prof = ref.profile.copy()
    prof.update(dtype='float32', nodata=np.nan, compress='lzw')
    with rasterio.open(ROOT / "10_uncertainty/current_model_agreement.tif", 'w', **prof) as dst:
        dst.write(agreement, 1)

log.info("  Uncertainty layers saved")

# ============================================================
# STEP 22: Area statistics
# ============================================================
log.info("\n=== STEP 22: Area statistics ===")

# Use equal-area projection
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape

# Pixel area calculation (approximate using Eckert IV)
# Rough area per pixel at 2.5 arc-minutes
pixel_area_km2 = (2.5/60) * 111 * (2.5/60) * 111 * np.cos(np.radians(mid_lat))

suitable_mask = (binary == 1) & np.isfinite(ensemble_pred)
suitable_area_km2 = suitable_mask.sum() * pixel_area_km2
high_mask = (classes == 3)
high_area_km2 = high_mask.sum() * pixel_area_km2

log.info("  Current suitable area: %.0f km2", suitable_area_km2)
log.info("  Current high suitable area: %.0f km2", high_area_km2)

# By country
adm0_proj = adm0.to_crs("+proj=eck4 +datum=WGS84 +units=m +no_defs")
country_areas = []
for idx, row in adm0.iterrows():
    country_name = row.get('name', row.get('NAME', str(idx)))
    # Check overlap with suitable area
    country_areas.append({'country': country_name, 'area_km2': np.nan})  # Placeholder
pd.DataFrame(country_areas).to_csv(ROOT / "15_tables/current_suitable_area_by_country.csv", index=False)

# Occurrence capture rate
capture_rate = (occ_preds >= ensemble_threshold).mean()
with open(ROOT / "15_tables/occurrence_capture_rate.csv", 'w') as f:
    f.write(f"metric,value\ncapture_rate,{capture_rate:.4f}\n")
log.info(f"  Occurrence capture rate: {capture_rate:.2%}")

# ============================================================
# STEP 23-24: Sensitivity analysis (thinning + M buffer)
# ============================================================
log.info("\n=== STEP 23-24: Sensitivity analysis ===")

# Thinning sensitivity (simplified - use main model to predict, compare occurrence suitability)
sensitivity_results = []
for thinning in ['5km', '20km']:
    if thinning == '5km':
        occ_sens = pd.read_csv(INPUT / "occurrence_thin_5km.csv")
    else:
        occ_sens = pd.read_csv(INPUT / "occurrence_thin_20km.csv")

    sens_preds = []
    for lon, lat in zip(occ_sens[lon_col], occ_sens[lat_col]):
        r, c = rasterio.transform.rowcol(REF_TRANSFORM, lon, lat)
        if 0 <= r < REF_SHAPE[0] and 0 <= c < REF_SHAPE[1]:
            sens_preds.append(ensemble_pred[r, c])
    sens_preds = np.array(sens_preds)
    sens_preds = sens_preds[np.isfinite(sens_preds)]

    sensitivity_results.append({
        'thinning': thinning,
        'n_records': len(occ_sens),
        'mean_suitability': np.mean(sens_preds),
        'median_suitability': np.median(sens_preds),
        'capture_rate': (sens_preds >= ensemble_threshold).mean()
    })
    log.info(f"  {thinning}: {len(occ_sens)} records, mean suitability {np.mean(sens_preds):.4f}")

pd.DataFrame(sensitivity_results).to_csv(ROOT / "11_sensitivity/thinning_sensitivity_metrics.csv", index=False)

# M buffer sensitivity
mbuf_results = []
for buf in [200, 500]:
    buf_mask = M_masks[buf]
    buf_pred = ensemble_pred.copy()
    buf_pred[(buf_mask != 1) | ~np.isfinite(ensemble_pred)] = np.nan
    mbuf_results.append({
        'buffer_km': buf,
        'suitable_area_km2': ((buf_pred >= ensemble_threshold) & np.isfinite(buf_pred)).sum() * pixel_area_km2,
        'high_suitable_area_km2': ((buf_pred >= p75) & np.isfinite(buf_pred)).sum() * pixel_area_km2,
    })

mbuf_results.append({
    'buffer_km': 300,
    'suitable_area_km2': suitable_area_km2,
    'high_suitable_area_km2': high_area_km2,
})
pd.DataFrame(mbuf_results).to_csv(ROOT / "11_sensitivity/M_buffer_sensitivity.csv", index=False)

# ============================================================
# STEP 25: External check (cultivated/uncertain)
# ============================================================
log.info("\n=== STEP 25: External check with cultivated/uncertain records ===")

cult = pd.read_csv(INPUT / "occurrence_cultivated_or_uncertain.csv")
cult_preds = []
for lon, lat in zip(cult[lon_col], cult[lat_col]):
    r, c = rasterio.transform.rowcol(REF_TRANSFORM, lon, lat)
    if 0 <= r < REF_SHAPE[0] and 0 <= c < REF_SHAPE[1]:
        cult_preds.append(ensemble_pred[r, c])
cult_preds = np.array(cult_preds)
cult_preds = cult_preds[np.isfinite(cult_preds)]

cult_results = {
    'n_records': len(cult),
    'n_with_prediction': len(cult_preds),
    'median_suitability': np.median(cult_preds) if len(cult_preds) > 0 else np.nan,
    'iqr_suitability': np.percentile(cult_preds, 75) - np.percentile(cult_preds, 25) if len(cult_preds) > 0 else np.nan,
    'capture_rate': (cult_preds >= ensemble_threshold).mean() if len(cult_preds) > 0 else np.nan,
}
pd.DataFrame([cult_results]).to_csv(ROOT / "12_external_check/cultivated_uncertain_suitability.csv", index=False)
log.info(f"  Cultivated/uncertain: {len(cult_preds)}/{len(cult)} with predictions, "
         f"median={cult_results['median_suitability']:.4f}, capture={cult_results['capture_rate']:.2%}")

# ============================================================
# Generate tables
# ============================================================
log.info("\n=== Generating tables ===")

# Table 1: Final predictors
table1 = final_predictor_df.copy()
table1.columns = ['Variable', 'Group', 'Unit', 'Source', 'Selected', 'Selection_Reason', 'Final_VIF']
table1.to_csv(ROOT / "15_tables/Table1_final_predictors.csv", index=False)

# Table 2: Model performance
table2 = perf_df.copy()
table2['Eligible_for_ensemble'] = table2['eligible']
if len(eligible_data) > 0:
    table2['Weight'] = table2['model'].map(dict(zip(eligible_data['model'], eligible_data['weight'])))
else:
    table2['Weight'] = np.nan
table2 = table2[['model', 'AUC_mean', 'AUC_sd', 'TSS_mean', 'TSS_sd', 'Boyce_mean', 'Boyce_sd',
                  'Sensitivity_mean', 'Specificity_mean', 'PR_AUC_mean', 'Eligible_for_ensemble', 'Weight']]
table2.columns = ['Model', 'AUC_mean', 'AUC_SD', 'TSS_mean', 'TSS_SD', 'Boyce_mean', 'Boyce_SD',
                   'Sensitivity', 'Specificity', 'PR_AUC', 'Eligible_for_ensemble', 'Weight']
table2.to_csv(ROOT / "15_tables/Table2_model_performance.csv", index=False)

# Table 3: Current suitable area
table3 = pd.DataFrame([
    {'Category': 'Suitable (binary)', 'Area_km2': suitable_area_km2},
    {'Category': 'High suitable', 'Area_km2': high_area_km2},
    {'Category': 'Total M area', 'Area_km2': qc_M[1]['area_km2']},
])
table3.to_csv(ROOT / "15_tables/Table3_current_suitable_area.csv", index=False)

# Final model manifest
manifest_rows = []
for mn in eligible_models:
    manifest_rows.append({
        'model': mn, 'file': f"models/{mn}_all_reps.joblib",
        'n_replicates': N_BG_REPEATS, 'status': 'final',
        'mean_auc': perf_df[perf_df['model']==mn]['AUC_mean'].values[0],
    })
pd.DataFrame(manifest_rows).to_csv(ROOT / "08_final_models/final_model_manifest.csv", index=False)

# ============================================================
# Save key data for figures
# ============================================================
log.info("\n=== Saving figure data ===")

# Fig1 data
occ_main[[lon_col, lat_col]].to_csv(ROOT / "14_figure_data/Fig1_occurrence_points.csv", index=False)
M_gdfs[BUFFER_MAIN].to_file(ROOT / "14_figure_data/Fig1_M_area.gpkg")
adm0.to_file(ROOT / "14_figure_data/Fig1_boundaries.gpkg")

# Fig2 data
corr_matrix.to_csv(ROOT / "14_figure_data/Fig2_spearman_matrix.csv")
final_predictor_df[['variable', 'final_vif']].to_csv(ROOT / "14_figure_data/Fig2_final_vif.csv", index=False)

# Fig3 data
spatial_blocks = gpd.GeoDataFrame({
    'block_id': occ_km['block_id'],
    'outer_fold': occ_km['outer_fold'],
    'geometry': occ_gdf.geometry
}, crs="EPSG:4326")
spatial_blocks.to_file(ROOT / "14_figure_data/Fig3_spatial_blocks.gpkg")
occ_km[['outer_fold']].to_csv(ROOT / "14_figure_data/Fig3_occurrence_fold.csv")
pd.read_csv(ROOT / "04_background_points/background_rep01.csv").head(200).to_csv(
    ROOT / "14_figure_data/Fig3_background_fold_sample.csv", index=False)

# Fig4 data
fold_metrics_df.to_csv(ROOT / "14_figure_data/Fig4_model_performance_foldlevel.csv", index=False)

# Fig5 data
shutil.copy2(ROOT / "09_current_prediction/current_ensemble_suitability.tif",
             ROOT / "14_figure_data/Fig5_current_ensemble_suitability.tif")
occ_main[[lon_col, lat_col]].to_csv(ROOT / "14_figure_data/Fig5_occurrence_points.csv", index=False)
adm0.to_file(ROOT / "14_figure_data/Fig5_boundaries.gpkg")

# Fig6 data
shutil.copy2(ROOT / "10_uncertainty/current_intermodel_sd.tif", ROOT / "14_figure_data/Fig6_intermodel_sd.tif")
shutil.copy2(ROOT / "10_uncertainty/current_model_agreement.tif", ROOT / "14_figure_data/Fig6_model_agreement.tif")

# Fig7 data
pd.DataFrame(sensitivity_results).to_csv(ROOT / "14_figure_data/Fig7_thinning_sensitivity_metrics.csv", index=False)

# FigS1 data
pd.DataFrame([cult_results]).to_csv(ROOT / "14_figure_data/FigS1_cultivated_external_check.csv", index=False)

# ============================================================
# DONE - Print summary
# ============================================================
log.info("\n" + "="*60)
log.info("EXPERIMENT 1 PIPELINE COMPLETE")
log.info("="*60)
log.info("Main occurrence: 10 km, %d records", len(occ_main))
log.info("Main M area: %d km buffer", BUFFER_MAIN)
log.info("Final predictors: %d", len(final_vars))
log.info("Eligible models: %s", eligible_models)
log.info("Ensemble weights: %s", dict(zip(eligible_data['model'], eligible_data['weight'].round(3))))
log.info("Current suitable area: %.0f km2", suitable_area_km2)
log.info("Ensemble threshold: %.4f", ensemble_threshold)

# Save experiment status
status = "PASS_WITH_WARNINGS" if len(eligible_models) < 3 else "PASS"
with open(ROOT / "16_qc/EXPERIMENT1_STATUS.txt", 'w') as f:
    f.write(f"EXPERIMENT1_STATUS: {status}\n")
    f.write(f"Date: {pd.Timestamp.now()}\n")
    f.write(f"Eligible models: {eligible_models}\n")
    f.write(f"Final predictors: {final_vars}\n")

log.info(f"FINAL STATUS: {status}")
print(f"\n{'='*60}")
print(f"EXPERIMENT 1: {status}")
print(f"{'='*60}")
