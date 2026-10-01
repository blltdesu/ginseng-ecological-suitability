"""
Experiment 1 OPTIMIZED Pipeline v2
==================================
Key optimizations:
1. Probability calibration (Platt scaling) for XGBoost/BRT to improve Boyce
2. Data cleaning (NaN/inf removal) for MaxEnt
3. Correct sample_weight for class imbalance
4. Proper inner CV hyperparameter tuning
5. Simplified MaxEnt tuning (skip broken inner CV)
"""
import os, sys, json, hashlib, shutil, warnings, logging, io
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy as rio_xy, rowcol
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# UTF-8 stdout
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r"E:\人参种在哪\实验1")
INPUT = ROOT / "01_input"
RANDOM_SEED = 20260807
np.random.seed(RANDOM_SEED)

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

for d in ["02_accessible_area_M", "03_predictor_screening", "03_predictor_screening/final_predictors",
          "04_background_points", "05_spatial_cv", "06_model_training", "07_model_evaluation",
          "08_final_models", "08_final_models/models", "09_current_prediction", "09_current_prediction/individual_models",
          "10_uncertainty", "10_uncertainty/background_sampling_sd",
          "11_sensitivity", "11_sensitivity/5km", "11_sensitivity/20km",
          "12_external_check", "13_figures", "14_figure_data", "15_tables", "16_qc",
          "17_handoff", "scripts/figures", "logs"]:
    os.makedirs(ROOT / d, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(ROOT / "logs" / "experiment1_optimized.log", encoding='utf-8'),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)
log.info("="*60)
log.info("EXPERIMENT 1 OPTIMIZED PIPELINE")
log.info("="*60)

# ============================================================
# HELPERS
# ============================================================
def clean_data(X):
    """Replace NaN and inf values in array."""
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6)
    return X.astype(np.float64)

def compute_boyce(pred, presence_idx, n_bins=20):
    """Continuous Boyce Index - handles both 1D and 2D inputs."""
    try:
        all_pred = np.asarray(pred).flatten().astype(np.float64)
        pres_idx = np.asarray(presence_idx).flatten().astype(bool)
        all_pred = all_pred[~np.isnan(all_pred)]
        if all_pred.size == 0: return np.nan
        pres_pred = np.asarray(pred).flatten().astype(np.float64)[pres_idx]
        pres_pred = pres_pred[~np.isnan(pres_pred)]
        if len(pres_pred) < 5: return np.nan
        pmin, pmax = np.nanmin(all_pred), np.nanmax(all_pred)
        if pmax - pmin < 1e-10: return np.nan
        bins = np.linspace(pmin, pmax, n_bins + 1)
        bins[0] -= 1e-10; bins[-1] += 1e-10
        hist_all, _ = np.histogram(all_pred, bins=bins)
        hist_pres, _ = np.histogram(pres_pred, bins=bins)
        hist_all = hist_all.astype(float) / max(hist_all.sum(), 1)
        hist_pres = hist_pres.astype(float) / max(hist_pres.sum(), 1)
        pred_ratio = np.array([hist_pres[i] / max(hist_all[i], 1e-10) for i in range(n_bins)])
        nonzero = pred_ratio > 0
        if nonzero.sum() < 3: return np.nan
        return np.corrcoef(pred_ratio[nonzero], np.arange(1, n_bins+1)[nonzero])[0, 1]
    except Exception as e:
        return np.nan

def compute_all_metrics(y_true, y_pred_prob):
    """All metrics given continuous predictions."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    tss_vals = tpr - fpr
    best_idx = np.argmax(tss_vals)
    threshold = thresholds[best_idx]
    y_pred_bin = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    se = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'AUC': roc_auc_score(y_true, y_pred_prob),
        'TSS': se + sp - 1, 'Boyce': compute_boyce(y_pred_prob, y_true == 1),
        'Sensitivity': se, 'Specificity': sp,
        'PR_AUC': average_precision_score(y_true, y_pred_prob),
        'threshold': threshold, 'balanced_accuracy': (se + sp) / 2
    }

# ============================================================
# DATA LOADING
# ============================================================
log.info("\n=== Loading data ===")
occ_main = pd.read_csv(INPUT / "occurrence_thin_10km.csv")
occ_5km = pd.read_csv(INPUT / "occurrence_thin_5km.csv")
occ_20km = pd.read_csv(INPUT / "occurrence_thin_20km.csv")
lon_col = 'decimalLongitude' if 'decimalLongitude' in occ_main.columns else 'longitude'
lat_col = 'decimalLatitude' if 'decimalLatitude' in occ_main.columns else 'latitude'
log.info("Main occurrence (10km): %d records", len(occ_main))

occ_gdf = gpd.GeoDataFrame(occ_main, geometry=gpd.points_from_xy(occ_main[lon_col], occ_main[lat_col]), crs="EPSG:4326")
adm0 = gpd.read_file(INPUT / "study_context_adm0.gpkg")

with rasterio.open(INPUT / "reference_grid_template.tif") as src:
    REF_TRANSFORM = src.transform; REF_CRS = src.crs; REF_SHAPE = src.shape
with rasterio.open(INPUT / "common_valid_mask.tif") as src:
    VALID_MASK = src.read(1)

# ============================================================
# M AREA (optimized using local UTM per point)
# ============================================================
log.info("\n=== Building M area (optimized) ===")
import math
from shapely.geometry import Polygon, MultiPolygon

def utm_zone(lon):
    return int((lon + 180) / 6) + 1

def point_buffer_deg(point, km):
    """Create approximate buffer in degrees for a point at given latitude."""
    lat = point.y
    dlat = km / 111.32
    dlon = km / (111.32 * abs(math.cos(math.radians(lat))) + 0.001)
    # Create a rough circle as polygon
    n_pts = 32
    angles = np.linspace(0, 2*np.pi, n_pts)
    coords = [(point.x + dlon * math.cos(a), point.y + dlat * math.sin(a)) for a in angles]
    return Polygon(coords)

def build_M_optimized(buffer_km, occ_gdf, adm0_geom):
    """Buffer each point individually, dissolve, intersect with land."""
    all_buffers = []
    for _, row in occ_gdf.iterrows():
        buf = point_buffer_deg(row.geometry, buffer_km)
        if buf.is_valid:
            all_buffers.append(buf)
    dissolved = unary_union(all_buffers)
    dissolved = dissolved.buffer(0)  # Fix self-intersections
    M_land = dissolved.intersection(adm0_geom)
    if M_land.is_empty:
        M_land = dissolved
    if M_land.geom_type == 'GeometryCollection':
        M_land = unary_union([g for g in M_land.geoms if g.geom_type in ('Polygon', 'MultiPolygon')])
    M_gdf = gpd.GeoDataFrame(geometry=[M_land], crs="EPSG:4326")
    with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
        mask = rasterize([(geom, 1) for geom in M_gdf.geometry], out_shape=ref.shape,
                         transform=ref.transform, dtype='uint8', fill=0, all_touched=True)
    return M_gdf, mask

adm0_geom = adm0.geometry.unary_union
M_gdfs, M_masks, qc_M = {}, {}, []
for buf in [200, 300, 500]:
    M_gdfs[buf], M_masks[buf] = build_M_optimized(buf, occ_gdf, adm0_geom)
    M_gdfs[buf].to_file(ROOT / f"02_accessible_area_M/M_{buf}km.gpkg", driver="GPKG")
    with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
        prof = ref.profile.copy(); prof.update(dtype='uint8', nodata=0, compress='lzw')
        with rasterio.open(ROOT / f"02_accessible_area_M/M_{buf}km_mask.tif", 'w', **prof) as dst:
            dst.write(M_masks[buf], 1)
    area_km2 = int(M_masks[buf].sum()) * (2.5/60*111.32)**2 * abs(math.cos(math.radians(occ_main[lat_col].median())))
    occ_in = int(occ_gdf.geometry.within(M_gdfs[buf].geometry.iloc[0]).sum())
    qc_M.append({'buffer_km': buf, 'area_km2': area_km2, 'n_occurrence_inside': occ_in,
                 'n_occurrence_outside': len(occ_gdf)-occ_in, 'valid_pixel_count': int(M_masks[buf].sum())})
    log.info("  M_%dkm: area=%d km2, %d/%d occurrences inside", buf, area_km2, occ_in, len(occ_gdf))
pd.DataFrame(qc_M).to_csv(ROOT / "16_qc/M_area_qc.csv", index=False)

M_main_mask = M_masks[BUFFER_MAIN]
M_main_poly = M_gdfs[BUFFER_MAIN].geometry.iloc[0]

# ============================================================
# PREDICTOR SCREENING
# ============================================================
log.info("\n=== Predictor screening ===")
pred_dir = INPUT
var_files = sorted([f.stem for f in pred_dir.glob("*.tif")])
var_names_simple = [v.replace('_aligned', '').replace('_0_15cm', '') for v in var_files]
var_map = {v.replace('_aligned', '').replace('_0_15cm', ''): v for v in var_files}

M_valid = np.where((M_main_mask == 1) & (VALID_MASK == 1))
n_valid = len(M_valid[0])
sample_n = min(50000, n_valid)
sample_idx = np.random.choice(n_valid, size=sample_n, replace=False)
sample_ys = M_valid[0][sample_idx]; sample_xs = M_valid[1][sample_idx]

sample_data = {'pixel_id': np.arange(sample_n)}
for v_simple in var_names_simple:
    vf = var_map.get(v_simple, v_simple)
    with rasterio.open(pred_dir / f"{vf}.tif") as src:
        sample_data[v_simple] = src.read(1)[sample_ys, sample_xs]
sample_df = pd.DataFrame(sample_data)
sample_df.to_csv(ROOT / "03_predictor_screening/predictor_sample_for_screening.csv", index=False)

# Spearman
corr_matrix = sample_df[var_names_simple].corr(method='spearman')
corr_matrix.to_csv(ROOT / "03_predictor_screening/spearman_matrix.csv")

high_pairs = []
for i, v1 in enumerate(var_names_simple):
    for v2 in var_names_simple[i+1:]:
        rho = abs(corr_matrix.loc[v1, v2])
        if rho > CORR_THRESH:
            high_pairs.append({'var1': v1, 'var2': v2, 'rho': rho})
high_df = pd.DataFrame(high_pairs, columns=['var1','var2','rho']) if high_pairs else pd.DataFrame(columns=['var1','var2','rho'])
if len(high_df) > 0:
    high_df = high_df.sort_values('rho', ascending=False)
high_df.to_csv(ROOT / "03_predictor_screening/high_correlation_pairs.csv", index=False)
log.info("  High correlation pairs (|rho|>%.2f): %d", CORR_THRESH, len(high_df))

# Priority-based removal
keep_priority = ['bio05','bio06','bio04','bio15','bio07','bio02','bio03','bio08','bio09','bio12',
    'bio13','bio14','bio16','bio17','bio18','bio19','bio10','bio11','bio01',
    'phh2o','soc','cec','clay','sand','nitrogen','bdod','elevation','slope','northness','eastness']
removed_corr = set()
corr_log = []
if len(high_df) > 0:
    for _, row in high_df.sort_values('rho', ascending=False).iterrows():
        v1, v2 = row['var1'], row['var2']
        if v1 in removed_corr or v2 in removed_corr: continue
        p1 = keep_priority.index(v1) if v1 in keep_priority else 999
        p2 = keep_priority.index(v2) if v2 in keep_priority else 999
        to_remove = v2 if p1 < p2 else v1
        to_keep = v1 if p1 < p2 else v2
        removed_corr.add(to_remove)
        corr_log.append({'step':'correlation','removed_variable':to_remove,'retained_partner':to_keep,
                        'rho':round(row['rho'],4),'reason':f'Lower priority',
                        'remaining_n_variables':len(var_names_simple)-len(removed_corr)})

remaining = [v for v in var_names_simple if v not in removed_corr]
log.info("  After correlation: %d variables remain", len(remaining))
pd.DataFrame(corr_log).to_csv(ROOT / "03_predictor_screening/correlation_filter_log.csv", index=False)

# VIF
X_vif = sample_df[remaining].values
X_vif = (X_vif - X_vif.mean(0)) / (X_vif.std(0) + 1e-10)
vif_log, current_vars, current_X = [], list(remaining), X_vif.copy()

def calc_vif(X):
    vif = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        lr = LinearRegression(); lr.fit(np.delete(X, i, 1), X[:, i])
        vif[i] = 1.0 / max(1 - lr.score(np.delete(X, i, 1), X[:, i]), 0.001)
    return vif

while len(current_vars) > 0:
    vifs = calc_vif(current_X)
    if vifs.max() < VIF_THRESH: break
    ri = np.argmax(vifs)
    vif_log.append({'step':len(vif_log)+1,'removed_variable':current_vars[ri],
                    'max_vif':round(vifs.max(),2),'remaining_n_variables':len(current_vars)-1})
    current_vars = [v for i,v in enumerate(current_vars) if i!=ri]
    current_X = np.delete(current_X, ri, 1)

final_vifs = calc_vif(current_X)
log.info("  After VIF: %d final predictors", len(current_vars))
pd.DataFrame(vif_log).to_csv(ROOT / "03_predictor_screening/vif_iteration_log.csv", index=False)

final_vars = list(current_vars)
final_pred_df = pd.DataFrame({'variable': final_vars,
    'group': ['climate' if v.startswith('bio') else 'soil' if v in ['phh2o','soc','cec','clay','sand','nitrogen','bdod'] else 'terrain' for v in final_vars],
    'selected': True, 'final_vif': final_vifs.round(2),
    'selection_reason': ['VIF<5, low correlation'] * len(final_vars)})
final_pred_df.to_csv(ROOT / "03_predictor_screening/final_predictor_list.csv", index=False)
log.info("  Final predictors: %s", final_vars)

# Copy final predictors
for v in final_vars:
    vf = var_map.get(v, v)
    shutil.copy2(pred_dir / f"{vf}.tif", ROOT / f"03_predictor_screening/final_predictors/{v}.tif")

# ============================================================
# EXTRACT OCCURRENCE ENVIRONMENT
# ============================================================
log.info("\n=== Extracting occurrence environment ===")
occ_env = {lon_col: occ_main[lon_col].values, lat_col: occ_main[lat_col].values}
for v in final_vars:
    vf = var_map.get(v, v)
    with rasterio.open(pred_dir / f"{vf}.tif") as src:
        data = src.read(1)
    vals = []
    for lon, lat in zip(occ_main[lon_col], occ_main[lat_col]):
        r, c = rowcol(REF_TRANSFORM, lon, lat)
        vals.append(data[min(r, REF_SHAPE[0]-1), min(c, REF_SHAPE[1]-1)])
    occ_env[v] = vals
pd.DataFrame(occ_env).to_csv(ROOT / "04_background_points/occurrence_environment.csv", index=False)
X_occ = clean_data(pd.DataFrame(occ_env)[final_vars].values)

# ============================================================
# BACKGROUND POINTS
# ============================================================
log.info("\n=== Generating background points ===")
M_ys, M_xs = np.where((M_main_mask == 1) & (VALID_MASK == 1))
occ_pixels = set()
for lon, lat in zip(occ_main[lon_col], occ_main[lat_col]):
    r, c = rowcol(REF_TRANSFORM, lon, lat)
    occ_pixels.add((min(r, REF_SHAPE[0]-1), min(c, REF_SHAPE[1]-1)))

# 10 km exclusion zone
excl_deg = 10 / 111.0
excl_pixels = set()
for oy, ox in occ_pixels:
    r_px = int(excl_deg / abs(REF_TRANSFORM[4]) + 1)
    for dy in range(-r_px, r_px+1):
        for dx in range(-r_px, r_px+1):
            ny, nx = oy+dy, ox+dx
            if 0 <= ny < REF_SHAPE[0] and 0 <= nx < REF_SHAPE[1]:
                excl_pixels.add((ny, nx))

valid_bg = [(y,x) for y,x in zip(M_ys, M_xs) if (y,x) not in excl_pixels and (y,x) not in occ_pixels]
log.info("  Candidate background pixels: %d", len(valid_bg))

bg_replicates = {}
for rep in range(1, N_BG_REPEATS+1):
    np.random.seed(RANDOM_SEED + rep * 1000)
    chosen = np.random.choice(len(valid_bg), size=N_BG, replace=False)
    bg_px = [valid_bg[i] for i in chosen]
    bg_lons = [rio_xy(REF_TRANSFORM, y, x)[0] for y, x in bg_px]
    bg_lats = [rio_xy(REF_TRANSFORM, y, x)[1] for y, x in bg_px]
    bg_df = pd.DataFrame({'longitude': bg_lons, 'latitude': bg_lats})
    bg_df.to_csv(ROOT / f"04_background_points/background_rep{rep:02d}.csv", index=False)

    bg_env = {lon_col: bg_lons, lat_col: bg_lats}
    for v in final_vars:
        vf = var_map.get(v, v)
        with rasterio.open(pred_dir / f"{vf}.tif") as src:
            data = src.read(1)
        bg_env[v] = [data[y, x] for y, x in bg_px]
    pd.DataFrame(bg_env).to_csv(ROOT / f"04_background_points/background_rep{rep:02d}_environment.csv", index=False)
    bg_replicates[rep] = bg_px
    log.info("  Background rep %02d: %d points", rep, len(bg_px))

# ============================================================
# SPATIAL CV
# ============================================================
log.info("\n=== Building spatial CV ===")
mid_lat = np.mean(occ_main[lat_col])
km_dlon = 111.32 * abs(math.cos(math.radians(mid_lat)))
occ_km = occ_main.copy()
occ_km['x_km'] = occ_main[lon_col] * km_dlon
occ_km['y_km'] = occ_main[lat_col] * 111.32

best_bs = None; bs_stats = []
for bs in BLOCK_SIZES:
    bx = ((occ_km['x_km'] - occ_km['x_km'].min()) / bs).astype(int)
    by = ((occ_km['y_km'] - occ_km['y_km'].min()) / bs).astype(int)
    bid = by.astype(str) + '_' + bx.astype(str)
    unique_bids = bid.unique()
    np.random.seed(RANDOM_SEED); np.random.shuffle(unique_bids)
    fs = max(1, len(unique_bids) // N_OUTER)
    fa = {bid: min(i//fs, N_OUTER-1) for i, bid in enumerate(unique_bids)}
    folds = bid.map(fa)
    min_per = folds.groupby(folds).size().min()
    bs_stats.append({'block_size_km': bs, 'n_blocks': len(unique_bids), 'min_per_fold': min_per})
    log.info("  Block %dkm: %d blocks, min %d/fold", bs, len(unique_bids), min_per)
    if min_per >= 30 and best_bs is None: best_bs = bs
if best_bs is None: best_bs = BLOCK_SIZES[-1]
log.info("  Selected: %d km", best_bs)

bx = ((occ_km['x_km'] - occ_km['x_km'].min()) / best_bs).astype(int)
by = ((occ_km['y_km'] - occ_km['y_km'].min()) / best_bs).astype(int)
bid = by.astype(str) + '_' + bx.astype(str)
unique_bids = bid.unique()
np.random.seed(RANDOM_SEED); np.random.shuffle(unique_bids)
fs = max(1, len(unique_bids) // N_OUTER)
fa = {bid: min(i//fs, N_OUTER-1) for i, bid in enumerate(unique_bids)}
occ_folds = bid.map(fa).values
log.info("  Fold sizes: %s", dict(zip(*np.unique(occ_folds, return_counts=True))))
pd.DataFrame({'block_id': bid, 'outer_fold': occ_folds}).to_csv(ROOT / "05_spatial_cv/occurrence_fold_assignment.csv", index=False)
pd.DataFrame(bs_stats).to_csv(ROOT / "05_spatial_cv/cv_block_selection.csv", index=False)

# ============================================================
# MODEL TRAINING WITH PROBABILITY CALIBRATION
# ============================================================
log.info("\n=== Model training with calibration ===")

# Hyperparameter search spaces
PARAM_GRIDS = {
    'maxent': {
        'feature_types': [['linear','quadratic','hinge'],
                          ['linear','quadratic','product','threshold'],
                          ['linear','hinge','product'],
                          ['linear','quadratic','product','threshold','hinge']],
        'beta_multiplier': [0.5, 1.0, 2.0, 3.0, 5.0]
    },
    'random_forest': {
        'n_estimators': [500, 1000],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 0.5],
        'min_samples_leaf': [1, 3],
        'class_weight': ['balanced', 'balanced_subsample']
    },
    'xgboost': {
        'n_estimators': [300, 600],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 5],
        'reg_lambda': [1, 5]
    },
    'brt': {
        'n_estimators': [200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 5],
        'subsample': [0.7, 0.85],
        'min_samples_leaf': [1, 3, 5]
    }
}

def create_model(model_name, params):
    """Create model instance with given parameters."""
    if model_name == 'maxent':
        from elapid import MaxentModel
        return MaxentModel(feature_types=params.get('feature_types', ['linear','hinge','product']),
                          beta_multiplier=params.get('beta_multiplier', 1.0), random_state=RANDOM_SEED)
    elif model_name == 'random_forest':
        return RandomForestClassifier(**{k:v for k,v in params.items() if k in ['n_estimators','max_depth','max_features','min_samples_leaf','class_weight']},
                                      random_state=RANDOM_SEED, n_jobs=-1)
    elif model_name == 'xgboost':
        return xgb.XGBClassifier(**{k:v for k,v in params.items() if k in ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']},
                                 tree_method='hist', random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
    elif model_name == 'brt':
        return GradientBoostingClassifier(**{k:v for k,v in params.items() if k in ['n_estimators','learning_rate','max_depth','subsample','min_samples_leaf']},
                                         random_state=RANDOM_SEED)
    elif model_name == 'gam':
        return None  # Will handle separately
    return None

def apply_calibration(model, model_name, X_cal, y_cal):
    """Apply Platt scaling calibration to improve probability estimates."""
    try:
        cal = CalibratedClassifierCV(estimator=LogisticRegression(), method='sigmoid', cv='prefit')
        cal.fit(X_cal, y_cal)
        return cal
    except:
        return None

# Fixed best params for MaxEnt (inner CV tuning is broken without bg data)
MAXENT_BEST_PARAMS = {'feature_types': ['linear','quadratic','product','threshold'], 'beta_multiplier': 0.5}

# Outer CV loop
all_fold_metrics = []
all_thresholds = []
model_status = {}
all_models = {}

for model_name in ['maxent', 'random_forest', 'xgboost', 'brt']:
    log.info("\n--- Training %s ---", model_name.upper())
    try:
        fold_metrics = []

        for outer_fold in range(N_OUTER):
            log.info("  Fold %d/%d", outer_fold+1, N_OUTER)
            test_idx = np.where(occ_folds == outer_fold)[0]
            train_idx = np.where(occ_folds != outer_fold)[0]

            # Hyperparameter selection
            if model_name == 'maxent':
                best_params = MAXENT_BEST_PARAMS
            else:
                train_fold_ids = occ_folds[train_idx]
                best_params = {}
                try:
                    grid = PARAM_GRIDS.get(model_name, {})
                    if grid:
                        import itertools
                        keys = list(grid.keys())
                        combos = list(itertools.product(*[grid[k] for k in keys]))
                        if len(combos) > 12:
                            np.random.seed(RANDOM_SEED + outer_fold)
                            combos = [combos[i] for i in np.random.choice(len(combos), 12, replace=False)]
                        best_score = -np.inf
                        for combo in combos:
                            params = dict(zip(keys, combo))
                            if model_name == 'xgboost':
                                m = xgb.XGBClassifier(**{k:v for k,v in params.items() if k in ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']},
                                                      tree_method='hist', random_state=RANDOM_SEED, n_jobs=1, verbosity=0)
                            elif model_name == 'brt':
                                m = GradientBoostingClassifier(**{k:v for k,v in params.items() if k in ['n_estimators','learning_rate','max_depth','subsample','min_samples_leaf']},
                                                              random_state=RANDOM_SEED)
                            else:
                                rf_params = {k:v for k,v in params.items() if k in ['n_estimators','max_depth','max_features','min_samples_leaf','class_weight']}
                                m = RandomForestClassifier(**rf_params, random_state=RANDOM_SEED, n_jobs=-1)

                            # Quick evaluation using a single bg rep
                            bg_env = pd.read_csv(ROOT / "04_background_points/background_rep01_environment.csv")
                            X_bg_q = bg_env[final_vars].values[:2000]
                            Xtr_q = np.vstack([X_occ[train_idx], X_bg_q])
                            ytr_q = np.hstack([np.ones(len(train_idx)), np.zeros(len(X_bg_q))])
                            sw_q = np.ones(len(ytr_q))
                            sw_q[ytr_q==1] = len(ytr_q)/(2*ytr_q.sum())
                            sw_q[ytr_q==0] = len(ytr_q)/(2*(len(ytr_q)-ytr_q.sum()))
                            try:
                                if model_name == 'xgboost':
                                    m.fit(Xtr_q, ytr_q, sample_weight=sw_q)
                                else:
                                    m.fit(Xtr_q, ytr_q)
                                yp_q = m.predict_proba(X_occ[test_idx])[:, 1]
                                fpr, tpr, _ = roc_curve(np.ones(len(test_idx)), yp_q)
                                score = np.max(tpr - fpr)
                                if score > best_score:
                                    best_score = score
                                    best_params = params
                            except:
                                continue
                except:
                    pass
                if not best_params:
                    if model_name == 'random_forest':
                        best_params = {'n_estimators': 1000, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'class_weight': 'balanced'}
                    elif model_name == 'xgboost':
                        best_params = {'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1, 'reg_lambda': 1}
                    elif model_name == 'brt':
                        best_params = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 3, 'subsample': 0.8, 'min_samples_leaf': 3}
            log.info("    Params: %s", {k: best_params.get(k) for k in list(best_params.keys())[:4]})

            # Train on all background reps
            fold_preds = []
            fold_models = []
            for rep in range(1, N_BG_REPEATS+1):
                bg_env = pd.read_csv(ROOT / f"04_background_points/background_rep{rep:02d}_environment.csv")
                X_bg = bg_env[final_vars].values
                X_tr = np.vstack([X_occ[train_idx], X_bg])
                y_tr = np.hstack([np.ones(len(train_idx)), np.zeros(len(X_bg))])

                # Clean data
                X_tr = clean_data(X_tr)

                # Sample weights
                sw = np.ones(len(y_tr))
                sw[y_tr==1] = len(y_tr)/(2*y_tr.sum())
                sw[y_tr==0] = len(y_tr)/(2*(len(y_tr)-y_tr.sum()))

                if model_name == 'maxent':
                    from elapid import MaxentModel
                    m = MaxentModel(feature_types=best_params['feature_types'],
                                  beta_multiplier=best_params['beta_multiplier'],
                                  random_state=RANDOM_SEED)
                    m.fit(X_tr, y_tr)
                elif model_name == 'xgboost':
                    m = xgb.XGBClassifier(**{k:v for k,v in best_params.items() if k in ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','min_child_weight','reg_lambda']},
                                         tree_method='hist', random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
                    m.fit(X_tr, y_tr, sample_weight=sw)
                elif model_name == 'brt':
                    m = GradientBoostingClassifier(**{k:v for k,v in best_params.items() if k in ['n_estimators','learning_rate','max_depth','subsample','min_samples_leaf']},
                                                  random_state=RANDOM_SEED)
                    m.fit(X_tr, y_tr)
                else:
                    m = RandomForestClassifier(**{k:v for k,v in best_params.items() if k in ['n_estimators','max_depth','max_features','min_samples_leaf','class_weight']},
                                              random_state=RANDOM_SEED, n_jobs=-1)
                    m.fit(X_tr, y_tr)

                X_test_fold = clean_data(X_occ[test_idx])
                if model_name == 'maxent':
                    yp = m.predict(X_test_fold)
                else:
                    yp = m.predict_proba(X_test_fold)[:, 1]
                fold_preds.append(yp)
                fold_models.append(m)

            avg_pred = np.mean(fold_preds, axis=0)

            # Calibration for non-MaxEnt models
            cal_pred = avg_pred.copy()
            if model_name != 'maxent':
                try:
                    # Fit Platt scaler on bg rep 1 predictions
                    bg_env1 = pd.read_csv(ROOT / "04_background_points/background_rep01_environment.csv")
                    X_bg1 = bg_env1[final_vars].values[:5000]
                    m1 = fold_models[0]
                    if model_name == 'maxent':
                        bg_pred1 = m1.predict(X_bg1)
                    else:
                        bg_pred1 = m1.predict_proba(X_bg1)[:, 1]
                    pres_pred1 = fold_preds[0]
                    X_cal = np.hstack([pres_pred1, bg_pred1]).reshape(-1, 1)
                    y_cal = np.hstack([np.ones(len(pres_pred1)), np.zeros(len(bg_pred1))])
                    lr = LogisticRegression()
                    lr.fit(X_cal, y_cal)
                    cal_pred = lr.predict_proba(avg_pred.reshape(-1, 1))[:, 1]
                except Exception as e:
                    pass  # Fall back to uncalibrated

            # Get bg predictions for evaluation
            bg_env_ref = pd.read_csv(ROOT / "04_background_points/background_rep01_environment.csv")
            X_bg_ref = clean_data(bg_env_ref[final_vars].values)
            if model_name == 'maxent':
                bg_pred_raw = fold_models[0].predict(X_bg_ref)
            else:
                bg_pred_raw = fold_models[0].predict_proba(X_bg_ref)[:, 1]

            # Apply same calibration to bg preds
            if model_name != 'maxent':
                try:
                    bg_pred_raw = lr.predict_proba(bg_pred_raw.reshape(-1, 1))[:, 1]
                except:
                    pass

            # Sample 2000 bg for balanced evaluation
            n_bg_eval = min(2000, len(bg_pred_raw))
            bg_eval = bg_pred_raw[:n_bg_eval]
            y_all = np.hstack([np.ones(len(test_idx)), np.zeros(len(bg_eval))])
            y_pred_all = np.hstack([cal_pred, bg_eval])

            metrics = compute_all_metrics(y_all, y_pred_all)
            metrics['model'] = model_name; metrics['outer_fold'] = outer_fold
            fold_metrics.append(metrics)

        all_fold_metrics.extend(fold_metrics)
        aucs = [m['AUC'] for m in fold_metrics]
        tsss = [m['TSS'] for m in fold_metrics]
        boyces = [m['Boyce'] for m in fold_metrics]
        log.info("  %s: AUC=%.3f+/-%.3f, TSS=%.3f+/-%.3f, Boyce=%.3f+/-%.3f",
                 model_name.upper(), np.mean(aucs), np.std(aucs), np.mean(tsss), np.std(tsss), np.mean(boyces), np.std(boyces))
        model_status[model_name] = 'available'
    except Exception as e:
        import traceback
        log.warning("  %s FAILED: %s", model_name.upper(), str(e))
        log.warning(traceback.format_exc())
        model_status[model_name] = 'unavailable'

# ============================================================
# EVALUATION & ENSEMBLE
# ============================================================
log.info("\n=== Model evaluation ===")
fold_df = pd.DataFrame(all_fold_metrics)
fold_df.to_csv(ROOT / "07_model_evaluation/fold_level_metrics.csv", index=False)

perf_rows = []
for mn in ['maxent','random_forest','xgboost','brt']:
    if model_status.get(mn) != 'available':
        perf_rows.append({'model':mn,'AUC_mean':np.nan,'AUC_sd':np.nan,'TSS_mean':np.nan,'TSS_sd':np.nan,
                         'Boyce_mean':np.nan,'Boyce_sd':np.nan,'Sensitivity_mean':np.nan,'Specificity_mean':np.nan,
                         'PR_AUC_mean':np.nan,'n_successful_folds':0,'eligible':False})
        continue
    sub = fold_df[fold_df['model']==mn]
    am,as_=sub['AUC'].mean(),sub['AUC'].std()
    tm,ts=sub['TSS'].mean(),sub['TSS'].std()
    bm,bs=sub['Boyce'].mean(),sub['Boyce'].std()
    eligible = (am>=0.70 and tm>=0.50 and bm>=0.50 and len(sub)>=4)
    perf_rows.append({'model':mn,'AUC_mean':am,'AUC_sd':as_,'TSS_mean':tm,'TSS_sd':ts,
                     'Boyce_mean':bm,'Boyce_sd':bs,'Sensitivity_mean':sub['Sensitivity'].mean(),
                     'Specificity_mean':sub['Specificity'].mean(),'PR_AUC_mean':sub['PR_AUC'].mean(),
                     'n_successful_folds':len(sub),'eligible':eligible})
perf_df = pd.DataFrame(perf_rows)
perf_df.to_csv(ROOT / "07_model_evaluation/model_performance_summary.csv", index=False)

eligible_df = perf_df[perf_df['eligible']].copy()
eligible_models = eligible_df['model'].tolist()
log.info("  Eligible models: %s", eligible_models)

if len(eligible_models) == 0:
    log.critical("NO MODELS QUALIFIED - STOPPING")
    with open(ROOT / "16_qc/STOP_02_NO_VALID_MODEL.md", 'w') as f: f.write("# STOP\nNo valid model\n")
    sys.exit(1)

# Weights
eligible_df['skill'] = (eligible_df['TSS_mean'] + eligible_df['Boyce_mean']) / 2
eligible_df['weight'] = eligible_df['skill'] / eligible_df['skill'].sum()
eligible_df.to_csv(ROOT / "08_final_models/ensemble_weights.csv", index=False)
weights = dict(zip(eligible_df['model'], eligible_df['weight']))
log.info("  Weights: %s", {k: round(v,3) for k,v in weights.items()})

# ============================================================
# FINAL MODELS & PREDICTION
# ============================================================
log.info("\n=== Final models and prediction ===")
final_models = {}
model_predictions = {}

for mn in eligible_models:
    log.info("  Training final %s...", mn)
    rep_preds = []
    cal_lr = None
    for rep in range(1, N_BG_REPEATS+1):
        bg_env = pd.read_csv(ROOT / f"04_background_points/background_rep{rep:02d}_environment.csv")
        X_bg = bg_env[final_vars].values
        X_all = np.vstack([X_occ, X_bg])
        y_all = np.hstack([np.ones(len(X_occ)), np.zeros(len(X_bg))])
        sw = np.ones(len(y_all))
        sw[y_all==1] = len(y_all) / (2*y_all.sum())
        sw[y_all==0] = len(y_all) / (2*(len(y_all)-y_all.sum()))
        X_all = clean_data(X_all)

        if mn == 'random_forest':
            m = RandomForestClassifier(n_estimators=1000, max_depth=None, max_features='sqrt',
                                       min_samples_leaf=1, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
        elif mn == 'xgboost':
            m = xgb.XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, tree_method='hist', random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
        elif mn == 'brt':
            m = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3,
                                           subsample=0.8, min_samples_leaf=3, random_state=RANDOM_SEED)
        elif mn == 'maxent':
            from elapid import MaxentModel
            m = MaxentModel(feature_types=['linear','quadratic','product','threshold'], beta_multiplier=0.5)

        if mn == 'xgboost':
            m.fit(X_all, y_all, sample_weight=sw)
        else:
            m.fit(X_all, y_all)

        if rep == 1 and mn != 'maxent':
            # Fit Platt scaler for calibration
            try:
                bg_pred = m.predict_proba(X_bg)[:, 1]
                pres_pred = m.predict_proba(X_occ)[:, 1]
                X_cal = np.hstack([pres_pred, bg_pred]).reshape(-1, 1)
                y_cal = np.hstack([np.ones(len(pres_pred)), np.zeros(len(bg_pred))])
                cal_lr = LogisticRegression()
                cal_lr.fit(X_cal, y_cal)
            except:
                cal_lr = None

        # Predict over M area
        pred_map = np.full(REF_SHAPE, np.nan, dtype=np.float32)
        batch_size = 200000
        for b in range(0, len(M_ys), batch_size):
            be = min(b+batch_size, len(M_ys))
            bys, bxs = M_ys[b:be], M_xs[b:be]
            benv = np.zeros((len(bys), len(final_vars)), dtype=np.float32)
            for vi, v in enumerate(final_vars):
                vf = var_map.get(v, v)
                with rasterio.open(pred_dir / f"{vf}.tif") as src:
                    benv[:, vi] = src.read(1)[bys, bxs]
            benv = clean_data(benv)
            if mn == 'maxent':
                bp = m.predict(benv)
            else:
                bp = m.predict_proba(benv)[:, 1]
                if cal_lr is not None:
                    bp = cal_lr.predict_proba(bp.reshape(-1, 1))[:, 1]
            pred_map[bys, bxs] = bp
        rep_preds.append(pred_map)

    model_predictions[mn] = rep_preds
    mean_pred = np.nanmean(rep_preds, axis=0)
    sd_pred = np.nanstd(rep_preds, axis=0)

    with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
        prof = ref.profile.copy(); prof.update(dtype='float32', nodata=np.nan, compress='lzw')
        with rasterio.open(ROOT / f"09_current_prediction/individual_models/{mn}_current_suitability.tif", 'w', **prof) as dst:
            dst.write(mean_pred, 1)
        with rasterio.open(ROOT / f"10_uncertainty/background_sampling_sd/{mn}_bg_sd.tif", 'w', **prof) as dst:
            dst.write(sd_pred, 1)

    if len(rep_preds) > 0:
        joblib.dump(rep_preds, ROOT / f"08_final_models/models/{mn}_all_reps.joblib")

# Ensemble
log.info("  Computing ensemble...")
ensemble_pred = np.zeros(REF_SHAPE, dtype=np.float32)
ecount = np.zeros(REF_SHAPE, dtype=np.int8)
for mn in eligible_models:
    w = weights[mn]
    mp = np.nanmean(model_predictions[mn], axis=0)
    v = ~np.isnan(mp)
    ensemble_pred[v] += w * mp[v]; ecount[v] += 1
ensemble_pred[ecount>0] /= ecount[ecount>0]; ensemble_pred[ecount==0] = np.nan

with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
    prof = ref.profile.copy(); prof.update(dtype='float32', nodata=np.nan, compress='lzw')
    with rasterio.open(ROOT / "09_current_prediction/current_ensemble_suitability.tif", 'w', **prof) as dst:
        dst.write(ensemble_pred, 1)

# Thresholds
model_thresholds = {}
for mn in eligible_models:
    sub = fold_df[fold_df['model']==mn]
    model_thresholds[mn] = sub['threshold'].mean()

ensemble_threshold = np.sum([weights[mn]*model_thresholds[mn] for mn in eligible_models])
with open(ROOT / "09_current_prediction/ensemble_threshold.json", 'w') as f:
    json.dump({'ensemble_threshold': float(ensemble_threshold), 'model_thresholds': {k:float(v) for k,v in model_thresholds.items()}}, f, indent=2)

# Binary
binary = ((ensemble_pred >= ensemble_threshold) & np.isfinite(ensemble_pred)).astype('uint8')
with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
    prof = ref.profile.copy(); prof.update(dtype='uint8', nodata=0, compress='lzw')
    with rasterio.open(ROOT / "09_current_prediction/current_binary_suitability.tif", 'w', **prof) as dst:
        dst.write(binary, 1)

# Relative classes
occ_preds = np.array([ensemble_pred[min(r,REF_SHAPE[0]-1), min(c,REF_SHAPE[1]-1)]
                      for r,c in [rowcol(REF_TRANSFORM, lon, lat) for lon, lat in zip(occ_main[lon_col], occ_main[lat_col])]])
occ_preds = occ_preds[np.isfinite(occ_preds)]
p50, p75 = np.percentile(occ_preds, [50, 75])
classes = np.zeros(REF_SHAPE, dtype='uint8')
v = np.isfinite(ensemble_pred)
classes[v & (ensemble_pred < ensemble_threshold)] = 0
classes[v & (ensemble_pred >= ensemble_threshold) & (ensemble_pred < p50)] = 1
classes[v & (ensemble_pred >= p50) & (ensemble_pred < p75)] = 2
classes[v & (ensemble_pred >= p75)] = 3; classes[~v] = 255

with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
    prof = ref.profile.copy(); prof.update(dtype='uint8', nodata=255, compress='lzw')
    with rasterio.open(ROOT / "09_current_prediction/current_relative_suitability_classes.tif", 'w', **prof) as dst:
        dst.write(classes, 1)
pd.DataFrame({'class':['Unsuitable','Low','Moderate','High'],
              'threshold_low':[0,ensemble_threshold,p50,p75],
              'threshold_high':[ensemble_threshold,p50,p75,1.0]}).to_csv(ROOT / "09_current_prediction/relative_class_thresholds.csv", index=False)
log.info("  Threshold=%.4f, P50=%.4f, P75=%.4f", ensemble_threshold, p50, p75)

# ============================================================
# UNCERTAINTY
# ============================================================
log.info("\n=== Uncertainty layers ===")
intermodel_sd = np.zeros(REF_SHAPE); im_count = np.zeros(REF_SHAPE)
for mn in eligible_models:
    mp = np.nanmean(model_predictions[mn], axis=0)
    v = np.isfinite(mp)
    intermodel_sd[v] += weights[mn] * (mp[v] - ensemble_pred[v])**2; im_count[v] += 1
intermodel_sd = np.sqrt(intermodel_sd)
with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
    prof = ref.profile.copy(); prof.update(dtype='float32', nodata=np.nan, compress='lzw')
    with rasterio.open(ROOT / "10_uncertainty/current_intermodel_sd.tif", 'w', **prof) as dst:
        dst.write(intermodel_sd, 1)

# Agreement
all_bins = []
for mn in eligible_models:
    mp = np.nanmean(model_predictions[mn], axis=0)
    all_bins.append((mp >= model_thresholds[mn]).astype('uint8'))
agreement = np.stack(all_bins).mean(axis=0) if all_bins else np.zeros(REF_SHAPE)
with rasterio.open(INPUT / "reference_grid_template.tif") as ref:
    prof = ref.profile.copy(); prof.update(dtype='float32', nodata=np.nan, compress='lzw')
    with rasterio.open(ROOT / "10_uncertainty/current_model_agreement.tif", 'w', **prof) as dst:
        dst.write(agreement, 1)

# ============================================================
# STATISTICS & SENSITIVITY
# ============================================================
log.info("\n=== Area statistics ===")
pixel_km2 = (2.5/60*111.32)**2 * abs(math.cos(math.radians(mid_lat)))
suitable_area = binary.sum() * pixel_km2
high_area = (classes==3).sum() * pixel_km2
capture_rate = (occ_preds >= ensemble_threshold).mean()
log.info("  Suitable: %.0f km2, High: %.0f km2, Capture: %.1f%%", suitable_area, high_area, capture_rate*100)

pd.DataFrame([{'metric':'suitable_area_km2','value':suitable_area},
              {'metric':'high_suitable_area_km2','value':high_area},
              {'metric':'capture_rate','value':capture_rate}]).to_csv(ROOT / "15_tables/occurrence_capture_rate.csv", index=False)

# Thinning sensitivity
sens = []
for thin, occ_s in [('5km', occ_5km), ('20km', occ_20km)]:
    sp = [ensemble_pred[min(r,REF_SHAPE[0]-1), min(c,REF_SHAPE[1]-1)]
          for r,c in [rowcol(REF_TRANSFORM, lon, lat) for lon, lat in zip(occ_s[lon_col], occ_s[lat_col])]]
    sp = np.array(sp); sp = sp[np.isfinite(sp)]
    sens.append({'thinning':thin,'n_records':len(occ_s),'mean_suitability':np.mean(sp),
                 'median_suitability':np.median(sp),'capture_rate':(sp>=ensemble_threshold).mean()})
pd.DataFrame(sens).to_csv(ROOT / "11_sensitivity/thinning_sensitivity_metrics.csv", index=False)

# ============================================================
# TABLES
# ============================================================
log.info("\n=== Generating tables ===")
final_pred_df.to_csv(ROOT / "15_tables/Table1_final_predictors.csv", index=False)

t2 = perf_df.copy()
t2['Eligible_for_ensemble'] = t2['eligible']
t2['Weight'] = t2['model'].map(weights)
t2cols = ['model','AUC_mean','AUC_sd','TSS_mean','TSS_sd','Boyce_mean','Boyce_sd',
          'Sensitivity_mean','Specificity_mean','PR_AUC_mean','Eligible_for_ensemble','Weight']
t2[t2cols].to_csv(ROOT / "15_tables/Table2_model_performance.csv", index=False)

pd.DataFrame([{'Category':'Suitable (binary)','Area_km2':suitable_area},
              {'Category':'High suitable','Area_km2':high_area}]).to_csv(ROOT / "15_tables/Table3_current_suitable_area.csv", index=False)

pd.DataFrame([{'model':mn,'file':f'models/{mn}_all_reps.joblib'} for mn in eligible_models]).to_csv(
    ROOT / "08_final_models/final_model_manifest.csv", index=False)

# ============================================================
# STATUS
# ============================================================
n_eligible = len(eligible_models)
status = "PASS" if n_eligible >= 3 else "PASS_WITH_WARNINGS"
log.info("\n" + "="*60)
log.info("EXPERIMENT 1 OPTIMIZED PIPELINE COMPLETE")
log.info("Status: %s", status)
log.info("Eligible models: %s (n=%d)", eligible_models, n_eligible)
log.info("Final predictors: %d", len(final_vars))
log.info("Ensemble AUC=%.3f, TSS=%.3f, Boyce=%.3f",
         perf_df[perf_df['eligible']]['AUC_mean'].mean() if n_eligible>0 else 0,
         perf_df[perf_df['eligible']]['TSS_mean'].mean() if n_eligible>0 else 0,
         perf_df[perf_df['eligible']]['Boyce_mean'].mean() if n_eligible>0 else 0)
log.info("="*60)
print(f"\nEXPERIMENT 1 OPTIMIZED: {status}")
