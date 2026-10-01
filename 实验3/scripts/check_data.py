"""Quick data check script - verify model loading and fold mapping."""
import joblib
import pandas as pd
import numpy as np
import os

# Load all 4 calibrated models
models_dir = r'E:\人参种在哪\实验2\02_analysis_dataset'
for name in ['maxent', 'random_forest', 'xgboost', 'brt']:
    path = os.path.join(models_dir, f'{name}_calibrated_model.joblib')
    try:
        m = joblib.load(path)
        print(f'{name}: {type(m).__name__}, n_features={m.n_features_in_}, classes={m.classes_}')
        if hasattr(m, 'estimator'):
            print(f'  estimator: {type(m.estimator).__name__}')
    except Exception as e:
        print(f'{name}: ERROR - {e}')

# Check training matrix vs fold assignment
tm = pd.read_csv(r'E:\人参种在哪\实验3\00_input_from_experiment2\training_matrix_main.csv')
fa = pd.read_csv(r'E:\人参种在哪\实验3\00_input_from_experiment2\spatial_cv_fold_assignment.csv')
print(f'\nTraining matrix: {tm.shape}')
print(f'Fold assignment: {fa.shape}')
print(f'Presence rows: {(tm.sample_type=="presence").sum()}')
print(f'Background rows: {(tm.sample_type=="background").sum()}')
print(f'Folds in training matrix: {tm.outer_fold.unique()}')
print(f'Folds in assignment: {sorted(fa.outer_fold.unique())}')

# The fold assignment has 252 rows (same as presence count)
# The training matrix has 252 presence rows in order
# Check if they correspond 1:1
presence = tm[tm.sample_type == 'presence'].copy()
print(f'First 3 presence sample_ids: {presence.sample_id.values[:3]}')
print(f'First 3 fold assignment block_ids: {fa.block_id.values[:3]}')

# Assume presence rows map 1:1 to fold assignment rows (both have 252 rows)
# Merge by index
presence_idx = tm[tm.sample_type == 'presence'].index
tm_with_folds = tm.copy()
tm_with_folds.loc[presence_idx, 'outer_fold'] = fa['outer_fold'].values

# For background points, they need fold assignment too
# Each background replicate should use the same CV fold structure
# Check how experiment 2 handled this
print(f'\nBackground replicates: {sorted(tm.background_replicate.unique())}')

# Check if there are trained models from experiment 1
exp1_models = r'E:\人参种在哪\实验2\00_input_from_experiment1\models'
for f in os.listdir(exp1_models):
    print(f'Experiment 1 model: {f}')

# Check the trained (pre-calibration) models in experiment 2
exp2_trained = r'E:\人参种在哪\实验2\02_analysis_dataset'
for f in os.listdir(exp2_trained):
    if 'trained' in f:
        print(f'Experiment 2 trained: {f}')

print('\nData check complete.')
