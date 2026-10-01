import pandas as pd, os

root = r'E:\人参种在哪\实验3'

files_to_check = [
    '09_group_shapley/group_shapley_summary.csv',
    '09_group_shapley/group_shapley_contributions.csv',
    '08_ablation/drop_one_group_loss.csv',
    '08_ablation/standalone_predictive_power.csv',
    '10_complementarity/pairwise_synergy.csv',
    '10_complementarity/incremental_gain_paths.csv',
    '07_performance_comparison/ensemble_subset_summary.csv',
    '13_sensitivity/metric_sensitivity.csv',
    '13_sensitivity/group_contribution_by_algorithm.csv',
]

for f in files_to_check:
    path = os.path.join(root, f)
    exists = os.path.exists(path)
    print(f'{f}: {"EXISTS" if exists else "MISSING"}')

print('\n========== GROUP SHAPLEY SUMMARY ==========')
shap = pd.read_csv(os.path.join(root, '09_group_shapley/group_shapley_summary.csv'))
print(shap.to_string())

print('\n========== DROP-ONE ABLATION (AUC) ==========')
abl = pd.read_csv(os.path.join(root, '08_ablation/drop_one_group_loss.csv'))
print(abl.to_string())

print('\n========== ENSEMBLE SUBSET SUMMARY ==========')
ens = pd.read_csv(os.path.join(root, '07_performance_comparison/ensemble_subset_summary.csv'))
print(ens[['Subset','auc_mean','auc_SD','tss_mean']].to_string())

print('\n========== PAIRWISE SYNERGY ==========')
syn = pd.read_csv(os.path.join(root, '10_complementarity/pairwise_synergy.csv'))
print(syn.to_string())

print('\n========== METRIC SENSITIVITY ==========')
ms = pd.read_csv(os.path.join(root, '13_sensitivity/metric_sensitivity.csv'))
for metric in ['boyce', 'tss', 'auc']:
    shap_rows = ms[(ms['Metric'] == metric) & (ms['Type'] == 'Shapley %')]
    print(f'{metric}:', dict(zip(shap_rows['Group'], shap_rows['Value'].round(1))))
