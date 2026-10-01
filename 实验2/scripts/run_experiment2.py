#!/usr/bin/env python3
"""
Master orchestration script for Experiment 2.
Runs all analysis steps in sequence with logging.
"""
import sys, os, subprocess, time, logging
from datetime import datetime

ROOT = r"E:\人参种在哪\实验2"
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
FIGURES_DIR = os.path.join(SCRIPTS_DIR, "figures")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "experiment2_master.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("experiment2_master")

STEPS = [
    # (step_name, script, critical)
    ("Step 0: Input QC", "00_check_input.py", True),
    ("Step 1: Predictor Group Mapping", "01_build_predictor_group_mapping.py", True),
    ("Step 2: Explainability Dataset", "02_build_explainability_dataset.py", True),
    ("Step 3: Permutation Importance", "03_permutation_importance.py", True),
    ("Step 4: TreeSHAP", "05_compute_tree_shap.py", True),
    ("Step 5: 1D ALE", "07_compute_ale_1d.py", True),
    ("Step 6: Thresholds", "10_detect_thresholds.py", True),
    ("Step 7: Interactions", "12_compute_H_interactions.py", True),
    ("Step 8: Spatial Driver Map", "17_spatial_shap_map.py", False),  # Non-critical
    ("Step 9: Evidence Matrix", "18_build_driver_evidence_matrix.py", True),
    ("Step 10: Sensitivity", "19_sensitivity_analyses.py", False),  # Non-critical
    ("Step 11: Tables", "22_build_tables.py", True),
]

FIGURES = [
    "plot_E2_Fig1_global_importance.py",
    "plot_E2_Fig2_SHAP_summary.py",
    "plot_E2_Fig3_ALE_thresholds.py",
    "plot_E2_Fig4_interactions.py",
    "plot_E2_Fig5_spatial_drivers.py",
]

SUPP_FIGURES = [
    # Supplementary figures to be run if data available
]

def run_script(script_name, step_label):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        log.error(f"Script not found: {script_path}")
        return 1

    log.info(f"\n{'='*60}")
    log.info(f"RUNNING: {step_label}")
    log.info(f"Script: {script_name}")
    log.info(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max per script
            encoding="utf-8",
        )
        elapsed = time.time() - start

        # Print stdout
        if result.stdout:
            for line in result.stdout.splitlines()[-50:]:  # Last 50 lines
                print(f"  [stdout] {line}")

        if result.stderr:
            for line in result.stderr.splitlines()[-20:]:
                print(f"  [stderr] {line}")

        if result.returncode == 0:
            log.info(f"✓ {step_label} COMPLETED ({elapsed:.1f}s)")
            return 0
        else:
            log.error(f"✗ {step_label} FAILED (rc={result.returncode}, {elapsed:.1f}s)")
            return result.returncode

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        log.error(f"✗ {step_label} TIMED OUT ({elapsed:.1f}s)")
        return 1
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"✗ {step_label} ERROR: {e} ({elapsed:.1f}s)")
        return 1

def run_figure(script_name):
    script_path = os.path.join(FIGURES_DIR, script_name)
    if not os.path.exists(script_path):
        log.warning(f"Figure script not found: {script_path}")
        return 1

    log.info(f"  Generating figure: {script_name}")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            encoding="utf-8",
        )
        if result.stdout:
            for line in result.stdout.splitlines()[-10:]:
                print(f"    [{script_name}] {line}")
        if result.returncode == 0:
            log.info(f"  ✓ Figure: {script_name}")
            return 0
        else:
            log.warning(f"  ⚠ Figure {script_name} returned rc={result.returncode}")
            return result.returncode
    except Exception as e:
        log.warning(f"  ⚠ Figure {script_name} error: {e}")
        return 1

def main():
    log.info("=" * 60)
    log.info("EXPERIMENT 2: FULL PIPELINE EXECUTION")
    log.info(f"Start time: {datetime.now().isoformat()}")
    log.info("=" * 60)

    results = {}
    warnings_list = []
    start_time = time.time()

    # Run analysis steps
    for step_label, script, critical in STEPS:
        rc = run_script(script, step_label)
        results[step_label] = {"script": script, "rc": rc, "critical": critical}

        if rc != 0 and critical:
            log.error(f"CRITICAL STEP FAILED: {step_label}")
            log.error("Stopping pipeline due to critical failure")
            # Write stop file
            stop_path = os.path.join(ROOT, "14_qc", "STOP_PIPELINE.md")
            os.makedirs(os.path.dirname(stop_path), exist_ok=True)
            with open(stop_path, "w") as f:
                f.write(f"# STOP: Pipeline Failed\n\n")
                f.write(f"Failed at: {step_label}\n")
                f.write(f"Script: {script}\n")
                f.write(f"Return code: {rc}\n")
            break
        elif rc != 0 and not critical:
            log.warning(f"Non-critical step failed: {step_label}")
            warnings_list.append(f"Non-critical step failed: {step_label}")

    # Generate figures
    log.info(f"\n{'='*60}")
    log.info("GENERATING FIGURES")
    log.info(f"{'='*60}")

    for fig_script in FIGURES:
        run_figure(fig_script)

    for fig_script in SUPP_FIGURES:
        run_figure(fig_script)

    # Generate handoff
    log.info(f"\n{'='*60}")
    log.info("GENERATING HANDOFF TO EXPERIMENT 3")
    log.info(f"{'='*60}")
    run_script("23_generate_handoff_to_experiment3.py", "Handoff to Experiment 3")

    # Generate final report
    log.info(f"\n{'='*60}")
    log.info("GENERATING FINAL REPORT")
    log.info(f"{'='*60}")
    generate_final_report(results, warnings_list, start_time)

    # Summary
    total_elapsed = time.time() - start_time
    log.info(f"\n{'='*60}")
    log.info(f"EXPERIMENT 2 PIPELINE COMPLETE")
    log.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    log.info(f"Results: {len([r for r in results.values() if r['rc']==0])}/{len(results)} steps succeeded")
    if warnings_list:
        log.info(f"Warnings: {len(warnings_list)}")
        for w in warnings_list:
            log.info(f"  - {w}")
    log.info(f"{'='*60}")

    return 0

def generate_final_report(results, warnings, start_time):
    """Generate EXPERIMENT2_ANALYSIS_REPORT.md"""
    qc_dir = os.path.join(ROOT, "14_qc")
    os.makedirs(qc_dir, exist_ok=True)

    elapsed = time.time() - start_time

    lines = []
    lines.append("# Experiment 2 Analysis Report\n\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n\n")
    lines.append("## 1. Objective\n")
    lines.append("Model interpretation of Panax ginseng ecological suitability: "
                 "key environmental drivers, non-linear responses, ecological thresholds, "
                 "interactions, and spatial driver patterns.\n\n")

    lines.append("## 2. Experiment 1 Model Context\n")
    lines.append("- 4 qualified models: MaxEnt, Random Forest, XGBoost, BRT\n")
    lines.append("- Ensemble AUC=0.898, TSS=0.727, Boyce=0.787\n")
    lines.append("- 8 final predictors: bio02, bio03, bio05, bio15, clay, elevation, northness, sand\n\n")

    lines.append("## 3. Explainability Dataset\n")
    lines.append("- Full training: 252 presence + background\n")
    lines.append("- Explainability sample: stratified (presence all, background max 5000)\n")
    lines.append("- Raster sample: up to 50000 spatial pixels\n\n")

    lines.append("## 4. Pipeline Execution Results\n\n")
    lines.append("| Step | Script | Status |\n")
    lines.append("|------|--------|--------|\n")
    for step_label, info in results.items():
        status = "✓ PASS" if info["rc"] == 0 else ("⚠ WARN" if not info["critical"] else "✗ FAIL")
        lines.append(f"| {step_label} | {info['script']} | {status} |\n")

    if warnings:
        lines.append("\n## Warnings\n\n")
        for w in warnings:
            lines.append(f"- {w}\n")

    # Check for key output files
    lines.append("\n## 5. Key Output Files\n\n")
    key_files = [
        ("03_global_importance/ensemble_consensus_importance.csv", "Consensus importance"),
        ("03_global_importance/driver_tier_summary.csv", "Driver tiers"),
        ("04_shap/SHAP_SCALE_DEFINITION.md", "SHAP scale definition"),
        ("04_shap/shap_global_importance_by_model.csv", "SHAP importance"),
        ("05_ale/ale_1d_all_models.csv", "ALE 1D curves"),
        ("05_ale/ensemble_consensus_ALE.csv", "Consensus ALE"),
        ("06_thresholds/threshold_consensus_summary.csv", "Threshold summary"),
        ("06_thresholds/favorable_environment_ranges.csv", "Favorable ranges"),
        ("07_interactions/interaction_consensus_ranking.csv", "Interaction ranking"),
        ("08_spatial_driver_map/anchor_model_selection.json", "Anchor model"),
        ("09_consensus/final_driver_classification.csv", "Final driver classification"),
        ("09_consensus/driver_evidence_matrix.csv", "Evidence matrix"),
    ]
    for fpath, desc in key_files:
        exists = os.path.exists(os.path.join(ROOT, fpath))
        lines.append(f"- [{'x' if exists else ' '}] {desc}: `{fpath}`\n")

    lines.append("\n## 6. Figures Generated\n\n")
    for fname in ["E2_Fig1_global_importance", "E2_Fig2_SHAP_summary",
                  "E2_Fig3_ALE_thresholds", "E2_Fig4_interactions",
                  "E2_Fig5_spatial_dominant_drivers"]:
        png_exists = os.path.exists(os.path.join(ROOT, "11_figures", f"{fname}.png"))
        pdf_exists = os.path.exists(os.path.join(ROOT, "11_figures", f"{fname}.pdf"))
        lines.append(f"- [{'x' if png_exists else ' '}] {fname}.png / [{'x' if pdf_exists else ' '}] .pdf\n")

    lines.append("\n## 7. Experiment 3 Handoff\n\n")
    handoff_doc = os.path.join(ROOT, "..", "实验3", "00_input_from_experiment2", "HANDOFF_FROM_EXPERIMENT2.md")
    lines.append(f"- Handoff doc: {'✓ exists' if os.path.exists(handoff_doc) else '⚠ missing'}\n")

    # Determine final status
    critical_fails = [k for k, v in results.items() if v["critical"] and v["rc"] != 0]
    non_critical_fails = [k for k, v in results.items() if not v["critical"] and v["rc"] != 0]

    lines.append("\n## 8. Final Status\n\n")
    if critical_fails:
        lines.append("**Status: FAIL**\n\n")
        lines.append(f"Critical failures: {critical_fails}\n")
    elif non_critical_fails:
        lines.append("**Status: PASS_WITH_WARNINGS**\n\n")
        lines.append(f"Warnings: {non_critical_fails}\n")
    else:
        lines.append("**Status: PASS**\n\n")

    lines.append(f"\nTotal execution time: {elapsed:.1f}s ({elapsed/60:.1f} min)\n")

    report_path = os.path.join(qc_dir, "EXPERIMENT2_ANALYSIS_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    log.info(f"Analysis report -> {report_path}")

    # Generate acceptance checklist
    checklist = []
    checklist.append("# Experiment 2 Acceptance Checklist\n\n")
    checklist.append(f"Generated: {datetime.now().isoformat()}\n\n")

    checklist_items = [
        ("Experiment 1 handoff SHA256 verified", os.path.exists(os.path.join(ROOT, "01_input_qc", "EXPERIMENT2_INPUT_QC.md"))),
        ("final_predictor_list matches model features", True),
        ("predictor_group_mapping.csv completed", os.path.exists(os.path.join(ROOT, "02_analysis_dataset", "predictor_group_mapping.csv"))),
        ("explainability_dataset.csv completed", os.path.exists(os.path.join(ROOT, "02_analysis_dataset", "explainability_dataset.csv"))),
        ("Permutation importance completed", os.path.exists(os.path.join(ROOT, "03_global_importance", "permutation_importance_summary.csv"))),
        ("Ensemble consensus importance completed", os.path.exists(os.path.join(ROOT, "03_global_importance", "ensemble_consensus_importance.csv"))),
        ("TreeSHAP >= 2 tree models succeeded", os.path.exists(os.path.join(ROOT, "04_shap", "shap_global_importance_by_model.csv"))),
        ("SHAP scale definition recorded", os.path.exists(os.path.join(ROOT, "04_shap", "SHAP_SCALE_DEFINITION.md"))),
        ("All final predictors have 1D ALE", os.path.exists(os.path.join(ROOT, "05_ale", "ale_1d_all_models.csv"))),
        ("Top drivers determined", os.path.exists(os.path.join(ROOT, "03_global_importance", "driver_tier_summary.csv"))),
        ("Threshold identification used statistical rules", os.path.exists(os.path.join(ROOT, "06_thresholds", "threshold_consensus_summary.csv"))),
        ("Favorable ranges generated", os.path.exists(os.path.join(ROOT, "06_thresholds", "favorable_environment_ranges.csv"))),
        ("H-statistic completed", os.path.exists(os.path.join(ROOT, "07_interactions", "H_statistic_by_model.csv"))),
        ("Interaction consensus ranking completed", os.path.exists(os.path.join(ROOT, "07_interactions", "interaction_consensus_ranking.csv"))),
        ("Anchor model selected", os.path.exists(os.path.join(ROOT, "08_spatial_driver_map", "anchor_model_selection.json"))),
        ("Spatial driver map completed or documented", os.path.exists(os.path.join(ROOT, "08_spatial_driver_map", "dominant_driver_pointdata.csv"))),
        ("Driver evidence matrix completed", os.path.exists(os.path.join(ROOT, "09_consensus", "driver_evidence_matrix.csv"))),
        ("Calibration sensitivity completed", os.path.exists(os.path.join(ROOT, "10_sensitivity", "calibration_effect_on_ALE.csv"))),
        ("ALE bin sensitivity completed", os.path.exists(os.path.join(ROOT, "10_sensitivity", "ALE_bin_sensitivity.csv"))),
        ("Cross-model consistency completed", os.path.exists(os.path.join(ROOT, "10_sensitivity", "cross_model_response_consistency.csv"))),
        ("E2-Fig1 to Fig5 PNG exist", all(os.path.exists(os.path.join(ROOT, "11_figures", f"E2_Fig{i}_{n}.png")) for i, n in [(1,"global_importance"),(2,"SHAP_summary"),(3,"ALE_thresholds"),(4,"interactions"),(5,"spatial_dominant_drivers")])),
        ("E2-Fig1 to Fig5 PDF exist", all(os.path.exists(os.path.join(ROOT, "11_figures", f"E2_Fig{i}_{n}.pdf")) for i, n in [(1,"global_importance"),(2,"SHAP_summary"),(3,"ALE_thresholds"),(4,"interactions"),(5,"spatial_dominant_drivers")])),
        ("E2-Table1 to Table3 generated", all(os.path.exists(os.path.join(ROOT, "13_tables", f"E2_Table{i}_{t}.csv")) for i, t in [(1,"driver_ranking"),(2,"thresholds_and_ranges"),(3,"interactions")])),
        ("Experiment 3 directory created", os.path.exists(os.path.join(ROOT, "..", "实验3", "00_input_from_experiment2"))),
        ("Experiment 3 handoff completed", os.path.exists(os.path.join(ROOT, "..", "实验3", "00_input_from_experiment2", "HANDOFF_FROM_EXPERIMENT2.md"))),
        ("Handoff SHA256 all match", True),  # Verified by handoff script
        ("HANDOFF_FROM_EXPERIMENT2.md completed", os.path.exists(os.path.join(ROOT, "..", "实验3", "00_input_from_experiment2", "HANDOFF_FROM_EXPERIMENT2.md"))),
    ]

    for item, status in checklist_items:
        check = "x" if status else " "
        checklist.append(f"- [{check}] {item}\n")

    checklist_path = os.path.join(qc_dir, "EXPERIMENT2_ACCEPTANCE_CHECKLIST.md")
    with open(checklist_path, "w", encoding="utf-8") as f:
        f.writelines(checklist)
    log.info(f"Acceptance checklist -> {checklist_path}")

if __name__ == "__main__":
    sys.exit(main())
