#!/usr/bin/env python3
"""
Experiment 4 — Master Runner Script
Orchestrates the complete experiment pipeline from QC through handoff.

Usage:
    python run_experiment4.py [--step STEP_NUMBER] [--resume]

Steps:
    0  - Input QC
    1  - CMIP6 Download & Extraction
    2  - Climate Alignment & QC
    3  - Future Predictions (individual + ensemble + binary + change)
    4  - Analysis (area, centroid, elevation, landscape, SSP comparison)
    5  - Figures
    6  - Handoff & Report
    all - Run everything (default)
"""
import os, sys, subprocess, argparse
from pathlib import Path

EXP4_DIR = Path(r"E:\人参种在哪\实验4")
SCRIPTS_DIR = EXP4_DIR / "scripts"

STEPS = {
    "0": {
        "name": "Input QC",
        "script": SCRIPTS_DIR / "00_check_input.py",
        "description": "Verify Experiment 3 handoff data",
    },
    "1": {
        "name": "CMIP6 Download & Process",
        "script": SCRIPTS_DIR / "02_download_process_cmip6.py",
        "description": "Download and extract future climate data",
    },
    "2": {
        "name": "Climate Alignment & QC",
        "script": SCRIPTS_DIR / "03_align_and_qc_future_climate.py",
        "description": "Align future climate to reference grid, unit check, range exceedance",
    },
    "3": {
        "name": "Future Predictions",
        "script": SCRIPTS_DIR / "04_predict_future.py",
        "description": "Build predictor stacks, run 4 models, create ensemble, binary, change maps",
    },
    "4": {
        "name": "Analysis Pipeline",
        "script": SCRIPTS_DIR / "05_analysis_pipeline.py",
        "description": "Area statistics, centroid, elevation, landscape, SSP comparison",
    },
    "5": {
        "name": "Figures",
        "script": SCRIPTS_DIR / "figures" / "plot_figures.py",
        "description": "Generate all main and supplementary figures",
    },
    "6": {
        "name": "Handoff & Report",
        "script": SCRIPTS_DIR / "06_generate_handoff_and_report.py",
        "description": "Create Experiment 5 handoff, final report, acceptance checklist",
    },
}

def run_step(step_id):
    info = STEPS[step_id]
    print(f"\n{'=' * 60}")
    print(f"Step {step_id}: {info['name']}")
    print(f"{'=' * 60}")
    print(f"Script: {info['script']}")
    print(f"Description: {info['description']}")
    print(f"{'=' * 60}\n")

    if not info["script"].exists():
        print(f"ERROR: Script not found: {info['script']}")
        return False

    result = subprocess.run(
        [sys.executable, str(info["script"])],
        cwd=str(EXP4_DIR),
    )
    if result.returncode != 0:
        print(f"\nStep {step_id} FAILED with exit code {result.returncode}")
        return False
    print(f"\nStep {step_id} completed successfully.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Experiment 4 Master Runner")
    parser.add_argument("--step", type=str, default="all",
                       help="Step to run (0-6, or 'all')")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from the specified step")
    parser.add_argument("--list", action="store_true",
                       help="List all steps and exit")
    args = parser.parse_args()

    if args.list:
        print("\nExperiment 4 — Available Steps:\n")
        for sid, info in STEPS.items():
            print(f"  Step {sid}: {info['name']}")
            print(f"    Script: {info['script'].name}")
            print(f"    {info['description']}")
            print()
        return

    if args.step == "all":
        print("\n" + "=" * 60)
        print("EXPERIMENT 4 — COMPLETE PIPELINE")
        print("=" * 60)
        for sid in sorted(STEPS.keys()):
            if not run_step(sid):
                print(f"\nPipeline stopped at Step {sid}. Fix issues and re-run with --step {sid} --resume")
                break
        else:
            print("\n" + "=" * 60)
            print("EXPERIMENT 4 COMPLETE!")
            print("Check results in:")
            print(f"  {EXP4_DIR / '19_qc' / 'EXPERIMENT4_ANALYSIS_REPORT.md'}")
            print(f"  {EXP4_DIR / '19_qc' / 'EXPERIMENT4_ACCEPTANCE_CHECKLIST.md'}")
            print("=" * 60)
    elif args.step in STEPS:
        run_step(args.step)
    else:
        print(f"Invalid step: {args.step}. Use --list to see available steps.")

if __name__ == "__main__":
    main()
