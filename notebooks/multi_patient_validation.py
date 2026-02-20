#!/usr/bin/env python3
"""
Multi-patient validation for eDMDc glucose-insulin pipeline.

Runs the stability-constrained eDMDc pipeline for multiple patients,
extracts key metrics, and produces a comparison table and summary files.
"""

import os
import sys
import subprocess
import re
import pandas as pd

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PATIENTS = ["adolescent_001", "adult_001", "child_001"]
RESULTS_DIR = "results"

# Regex patterns for extracting metrics
PAT_RANK = re.compile(r"Rank used:\s+(\d+)")

# Old single-line RMSE patterns (fallback)
PAT_RMSE_2H_OLD = re.compile(r"2h RMSE:\s+([\d.]+)\s+mg/dL")
PAT_RMSE_6H_OLD = re.compile(r"6h RMSE:\s+([\d.]+)\s+mg/dL")
PAT_RMSE_12H_OLD = re.compile(r"12h RMSE:\s+([\d.]+)\s+mg/dL")

# New patterns for "full=" values from comparison block
PAT_FULL_2H = re.compile(r"2h RMSE:\s*full=([\d.]+)")
PAT_FULL_6H = re.compile(r"6h RMSE:\s*full=([\d.]+)")
PAT_FULL_12H = re.compile(r"12h RMSE:\s*full=([\d.]+)")

PAT_SPECTRAL_RADIUS = re.compile(r"Spectral radius:\s+([\d.]+)")
PAT_CONTROLLABILITY_RANK = re.compile(r"Controllability rank:\s+(\d+)\s+/\s+\d+")
PAT_MODAL_MAX = re.compile(r"Max:\s+([\deE\.+-]+)")
PAT_MODAL_MIN = re.compile(r"Min:\s+([\deE\.+-]+)")
PAT_MODAL_RATIO = re.compile(r"Ratio \(min/max\):\s+([\deE\.+-]+)")

def run_pipeline(patient):
    """Run edmdc_pipeline.py for one patient and return stdout as string."""
    cmd = [sys.executable, "edmdc_pipeline.py", "--patient", patient]
    print(f"Running: {' '.join(cmd)}")
    
    # Set environment to force UTF-8 encoding to avoid UnicodeEncodeError
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Pipeline failed for {patient} (return code {e.returncode})")
        print(e.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error running pipeline for {patient}: {e}")
        return None

def parse_metrics(stdout):
    """Extract metrics from pipeline stdout. Return dict or None if missing."""
    if stdout is None:
        return None
    data = {}

    # Rank
    m = PAT_RANK.search(stdout)
    data["rank"] = int(m.group(1)) if m else None

    # RMSEs: try full= pattern first, fallback to old single-line pattern
    m = PAT_FULL_2H.search(stdout)
    if m:
        data["rmse_2h"] = float(m.group(1))
    else:
        m = PAT_RMSE_2H_OLD.search(stdout)
        data["rmse_2h"] = float(m.group(1)) if m else None

    m = PAT_FULL_6H.search(stdout)
    if m:
        data["rmse_6h"] = float(m.group(1))
    else:
        m = PAT_RMSE_6H_OLD.search(stdout)
        data["rmse_6h"] = float(m.group(1)) if m else None

    m = PAT_FULL_12H.search(stdout)
    if m:
        data["rmse_12h"] = float(m.group(1))
    else:
        m = PAT_RMSE_12H_OLD.search(stdout)
        data["rmse_12h"] = float(m.group(1)) if m else None

    # Spectral radius
    m = PAT_SPECTRAL_RADIUS.search(stdout)
    data["spectral_radius"] = float(m.group(1)) if m else None

    # Controllability rank
    m = PAT_CONTROLLABILITY_RANK.search(stdout)
    data["controllability_rank"] = int(m.group(1)) if m else None

    # Modal controllability statistics
    m = PAT_MODAL_MAX.search(stdout)
    data["modal_max"] = float(m.group(1)) if m else None
    m = PAT_MODAL_MIN.search(stdout)
    data["modal_min"] = float(m.group(1)) if m else None
    m = PAT_MODAL_RATIO.search(stdout)
    data["modal_ratio"] = float(m.group(1)) if m else None

    return data

def main():
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_data = []
    for patient in PATIENTS:
        print("\n" + "=" * 60)
        print(f"Processing patient: {patient}")
        print("=" * 60)
        stdout = run_pipeline(patient)
        metrics = parse_metrics(stdout)
        if metrics:
            metrics["patient"] = patient
            all_data.append(metrics)
        else:
            print(f"WARNING: No metrics extracted for {patient}. Skipping.")
            # Append a row with NaNs
            all_data.append({
                "patient": patient,
                "rank": None,
                "spectral_radius": None,
                "rmse_2h": None,
                "rmse_6h": None,
                "rmse_12h": None,
                "controllability_rank": None,
                "modal_max": None,
                "modal_min": None,
                "modal_ratio": None
            })

    # Create DataFrame
    df = pd.DataFrame(all_data)
    # Reorder columns
    cols = ["patient", "rank", "spectral_radius", "rmse_2h", "rmse_6h", "rmse_12h",
            "controllability_rank", "modal_max", "modal_min", "modal_ratio"]
    df = df[cols]

    # Print nice table using pandas built-in formatting
    print("\n" + "=" * 60)
    print("MULTI-PATIENT COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False, float_format="%.4f"))

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "multi_patient_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV summary saved to: {csv_path}")

    # Save formatted text report
    txt_path = os.path.join(RESULTS_DIR, "multi_patient_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Multi-Patient eDMDc Validation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(df.to_string(index=False, float_format="%.4f"))
    print(f"Text report saved to: {txt_path}")

    print("\nMulti-patient validation complete.")

if __name__ == "__main__":
    main()