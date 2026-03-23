#!/usr/bin/env python3
"""
validate_saved_model.py

Independent validation of a saved Koopman eDMDc model.
Loads model from .npz, reimplements delay embedding, performs open-loop simulation,
and reports prediction errors and structural properties.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def spectral_radius(A):
    return np.max(np.abs(eigvals(A)))

def unstable_modes(A):
    return np.sum(np.abs(eigvals(A)) > 1.0)

# ----------------------------------------------------------------------
# Main validation function
# ----------------------------------------------------------------------
def validate(patient):
    # Paths
    model_path = f"models/edmdc_{patient}.npz"
    data_path = f"data/{patient}.csv"
    results_dir = f"results/independent_validation_{patient}/"

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {model_path}")
    data = np.load(model_path)
    A = data["A"]
    B = data["B"]
    mean_g = data["mean_g"]
    std_g = data["std_g"]
    mean_i = data["mean_i"]
    std_i = data["std_i"]
    mean_m = data["mean_m"]
    std_m = data["std_m"]
    delays = int(data["delays"])
    dt = float(data["dt"])
    best_rank = int(data["best_rank"])  # not used directly

    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"Delays: {delays}, dt: {dt} min")
    print(f"Spectral radius: {spectral_radius(A):.4f}")
    print(f"Unstable modes: {unstable_modes(A)}")

    # ------------------------------------------------------------------
    # 2. Load and normalize data
    # ------------------------------------------------------------------
    print(f"Loading data from {data_path}")
    # Assume CSV has columns: time, BG, insulin, meal (time may be unused)
    raw = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    time = raw[:, 0]
    bg = raw[:, 1]
    insulin = raw[:, 2]
    meal = raw[:, 3]

    N = len(bg)
    print(f"Data length: {N} samples")

    # Normalize
    bg_norm = (bg - mean_g) / std_g
    insulin_norm = (insulin - mean_i) / std_i
    meal_norm = (meal - mean_m) / std_m
    u_norm = np.column_stack((insulin_norm, meal_norm))  # shape (N, 2)

    # ------------------------------------------------------------------
    # 3. Reconstruct initial lifted state and simulate
    # ------------------------------------------------------------------
    # State is assumed to be the vector of past `delays` normalized glucose values:
    #   phi_k = [bg_norm[k], bg_norm[k-1], ..., bg_norm[k-delays+1]]
    # This requires at least `delays` samples to form the first state.
    if N < delays:
        raise ValueError(f"Data length {N} < delays {delays}, cannot form initial state.")

    # Start index for which we have a full state
    start_idx = delays - 1

    # Initialize state with true data
    phi = bg_norm[start_idx - delays + 1 : start_idx + 1][::-1]  # reversed to have current last? Wait: we want [current, past1, past2,...]
    # Actually bg_norm slice [i-delays+1 : i+1] gives [past... , current]. We need [current, past...] for state.
    # Let's create function to get state at index i: state_i = bg_norm[i-delays+1 : i+1][::-1]  (reverse so that index 0 is current)
    # For i = start_idx, we get bg_norm[0:delays][::-1] = [bg_norm[delays-1], ..., bg_norm[0]]
    # That matches: current at position 0.
    # So we'll use that ordering.

    # Preallocate prediction array
    pred_bg_norm = np.zeros(N) * np.nan

    # Initial state at start_idx
    phi = bg_norm[start_idx - delays + 1 : start_idx + 1][::-1].copy()

    # Open-loop simulation: for k from start_idx to N-2, predict next state and use it as new state
    for k in range(start_idx, N - 1):
        # Predict next state
        phi_next = A @ phi + B @ u_norm[k]
        # Extract predicted glucose (first element)
        pred_bg_norm[k + 1] = phi_next[0]
        # Update state for next step (open-loop)
        phi = phi_next

    # Denormalize predictions
    pred_bg = pred_bg_norm * std_g + mean_g
    true_bg = bg

    # Valid indices: predictions exist from start_idx+1 to N-1
    valid_mask = ~np.isnan(pred_bg)
    pred_bg_valid = pred_bg[valid_mask]
    true_bg_valid = true_bg[valid_mask]

    # ------------------------------------------------------------------
    # 4. Compute error metrics
    # ------------------------------------------------------------------
    rmse_val = rmse(true_bg_valid, pred_bg_valid)
    mae_val = mae(true_bg_valid, pred_bg_valid)
    print(f"\nOverall Open-Loop Prediction Errors")
    print(f"RMSE: {rmse_val:.2f} mg/dL")
    print(f"MAE : {mae_val:.2f} mg/dL")

    # ------------------------------------------------------------------
    # 5. Long-horizon stress tests (from first full state)
    # ------------------------------------------------------------------
    # Number of steps corresponding to 2h, 6h, 12h
    steps_2h = int(2 * 60 / dt)
    steps_6h = int(6 * 60 / dt)
    steps_12h = int(12 * 60 / dt)

    # Starting index for long simulation (first full state)
    sim_start_idx = start_idx
    # Reset state to true initial condition
    phi_long = bg_norm[sim_start_idx - delays + 1 : sim_start_idx + 1][::-1].copy()
    sim_pred_norm = []
    sim_true_norm = bg_norm[sim_start_idx + 1 : sim_start_idx + 1 + steps_12h + 1]  # true up to horizon

    for step in range(steps_12h + 1):  # +1 to include prediction at horizon
        if step > 0:  # first step uses initial state, no prediction for t=sim_start_idx
            phi_long = A @ phi_long + B @ u_norm[sim_start_idx + step - 1]
            sim_pred_norm.append(phi_long[0])
        # else: skip, we don't predict the initial point

    sim_pred_norm = np.array(sim_pred_norm)
    sim_true_norm = sim_true_norm[1:]  # align with predictions

    # Compute RMSE for each horizon
    rmse_2h = rmse(sim_true_norm[:steps_2h] * std_g + mean_g,
                   sim_pred_norm[:steps_2h] * std_g + mean_g)
    rmse_6h = rmse(sim_true_norm[:steps_6h] * std_g + mean_g,
                   sim_pred_norm[:steps_6h] * std_g + mean_g)
    rmse_12h = rmse(sim_true_norm * std_g + mean_g,
                    sim_pred_norm * std_g + mean_g)

    print(f"\nLong-horizon RMSE from first state:")
    print(f"2h  : {rmse_2h:.2f} mg/dL")
    print(f"6h  : {rmse_6h:.2f} mg/dL")
    print(f"12h : {rmse_12h:.2f} mg/dL")

    # ------------------------------------------------------------------
    # 6. Structural validation
    # ------------------------------------------------------------------
    # State norm growth over time (using the open-loop simulation we already did)
    # We need to recompute state norms for each step. We can store them during simulation.
    # Let's re-run simulation but store norms.
    phi = bg_norm[start_idx - delays + 1 : start_idx + 1][::-1].copy()
    phi_norms = [np.linalg.norm(phi)]
    for k in range(start_idx, N - 1):
        phi = A @ phi + B @ u_norm[k]
        phi_norms.append(np.linalg.norm(phi))
    phi_norms = np.array(phi_norms)

    print(f"\nState norm statistics:")
    print(f"Initial norm : {phi_norms[0]:.4f}")
    print(f"Final norm   : {phi_norms[-1]:.4f}")
    print(f"Max norm     : {np.max(phi_norms):.4f}")
    print(f"Mean norm    : {np.mean(phi_norms):.4f}")
    if np.max(phi_norms) > 1e6:
        print("WARNING: State norm exploded!")

    # ------------------------------------------------------------------
    # 7. Plotting
    # ------------------------------------------------------------------
    time_axis = time  # assume time in minutes or hours, use as is

    # Figure 1: True vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, true_bg, 'b-', label='True BG', linewidth=1.5)
    plt.plot(time_axis, pred_bg, 'r--', label='Predicted BG (open-loop)', linewidth=1.5)
    plt.xlabel('Time (min)')
    plt.ylabel('BG (mg/dL)')
    plt.title(f'Patient {patient}: Open-Loop Koopman Simulation')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'true_vs_pred.png'), dpi=150)
    plt.close()

    # Figure 2: Residuals
    residuals = true_bg_valid - pred_bg_valid
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis[valid_mask], residuals, 'g-', linewidth=1)
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.xlabel('Time (min)')
    plt.ylabel('Residual (mg/dL)')
    plt.title('Prediction Residuals')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'residuals.png'), dpi=150)
    plt.close()

    # Figure 3: Error growth (cumulative RMSE over prediction horizon)
    # Compute RMSE over expanding window
    cum_rmse = [rmse(true_bg_valid[:i+1], pred_bg_valid[:i+1]) for i in range(len(true_bg_valid))]
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis[valid_mask], cum_rmse, 'm-', linewidth=1.5)
    plt.xlabel('Time (min)')
    plt.ylabel('Cumulative RMSE (mg/dL)')
    plt.title('Error Growth Over Prediction Horizon')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'error_growth.png'), dpi=150)
    plt.close()

    # Figure 4: Long-horizon example (first 12h)
    plt.figure(figsize=(12, 6))
    sim_time = time_axis[sim_start_idx + 1 : sim_start_idx + 1 + steps_12h + 1]
    plt.plot(sim_time, sim_true_norm * std_g + mean_g, 'b-', label='True BG')
    plt.plot(sim_time, sim_pred_norm * std_g + mean_g, 'r--', label='Predicted BG')
    plt.xlabel('Time (min)')
    plt.ylabel('BG (mg/dL)')
    plt.title(f'Patient {patient}: 12h Open-Loop Simulation from Start')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'long_horizon_12h.png'), dpi=150)
    plt.close()

    # Figure 5: State norm over time
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis[start_idx:], phi_norms, 'k-', linewidth=1)
    plt.xlabel('Time (min)')
    plt.ylabel('||Φ||')
    plt.title('Lifted State Norm Evolution')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'state_norm.png'), dpi=150)
    plt.close()

    print(f"\nPlots saved to {results_dir}")

    # ------------------------------------------------------------------
    # 8. Save summary text
    # ------------------------------------------------------------------
    summary_path = os.path.join(results_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Independent Validation Report for patient {patient}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model file: {model_path}\n")
        f.write(f"A shape: {A.shape}\n")
        f.write(f"B shape: {B.shape}\n")
        f.write(f"Spectral radius: {spectral_radius(A):.6f}\n")
        f.write(f"Unstable modes: {unstable_modes(A)}\n")
        f.write(f"Delays: {delays}\n")
        f.write(f"dt: {dt} min\n\n")
        f.write(f"Overall Open-Loop Errors:\n")
        f.write(f"  RMSE = {rmse_val:.2f} mg/dL\n")
        f.write(f"  MAE  = {mae_val:.2f} mg/dL\n\n")
        f.write(f"Long-horizon RMSE (from first state):\n")
        f.write(f"  2h  = {rmse_2h:.2f} mg/dL\n")
        f.write(f"  6h  = {rmse_6h:.2f} mg/dL\n")
        f.write(f"  12h = {rmse_12h:.2f} mg/dL\n\n")
        f.write(f"State norm statistics:\n")
        f.write(f"  Initial norm : {phi_norms[0]:.4f}\n")
        f.write(f"  Final norm   : {phi_norms[-1]:.4f}\n")
        f.write(f"  Max norm     : {np.max(phi_norms):.4f}\n")
        f.write(f"  Mean norm    : {np.mean(phi_norms):.4f}\n")
    print(f"Summary written to {summary_path}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_saved_model.py <patient>")
        sys.exit(1)
    patient = sys.argv[1]
    validate(patient)