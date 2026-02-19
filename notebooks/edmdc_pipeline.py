#!/usr/bin/env python3
"""
Stability-Constrained eDMDc Pipeline for Glucose-Insulin System

This script implements a complete pipeline for learning a control-oriented
surrogate model of the glucose-insulin system using Extended Dynamic Mode
Decomposition with control (eDMDc).  It includes:

- Data loading, cleaning, and chronological splitting
- Delay embedding and nonlinear feature lifting (polynomial, optional RBF)
- Ridge and truncated-SVD identification with automatic rank selection
- Stability enforcement by projecting unstable eigenvalues inside the unit circle
- Multi-step prediction and comprehensive evaluation
- Publication-ready plots and model saving

Usage example:
    python edmdc_pipeline.py --patient adolescent_001 --delays 4 --poly_order 2
"""

import os
import sys
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd, solve, eig, inv

# ----------------------------------------------------------------------
#  Command line arguments (modified to work in Jupyter)
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='eDMDc for glucose-insulin')
    parser.add_argument('--patient', type=str, default='adolescent_001',
                        help='Patient identifier (CSV file name without extension)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing patient CSV files')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save plots')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory to save trained model')
    parser.add_argument('--delays', type=int, default=4,
                        help='Number of delays for each variable')
    parser.add_argument('--dt', type=float, default=3.0,
                        help='Sampling interval in minutes')
    parser.add_argument('--poly_order', type=int, default=2,
                        help='Order of polynomial features (0 = no poly, 1 = identity only)')
    parser.add_argument('--use_rbf', action='store_true',
                        help='Add RBF features (experimental)')
    parser.add_argument('--rbf_centers', type=int, default=5,
                        help='Number of RBF centers per dimension')
    parser.add_argument('--ridge_lambda', type=float, default=1e-4,
                        help='Ridge regularisation parameter')
    parser.add_argument('--energy_threshold', type=float, default=0.999,
                        help='Cumulative energy for automatic rank (SVD)')
    parser.add_argument('--max_rank', type=int, default=None,
                        help='Maximum rank to consider (default: feature dimension + input dim)')
    parser.add_argument('--prediction_horizon', type=int, default=40,
                        help='Number of steps for 2-hour prediction (default 40 * 3 min)')
    parser.add_argument('--train_frac', type=float, default=0.8,
                        help='Fraction of data for training')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Fraction of data for validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Parse known args to avoid conflicts with Jupyter kernel arguments
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}", file=sys.stderr)
    return args

# ----------------------------------------------------------------------
#  Data loading and preprocessing
# ----------------------------------------------------------------------
def load_patient_data(patient, data_dir, dt):
    """Load a single patient CSV, sort by time, handle missing values."""
    file_path = os.path.join(data_dir, f"{patient}.csv")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Patient file not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=['time'])
    df.sort_values('time', inplace=True)
    df.drop_duplicates(subset=['time'], inplace=True)

    # Forward-fill then backward-fill missing values in essential columns
    essential = ['BG', 'insulin', 'meal']
    for col in essential:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()
    if df[essential].isnull().any().any():
        raise ValueError(f"NaNs remain in patient {patient} after filling.")

    # Ensure uniform sampling (warn if not)
    if 't_minutes' in df.columns:
        t = df['t_minutes'].values
    else:
        # create t_minutes from datetime
        t = (df['time'] - df['time'].iloc[0]).dt.total_seconds() / 60.0
        df['t_minutes'] = t

    diffs = np.diff(t)
    unique_diffs = np.unique(diffs)
    if not np.allclose(unique_diffs, dt, atol=0.1):
        warnings.warn(f"Sampling interval not uniform {dt} min. Found {unique_diffs}")

    BG = df['BG'].values.astype(float)
    insulin = df['insulin'].values.astype(float)
    meal = df['meal'].values.astype(float)
    t_minutes = t.astype(float)

    return BG, insulin, meal, t_minutes

def normalize_train_val_test(train, val, test):
    """Compute mean/std from training set and normalize all sets."""
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std[std == 0] = 1.0
    train_norm = (train - mean) / std
    val_norm = (val - mean) / std
    test_norm = (test - mean) / std
    return train_norm, val_norm, test_norm, mean, std

# ----------------------------------------------------------------------
#  Delay embedding
# ----------------------------------------------------------------------
def delay_embed(data, d):
    """
    data: 1D array of length T
    returns: matrix (d, T-d+1) where each column is [x_k, x_{k-1}, ..., x_{k-d+1]]^T
    """
    T = len(data)
    n = T - d + 1
    emb = np.zeros((d, n))
    for i in range(d):
        emb[i, :] = data[d-1-i : T-i]
    return emb

# ----------------------------------------------------------------------
#  Nonlinear lifting
# ----------------------------------------------------------------------
def build_polynomial_features(Z, order):
    """
    Z: raw state matrix (n_raw, N)
    order: polynomial order (1 = identity only, 2 = add squares and all pairwise products)
    Returns lifted matrix (n_features, N)
    """
    n_raw, N = Z.shape
    features = [Z]   # identity
    if order >= 2:
        # Add all pairwise products (including squares)
        ZT = Z.T  # (N, n_raw)
        quad_list = []
        for i in range(n_raw):
            for j in range(i, n_raw):
                quad_list.append(ZT[:, i] * ZT[:, j])
        quad = np.array(quad_list).T  # (N, n_quad)
        features.append(quad.T)
    return np.vstack(features)

def rbf_centers_from_data(data, n_centers):
    """Compute quantile-based centers for each column of data."""
    centers = []
    for col in range(data.shape[1]):
        q = np.linspace(0, 100, n_centers+2)[1:-1]
        cents = np.percentile(data[:, col], q)
        centers.append(cents)
    return np.array(centers)

def rbf_features(Z, centers, gamma=1.0):
    """
    Z: raw state (n_raw, N)
    centers: (n_raw, n_centers)
    returns RBF features (n_raw * n_centers, N)
    """
    n_raw, N = Z.shape
    n_centers = centers.shape[1]
    rbf = np.zeros((n_raw * n_centers, N))
    idx = 0
    for i in range(n_raw):
        for j in range(n_centers):
            rbf[idx, :] = np.exp(-gamma * (Z[i, :] - centers[i, j])**2)
            idx += 1
    return rbf

# ----------------------------------------------------------------------
#  Build eDMDc data matrices (X, X', U)
# ----------------------------------------------------------------------
def build_edmdc_matrices(Phi, U):
    """
    Phi: lifted features (n_features, T)
    U: inputs (2, T)
    Returns X, Xprime, Umat (all with N = T-1 columns)
    """
    X = Phi[:, :-1]
    Xprime = Phi[:, 1:]
    Umat = U[:, :-1]
    return X, Xprime, Umat

# ----------------------------------------------------------------------
#  Identification methods
# ----------------------------------------------------------------------
def train_ridge(X, Xprime, U, lam):
    """Ridge regression: [A B] = X' Omega^T (Omega Omega^T + lam I)^{-1}"""
    n = X.shape[0]
    m = U.shape[0]
    Omega = np.vstack([X, U])
    # M = X' Omega^T
    M = Xprime @ Omega.T
    # Gram = Omega Omega^T + lam I
    Gram = Omega @ Omega.T + lam * np.eye(n + m)
    # Solve Gram * Y = M^T  -> Y = (A B)^T
    Y = solve(Gram, M.T, assume_a='pos')
    AB = Y.T
    A = AB[:, :n]
    B = AB[:, n:]
    return A, B

def train_truncated_svd(X, Xprime, U, rank):
    """[A B] = X' Omega^+ using truncated SVD."""
    Omega = np.vstack([X, U])
    U1, s, Vh = svd(Omega, full_matrices=False)
    if rank == 'full' or rank >= len(s):
        r = len(s)
    else:
        r = int(rank)
    # Pseudoinverse
    Omega_pinv = Vh[:r, :].T @ np.diag(1.0 / s[:r]) @ U1[:, :r].T
    AB = Xprime @ Omega_pinv
    n = X.shape[0]
    A = AB[:, :n]
    B = AB[:, n:]
    return A, B

# ----------------------------------------------------------------------
#  Stability enforcement
# ----------------------------------------------------------------------
def enforce_stability(A, tol=0.99):
    """
    If any eigenvalue has magnitude > 1, scale it inside the unit circle.
    Returns stabilized A and list of indices that were clipped.
    """
    eigvals, V = eig(A)
    unstable = np.abs(eigvals) > 1.0
    if not np.any(unstable):
        return A, []
    eigvals_stable = eigvals.copy()
    for i in np.where(unstable)[0]:
        eigvals_stable[i] = tol * eigvals[i] / np.abs(eigvals[i])
    # Reconstruct A = V diag(λ) V^{-1}
    A_stable = V @ np.diag(eigvals_stable) @ inv(V)
    # Take real part (should be real if complex pairs appear together)
    A_stable = np.real(A_stable)
    clipped = np.where(unstable)[0].tolist()
    return A_stable, clipped

# ----------------------------------------------------------------------
#  Multi-step prediction
# ----------------------------------------------------------------------
def predict_multi_step(A, B, phi0, U_seq, mean_g, std_g, clip=True):
    """
    Roll out model for len(U_seq) steps.
    Returns predicted glucose (mg/dL) for steps 0..len(U_seq) (inclusive).
    """
    steps = U_seq.shape[1]
    n = A.shape[0]
    Phi_pred = np.zeros((n, steps + 1))
    Phi_pred[:, 0] = phi0
    for t in range(steps):
        Phi_pred[:, t+1] = A @ Phi_pred[:, t] + B @ U_seq[:, t]
        if clip:
            # Clip extreme values to avoid explosion (optional)
            Phi_pred[:, t+1] = np.clip(Phi_pred[:, t+1], -10, 10)
    glucose_norm = Phi_pred[0, :]
    glucose = glucose_norm * std_g + mean_g
    return glucose

def compute_metrics(true, pred):
    rmse = np.sqrt(np.mean((true - pred)**2))
    mae = np.mean(np.abs(true - pred))
    nrmse = rmse / (np.max(true) - np.min(true)) if np.max(true) != np.min(true) else np.nan
    return rmse, mae, nrmse

# ----------------------------------------------------------------------
#  Rank selection via validation
# ----------------------------------------------------------------------
def select_rank_by_validation(X_tr, Xpr_tr, U_tr,
                              X_val, Xpr_val, U_val,
                              mean_g, std_g,
                              rank_list, prediction_horizons=(10,20,40),
                              method='svd'):
    """
    For each rank, train on training set and evaluate average RMSE over
    multiple prediction horizons on validation set. Return best rank and model.
    """
    best_rank = None
    best_score = np.inf
    best_A = None
    best_B = None

    # Pre-compute true glucose sequences for validation at each horizon
    phi0_val = X_val[:, 0]
    # For each horizon, we need true glucose values of length horizon+1
    true_vals = {}
    for H in prediction_horizons:
        if H >= X_val.shape[1]:
            H = X_val.shape[1] - 1
        true_norm = np.concatenate([[X_val[0,0]], Xpr_val[0, :H]])
        true_vals[H] = true_norm * std_g + mean_g

    for rank in rank_list:
        if method == 'svd':
            A, B = train_truncated_svd(X_tr, Xpr_tr, U_tr, rank)
        else:
            raise ValueError("Only 'svd' method supported for rank selection")
        # Evaluate on each horizon
        rmse_list = []
        for H in prediction_horizons:
            if H >= X_val.shape[1]:
                continue
            U_seq = U_val[:, :H]
            pred = predict_multi_step(A, B, phi0_val, U_seq, mean_g, std_g, clip=True)
            true = true_vals[H]
            rmse = compute_metrics(true, pred)[0]
            rmse_list.append(rmse)
        if not rmse_list:
            continue
        avg_rmse = np.mean(rmse_list)
        if avg_rmse < best_score:
            best_score = avg_rmse
            best_rank = rank
            best_A = A.copy()
            best_B = B.copy()

    return best_rank, best_A, best_B, best_score

# ----------------------------------------------------------------------
#  Plotting functions (save to results_dir)
# ----------------------------------------------------------------------
def plot_singular_values(s, save_path):
    plt.figure(figsize=(8,4))
    plt.semilogy(s, 'o-', markersize=4)
    plt.xlabel('Index')
    plt.ylabel('Singular value')
    plt.title('Singular value spectrum of $\\Omega$')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_eigenvalues(A, save_path):
    eigvals = np.linalg.eigvals(A)
    plt.figure(figsize=(6,6))
    stable = np.abs(eigvals) < 1.0
    plt.plot(np.real(eigvals[stable]), np.imag(eigvals[stable]), 'bo', label='Stable')
    plt.plot(np.real(eigvals[~stable]), np.imag(eigvals[~stable]), 'rx', label='Unstable')
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1)
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.title('Eigenvalues of $A$')
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_predictions(time_axis, true, pred, save_path):
    plt.figure(figsize=(10,4))
    plt.plot(time_axis, true, 'b-', linewidth=1.5, label='True BG')
    plt.plot(time_axis, pred, 'r--', linewidth=1.5, label='Predicted BG')
    plt.xlabel('Time (minutes)')
    plt.ylabel('BG (mg/dL)')
    plt.title('2-Hour Glucose Prediction')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_residuals(time_axis, residuals, save_path):
    plt.figure(figsize=(10,3))
    plt.plot(time_axis, residuals, 'k-', linewidth=0.8)
    plt.axhline(0, color='r', linestyle='--', linewidth=0.8)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Residual (mg/dL)')
    plt.title('Prediction Residuals')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_error_growth(horizons, errors, save_path):
    plt.figure(figsize=(8,4))
    plt.plot(horizons, errors, 'o-')
    plt.xlabel('Prediction horizon (steps)')
    plt.ylabel('RMSE (mg/dL)')
    plt.title('Error growth')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ----------------------------------------------------------------------
#  Main pipeline
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Determine project root robustly (works in script and Jupyter)
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ not defined (e.g., in Jupyter)
        project_root = os.getcwd()

    # If script is inside notebooks folder, move up one level
    if os.path.basename(project_root) == "notebooks":
        project_root = os.path.dirname(project_root)
        

    # Resolve directories (handle absolute/relative paths)
    if not os.path.isabs(args.data_dir):
        data_dir = os.path.join(project_root, args.data_dir)
    else:
        data_dir = args.data_dir

    if not os.path.isabs(args.results_dir):
        results_dir = os.path.join(project_root, args.results_dir)
    else:
        results_dir = args.results_dir

    if not os.path.isabs(args.models_dir):
        models_dir = os.path.join(project_root, args.models_dir)
    else:
        models_dir = args.models_dir

    # Create output directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("Stability-Constrained eDMDc Pipeline")
    print("=" * 60)
    print(f"Patient: {args.patient}")
    print(f"Delays: {args.delays}")
    print(f"Polynomial order: {args.poly_order}")
    print(f"RBF: {args.use_rbf}")
    print(f"Ridge lambda: {args.ridge_lambda}")
    print(f"Prediction horizon: {args.prediction_horizon} steps ({args.prediction_horizon*args.dt:.0f} min)")
    print("-" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    BG, insulin, meal, t_minutes = load_patient_data(args.patient, data_dir, args.dt)

    T_total = len(BG)
    print(f"Total samples: {T_total}")

    # ------------------------------------------------------------------
    # 2. Chronological split (train/val/test)
    # ------------------------------------------------------------------
    train_end = int(args.train_frac * T_total)
    val_end = train_end + int(args.val_frac * T_total)
    if val_end > T_total:
        val_end = T_total

    BG_train = BG[:train_end]
    BG_val   = BG[train_end:val_end]
    BG_test  = BG[val_end:]

    ins_train = insulin[:train_end]
    ins_val   = insulin[train_end:val_end]
    ins_test  = insulin[val_end:]

    meal_train = meal[:train_end]
    meal_val   = meal[train_end:val_end]
    meal_test  = meal[val_end:]

    t_train = t_minutes[:train_end]
    t_val   = t_minutes[train_end:val_end]
    t_test  = t_minutes[val_end:]

    print(f"Train: {len(BG_train)} samples")
    print(f"Validation: {len(BG_val)} samples")
    print(f"Test: {len(BG_test)} samples")

    # ------------------------------------------------------------------
    # 3. Normalize each variable separately using training statistics
    # ------------------------------------------------------------------
    train_stack = np.column_stack([BG_train, ins_train, meal_train])
    val_stack   = np.column_stack([BG_val, ins_val, meal_val])
    test_stack  = np.column_stack([BG_test, ins_test, meal_test])

    train_norm, val_norm, test_norm, means, stds = normalize_train_val_test(
        train_stack, val_stack, test_stack
    )
    mean_g, mean_i, mean_m = means
    std_g, std_i, std_m = stds

    BG_train_n, ins_train_n, meal_train_n = train_norm[:,0], train_norm[:,1], train_norm[:,2]
    BG_val_n,   ins_val_n,   meal_val_n   = val_norm[:,0],   val_norm[:,1],   val_norm[:,2]
    BG_test_n,  ins_test_n,  meal_test_n  = test_norm[:,0],  test_norm[:,1],  test_norm[:,2]

    # ------------------------------------------------------------------
    # 4. Delay embedding
    # ------------------------------------------------------------------
    def build_raw_state(BG_n, ins_n, meal_n, d):
        Xg = delay_embed(BG_n, d)
        Xi = delay_embed(ins_n, d)
        Xm = delay_embed(meal_n, d)
        # They should have same number of columns
        N = Xg.shape[1]
        Z = np.vstack([Xg, Xi, Xm])   # shape (3*d, N)
        return Z

    Z_train = build_raw_state(BG_train_n, ins_train_n, meal_train_n, args.delays)
    Z_val   = build_raw_state(BG_val_n,   ins_val_n,   meal_val_n,   args.delays)
    Z_test  = build_raw_state(BG_test_n,  ins_test_n,  meal_test_n,  args.delays)

    # Control inputs for the same snapshots (start at index d-1)
    def build_U(ins_n, meal_n, d):
        start = d - 1
        U = np.vstack([ins_n[start:], meal_n[start:]])  # (2, T - start)
        return U

    U_train = build_U(ins_train_n, meal_train_n, args.delays)
    U_val   = build_U(ins_val_n,   meal_val_n,   args.delays)
    U_test  = build_U(ins_test_n,  meal_test_n,  args.delays)

    # ------------------------------------------------------------------
    # 5. Nonlinear lifting
    # ------------------------------------------------------------------
    # Polynomial features
    if args.poly_order >= 1:
        Phi_train = build_polynomial_features(Z_train, args.poly_order)
        Phi_val   = build_polynomial_features(Z_val,   args.poly_order)
        Phi_test  = build_polynomial_features(Z_test,  args.poly_order)
    else:
        # No polynomial (identity only)
        Phi_train = Z_train
        Phi_val   = Z_val
        Phi_test  = Z_test

    # RBF features (optional)
    if args.use_rbf:
        # Compute centers from training Z (columns = samples)
        Z_train_T = Z_train.T
        centers = rbf_centers_from_data(Z_train_T, args.rbf_centers)
        gamma = 1.0  # could be tuned
        rbf_train = rbf_features(Z_train, centers, gamma)
        rbf_val   = rbf_features(Z_val,   centers, gamma)
        rbf_test  = rbf_features(Z_test,  centers, gamma)
        # Append to Phi
        Phi_train = np.vstack([Phi_train, rbf_train])
        Phi_val   = np.vstack([Phi_val,   rbf_val])
        Phi_test  = np.vstack([Phi_test,  rbf_test])

    n_features = Phi_train.shape[0]
    print(f"Lifted feature dimension: {n_features}")

    # ------------------------------------------------------------------
    # 6. Build eDMDc matrices (X, X', U)
    # ------------------------------------------------------------------
    X_tr, Xpr_tr, U_tr = build_edmdc_matrices(Phi_train, U_train)
    X_val, Xpr_val, U_val_m = build_edmdc_matrices(Phi_val, U_val)
    X_te, Xpr_te, U_te = build_edmdc_matrices(Phi_test, U_test)

    # Number of snapshots (pairs)
    N_tr = X_tr.shape[1]
    N_val = X_val.shape[1]
    N_te = X_te.shape[1]
    print(f"Snapshots: train={N_tr}, val={N_val}, test={N_te}")

    # ------------------------------------------------------------------
    # 7. Ridge model (full rank)
    # ------------------------------------------------------------------
    print("\n--- Training Ridge model ---")
    A_ridge, B_ridge = train_ridge(X_tr, Xpr_tr, U_tr, args.ridge_lambda)

    # ------------------------------------------------------------------
    # 8. Rank selection via truncated SVD on validation set
    # ------------------------------------------------------------------
    # Determine candidate ranks
    max_possible_rank = min(n_features + U_tr.shape[0], N_tr)
    if args.max_rank is None:
        max_rank = max_possible_rank
    else:
        max_rank = min(args.max_rank, max_possible_rank)
    rank_list = list(range(2, max_rank+1))   # start from 2 (rank 1 usually too simple)
    print(f"\n--- Selecting rank via validation (ranks 2-{max_rank}) ---")
    horizons = (10, 20, 40)  # steps
    best_rank, A_svd, B_svd, best_val_score = select_rank_by_validation(
        X_tr, Xpr_tr, U_tr,
        X_val, Xpr_val, U_val_m,
        mean_g, std_g,
        rank_list, prediction_horizons=horizons, method='svd'
    )
    print(f"Best rank: {best_rank} with average validation RMSE = {best_val_score:.2f} mg/dL")

    # ------------------------------------------------------------------
    # 9. Stability enforcement on best model
    # ------------------------------------------------------------------
    print("\n--- Stability check ---")
    A_stable, clipped = enforce_stability(A_svd, tol=0.99)
    if clipped:
        print(f"Stabilized {len(clipped)} unstable modes (indices {clipped})")
    else:
        print("All eigenvalues are stable.")

    # Evaluate on test set before and after stabilization
    phi0_test = X_te[:, 0]
    steps_test = min(args.prediction_horizon, X_te.shape[1] - 1)
    U_seq_test = U_te[:, :steps_test]
    true_norm = np.concatenate([[X_te[0,0]], Xpr_te[0, :steps_test]])
    true_glucose = true_norm * std_g + mean_g

    pred_svd = predict_multi_step(A_svd, B_svd, phi0_test, U_seq_test, mean_g, std_g, clip=True)
    pred_stable = predict_multi_step(A_stable, B_svd, phi0_test, U_seq_test, mean_g, std_g, clip=True)

    rmse_svd, mae_svd, nrmse_svd = compute_metrics(true_glucose, pred_svd)
    rmse_stable, mae_stable, nrmse_stable = compute_metrics(true_glucose, pred_stable)

    print("\n--- Test Set Results (2-hour prediction) ---")
    print(f"SVD (rank={best_rank}): RMSE = {rmse_svd:.2f} mg/dL, MAE = {mae_svd:.2f}, NRMSE = {nrmse_svd:.3f}")
    print(f"Stabilized:           RMSE = {rmse_stable:.2f} mg/dL, MAE = {mae_stable:.2f}, NRMSE = {nrmse_stable:.3f}")

    # Use stabilized model for further analysis if it improved or not worse
    if rmse_stable <= rmse_svd * 1.05:   # within 5%
        print("Using stabilized model for final.")
        A_final, B_final = A_stable, B_svd
    else:
        print("Stabilization degraded performance; using original SVD model.")
        A_final, B_final = A_svd, B_svd

    # ------------------------------------------------------------------
    # 10. Eigenanalysis of final model
    # ------------------------------------------------------------------
    eigvals = np.linalg.eigvals(A_final)
    dt = args.dt
    tau = -dt / np.log(np.abs(eigvals))
    print("\n--- Eigenvalues of Final Model ---")
    stable = np.abs(eigvals) < 1.0
    # Sort by magnitude descending
    order = np.argsort(np.abs(eigvals))[::-1]
    for i in order[:20]:   # top 20
        if stable[i]:
            print(f"  lambda = {eigvals[i].real:.4f} + {eigvals[i].imag:.4f}j, |lambda|={np.abs(eigvals[i]):.4f}, tau={tau[i]:.2f} min")
        else:
            print(f"  lambda = {eigvals[i].real:.4f} + {eigvals[i].imag:.4f}j, |lambda|={np.abs(eigvals[i]):.4f} (UNSTABLE)")

    # ------------------------------------------------------------------
    # 11. Generate and save plots
    # ------------------------------------------------------------------
    # Singular values of Omega_train
    Omega_full = np.vstack([X_tr, U_tr])
    _, s, _ = svd(Omega_full, full_matrices=False)
    plot_singular_values(s, os.path.join(results_dir, f"{args.patient}_singular_values.png"))

    # Eigenvalues
    plot_eigenvalues(A_final, os.path.join(results_dir, f"{args.patient}_eigenvalues.png"))

    # Prediction plot
    time_axis = t_test[0] + np.arange(steps_test+1) * dt
    plot_predictions(time_axis, true_glucose, pred_stable,
                     os.path.join(results_dir, f"{args.patient}_prediction.png"))

    # Residuals
    residuals = true_glucose - pred_stable
    plot_residuals(time_axis, residuals,
                   os.path.join(results_dir, f"{args.patient}_residuals.png"))

    # Error growth (compute for multiple horizons)
    horizons = [10, 20, 30, 40, 50]
    errors = []
    for H in horizons:
        if H >= steps_test:
            break 
        U_seq = U_te[:, :H]
        pred = predict_multi_step(A_final, B_final, phi0_test, U_seq, mean_g, std_g, clip=True)
        true = true_glucose[:H+1]
        rmse = compute_metrics(true, pred)[0]
        errors.append(rmse)
    plot_error_growth(horizons[:len(errors)], errors,
                      os.path.join(results_dir, f"{args.patient}_error_growth.png"))

    print(f"\nPlots saved to {results_dir}/")

    # ------------------------------------------------------------------
    # 12. Save model
    # ------------------------------------------------------------------
    model_path = os.path.join(models_dir, f"edmdc_{args.patient}.npz")
    np.savez(model_path,
             A=A_final,
             B=B_final,
             mean_g=mean_g, std_g=std_g,
             mean_i=mean_i, std_i=std_i,
             mean_m=mean_m, std_m=std_m,
             delays=args.delays,
             dt=args.dt,
             poly_order=args.poly_order,
             use_rbf=args.use_rbf,
             best_rank=best_rank,
             ridge_lambda=args.ridge_lambda,
             rmse_test=rmse_stable)
    print(f"Model saved to {model_path}")

    # ------------------------------------------------------------------
    # 13. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Patient: {args.patient}")
    print(f"Data: {T_total} samples, train={len(BG_train)}, val={len(BG_val)}, test={len(BG_test)}")
    print(f"Delays: {args.delays}")
    print(f"Lifted feature dimension: {n_features}")
    print(f"Best rank (SVD): {best_rank}")
    print(f"Test RMSE (stabilized): {rmse_stable:.2f} mg/dL")
    print(f"Test MAE: {mae_stable:.2f} mg/dL")
    print(f"Test NRMSE: {nrmse_stable:.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main()