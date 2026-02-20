#!/usr/bin/env python3
"""
Stability-Constrained eDMDc Pipeline for Glucose-Insulin System

This script implements a complete pipeline for learning a control-oriented
surrogate model of the glucose-insulin system using Extended Dynamic Mode
Decomposition with control (eDMDc).  It includes:

- Data loading, cleaning, and chronological splitting
- Delay embedding and physiologically‑informed nonlinear lifting
- Ridge and truncated-SVD identification with automatic rank selection
- Stability enforcement by uniform spectral radius capping
- Multi-step prediction and comprehensive evaluation
- Publication-ready plots and model saving

Usage example:
    python edmdc_pipeline.py --patient adolescent_001 --delays 4
"""

import os
import sys
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import svd, solve, eig, inv, norm, solve_discrete_lyapunov

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
                        help='Ignored – kept for compatibility')
    parser.add_argument('--use_rbf', action='store_true',
                        help='Add RBF features (experimental)')
    parser.add_argument('--rbf_centers', type=int, default=5,
                        help='Number of RBF centers per dimension')
    parser.add_argument('--ridge_lambda', type=float, default=0.05,
                        help='Ridge regularisation parameter (default 0.05)')
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
    # New arguments for upgraded pipeline
    parser.add_argument('--identity_only', action='store_true',
                        help='Use only delay embedding (no nonlinear terms)')
    parser.add_argument('--stability_tol', type=float, default=0.99,
                        help='Target spectral radius after stabilisation (unused now)')
    parser.add_argument('--validation_stride', type=int, default=1,
                        help='Stride for rolling validation windows (1 = use every start)')
    # Additional experimental arguments
    parser.add_argument('--fixed_rank', type=int, default=None,
                        help='If set, skip rank selection and use this rank for SVD model.')
    parser.add_argument('--spectral_radius_cap', type=float, default=None,
                        help='If set, uniformly scale A so that spectral radius <= cap.')

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

    essential = ['BG', 'insulin', 'meal']
    for col in essential:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()
    if df[essential].isnull().any().any():
        raise ValueError(f"NaNs remain in patient {patient} after filling.")

    if 't_minutes' in df.columns:
        t = df['t_minutes'].values
    else:
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
#  Physiologically‑informed nonlinear dictionary
# ----------------------------------------------------------------------
def build_physiological_features(Z):
    """
    Z: raw state matrix (n_raw, N) where n_raw = 3*d (G_delays, I_delays, M_delays stacked)
    Returns lifted matrix (n_features, N) with:
    - identity: Z (all delays)
    - for each delay k: G_k * I_k
    - for each delay k: G_k * M_k
    - tanh(I_k)  (insulin saturation)
    """
    n_raw, N = Z.shape
    d = n_raw // 3                     # number of delays per variable
    G = Z[:d, :]                       # glucose delays
    I = Z[d:2*d, :]                     # insulin delays
    M = Z[2*d:3*d, :]                   # meal delays

    features = [Z]                      # identity

    # Glucose‑insulin interaction per delay
    GI = G * I
    features.append(GI)

    # Glucose‑meal interaction per delay
    GM = G * M
    features.append(GM)

    # Insulin saturation (mild nonlinearity)
    tanh_I = np.tanh(I)
    features.append(tanh_I)

    return np.vstack(features)

# ----------------------------------------------------------------------
#  RBF features (kept optional)
# ----------------------------------------------------------------------
def rbf_centers_from_data(data, n_centers):
    centers = []
    for col in range(data.shape[1]):
        q = np.linspace(0, 100, n_centers+2)[1:-1]
        cents = np.percentile(data[:, col], q)
        centers.append(cents)
    return np.array(centers)

def rbf_features(Z, centers, gamma=1.0):
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
    X = Phi[:, :-1]
    Xprime = Phi[:, 1:]
    Umat = U[:, :-1]
    return X, Xprime, Umat

# ----------------------------------------------------------------------
#  Identification methods
# ----------------------------------------------------------------------
def train_ridge(X, Xprime, U, lam):
    n = X.shape[0]
    m = U.shape[0]
    Omega = np.vstack([X, U])
    M = Xprime @ Omega.T
    Gram = Omega @ Omega.T + lam * np.eye(n + m)
    Y = solve(Gram, M.T, assume_a='pos')
    AB = Y.T
    A = AB[:, :n]
    B = AB[:, n:]
    return A, B

def train_truncated_svd(X, Xprime, U, rank):
    Omega = np.vstack([X, U])
    U1, s, Vh = svd(Omega, full_matrices=False)
    if rank == 'full' or rank >= len(s):
        r = len(s)
    else:
        r = int(rank)
    Omega_pinv = Vh[:r, :].T @ np.diag(1.0 / s[:r]) @ U1[:, :r].T
    AB = Xprime @ Omega_pinv
    n = X.shape[0]
    A = AB[:, :n]
    B = AB[:, n:]
    return A, B

# ----------------------------------------------------------------------
#  Stability‑aware utilities
# ----------------------------------------------------------------------
def spectral_radius(A):
    return np.max(np.abs(np.linalg.eigvals(A)))

def stabilise_model(A, B, X_val, Xpr_val, U_val, mean_g, std_g, horizons,
                    factors=(0.95, 0.99, 0.999), stride=1):
    eigvals, V = eig(A)
    best_A = A
    best_score = np.inf

    if spectral_radius(A) <= 1.0 + 1e-12:
        factors = list(factors) + [1.0]
    else:
        factors = list(factors)

    for factor in factors:
        if factor >= 1.0 and spectral_radius(A) <= 1.0 + 1e-12:
            A_test = A
        else:
            unstable = np.abs(eigvals) > 1.0
            if not np.any(unstable):
                A_test = A
            else:
                eigvals_stable = eigvals.copy()
                for i in np.where(unstable)[0]:
                    eigvals_stable[i] = factor * eigvals[i] / np.abs(eigvals[i])
                A_test = V @ np.diag(eigvals_stable) @ inv(V)
                A_test = np.real(A_test)

        if spectral_radius(A_test) > 1.0 + 1e-6:
            continue

        score = evaluate_model_rolling(A_test, B, X_val, Xpr_val, U_val,
                                       mean_g, std_g, horizons, stride=stride)
        if score < best_score:
            best_score = score
            best_A = A_test

    return best_A, best_score

def evaluate_model_rolling(A, B, X, Xpr, U, mean_g, std_g, horizons, stride=1):
    N = X.shape[1]
    max_horizon = max(horizons)
    if N <= max_horizon:
        starts = [0]
    else:
        starts = list(range(0, N - max_horizon, stride))
        if (N - max_horizon - 1) % stride != 0:
            starts.append(N - max_horizon - 1)

    horizon_scores = {h: [] for h in horizons}
    for start in starts:
        phi0 = X[:, start]
        for H in horizons:
            if start + H >= N:
                continue
            U_seq = U[:, start:start+H]
            pred = predict_multi_step(A, B, phi0, U_seq, mean_g, std_g)
            true_norm = np.concatenate([[X[0, start]], Xpr[0, start:start+H]])
            true = true_norm * std_g + mean_g
            rmse = np.sqrt(np.mean((true - pred)**2))
            horizon_scores[H].append(rmse)

    avg_per_horizon = [np.mean(horizon_scores[H]) for H in horizons if horizon_scores[H]]
    return np.mean(avg_per_horizon) if avg_per_horizon else np.inf

# ----------------------------------------------------------------------
#  Multi-step prediction (NO CLIPPING)
# ----------------------------------------------------------------------
def predict_multi_step(A, B, phi0, U_seq, mean_g, std_g):
    steps = U_seq.shape[1]
    n = A.shape[0]
    Phi_pred = np.zeros((n, steps + 1))
    Phi_pred[:, 0] = phi0
    for t in range(steps):
        Phi_pred[:, t+1] = A @ Phi_pred[:, t] + B @ U_seq[:, t]
    glucose_norm = Phi_pred[0, :]
    glucose = glucose_norm * std_g + mean_g
    return glucose

def compute_metrics(true, pred):
    rmse = np.sqrt(np.mean((true - pred)**2))
    mae = np.mean(np.abs(true - pred))
    nrmse = rmse / (np.max(true) - np.min(true)) if np.max(true) != np.min(true) else np.nan
    return rmse, mae, nrmse

# ----------------------------------------------------------------------
#  Rank selection via validation with stabilisation
# ----------------------------------------------------------------------
def select_rank_by_validation(X_tr, Xpr_tr, U_tr,
                              X_val, Xpr_val, U_val,
                              mean_g, std_g,
                              rank_list, prediction_horizons=(10,20,40),
                              method='svd', stability_factors=(0.95,0.99,0.999),
                              validation_stride=1):
    best_rank = None
    best_score = np.inf
    best_A = None
    best_B = None

    for rank in rank_list:
        if method == 'svd':
            A, B = train_truncated_svd(X_tr, Xpr_tr, U_tr, rank)
        else:
            raise ValueError("Only 'svd' method supported for rank selection")

        A_stab, score = stabilise_model(A, B, X_val, Xpr_val, U_val,
                                        mean_g, std_g, prediction_horizons,
                                        factors=stability_factors,
                                        stride=validation_stride)

        if score < best_score:
            best_score = score
            best_rank = rank
            best_A = A_stab
            best_B = B

    if best_rank is None and rank_list:
        rank = rank_list[0]
        A, B = train_truncated_svd(X_tr, Xpr_tr, U_tr, rank)
        best_A, best_score = stabilise_model(A, B, X_val, Xpr_val, U_val,
                                             mean_g, std_g, prediction_horizons,
                                             factors=stability_factors,
                                             stride=validation_stride)
        best_B = B
        best_rank = rank

    return best_rank, best_A, best_B, best_score

# ----------------------------------------------------------------------
#  Helper to print singular values of A
# ----------------------------------------------------------------------
def print_singular_values_of_A(A):
    s = np.linalg.svd(A, compute_uv=False)
    s_sorted = np.sort(s)[::-1]
    print("\n--- Singular values of A ---")
    print(f"Top 5: {s_sorted[:5]}")
    print(f"Bottom 5: {s_sorted[-5:] if len(s_sorted) >= 5 else s_sorted}")

# ----------------------------------------------------------------------
#  Controllability analysis (enhanced)
# ----------------------------------------------------------------------
def controllability_analysis(A, B):
    n = A.shape[0]
    m = B.shape[1]
    C = B
    Ai = np.eye(n)
    for i in range(1, n):
        Ai = Ai @ A
        C = np.hstack((C, Ai @ B))
    rank = np.linalg.matrix_rank(C)
    cond = np.linalg.cond(C)
    s = np.linalg.svd(C, compute_uv=False)
    s_sorted = np.sort(s)[::-1]
    print("\n--- Controllability Analysis ---")
    print(f"Controllability matrix rank: {rank} / {n} (full rank = {rank == n})")
    print(f"Condition number: {cond:.2e}")
    print(f"Smallest 5 singular values: {s_sorted[-5:] if len(s_sorted)>=5 else s_sorted}")
    print(f"Ratio σ_min/σ_max: {s_sorted[-1]/s_sorted[0]:.2e}")
    if rank == n:
        print("System is fully controllable. Glucose state (first coordinate) is controllable.")
    else:
        e1 = np.zeros(n)
        e1[0] = 1.0
        x, residuals, _, _ = np.linalg.lstsq(C, e1, rcond=None)
        e1_proj = C @ x
        error = norm(e1 - e1_proj)
        if error < 1e-6:
            print("Glucose state (first coordinate) lies in controllable subspace.")
        else:
            print("Glucose state (first coordinate) may not be fully controllable.")
    return rank, cond, s

# ----------------------------------------------------------------------
#  Spectral diagnostics
# ----------------------------------------------------------------------
def spectral_diagnostics(A, dt):
    eigvals = np.linalg.eigvals(A)
    sr = np.max(np.abs(eigvals))
    unstable = np.abs(eigvals) > 1.0 + 1e-12
    n_unstable = np.sum(unstable)
    print("\n--- Spectral Diagnostics ---")
    print(f"Spectral radius: {sr:.6f}")
    print(f"Number of unstable modes: {n_unstable}")

    if n_unstable == 0:
        order = np.argsort(np.abs(eigvals))[::-1]
        print("\nDominant modes (top 5):")
        for i in order[:5]:
            lam = eigvals[i]
            mag = np.abs(lam)
            tau = -dt / np.log(mag) if mag < 1.0 else np.inf
            if np.imag(lam) != 0:
                p = np.log(lam) / dt
                zeta = -np.real(p) / np.abs(p)
                print(f"  λ = {lam.real:.4f} + {lam.imag:.4f}j, |λ|={mag:.4f}, τ={tau:.2f} min, ζ={zeta:.4f}")
            else:
                print(f"  λ = {lam.real:.4f}, |λ|={mag:.4f}, τ={tau:.2f} min")

# ----------------------------------------------------------------------
#  Modal Controllability Analysis
# ----------------------------------------------------------------------
def modal_controllability_analysis(A, B, dt):
    """
    Compute modal controllability measures:
    For each eigenmode i, m_i = || V_inv[i, :] @ B ||_2
    Prints sorted by |λ|, flags weakly controllable slow modes (|λ|>0.95 and m_i<1e-6).
    """
    eigvals, V = np.linalg.eig(A)
    V_inv = np.linalg.inv(V)

    n = A.shape[0]
    m_vals = np.zeros(n)
    for i in range(n):
        m_vals[i] = norm(V_inv[i, :] @ B)

    # Sort by |λ| descending
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals_sorted = eigvals[order]
    m_sorted = m_vals[order]

    print("\n" + "-" * 40)
    print("MODAL CONTROLLABILITY ANALYSIS")
    print("-" * 40)

    # Print table header
    print(f"{'Idx':>4} | {'λ (real)':>10} {'λ (imag)':>10} {'|λ|':>8} | {'Modal contr.':>12}")
    print("-" * 60)

    for idx_in_list, i in enumerate(order):
        lam = eigvals[i]
        mag = np.abs(lam)
        print(f"{idx_in_list:4d} | {lam.real:10.4f} {lam.imag:10.4f} {mag:8.4f} | {m_vals[i]:12.2e}")

    # Identify weakly controllable slow modes
    weak_slow = []
    for i in range(n):
        mag = np.abs(eigvals[i])
        if mag > 0.95 and m_vals[i] < 1e-6:
            weak_slow.append((i, eigvals[i], mag, m_vals[i]))

    if weak_slow:
        print("\n--- Weakly Controllable Slow Modes ---")
        for i, lam, mag, m in weak_slow:
            print(f"Mode {i}: λ={lam.real:.4f}+{lam.imag:.4f}j, |λ|={mag:.4f}, modal contr.={m:.2e}")
    else:
        print("\n--- No weakly controllable slow modes detected (|λ|>0.95 and m<1e-6) ---")

    # Summary statistics
    print("\nModal controllability statistics:")
    print(f"  Max: {m_vals.max():.2e}")
    print(f"  Min: {m_vals.min():.2e}")
    print(f"  Ratio (min/max): {m_vals.min()/m_vals.max():.2e}")

# ----------------------------------------------------------------------
#  NEW: Balanced Truncation Analysis (Control-aware reduction)
# ----------------------------------------------------------------------
def balanced_truncation_analysis(A, B, mean_g, std_g, phi0_test, U_te, X_te, Xpr_te, t_test, dt):
    """
    Perform balanced truncation to obtain a reduced-order model.
    Uses output matrix C = e1^T (selects glucose state).
    Prints Hankel singular values, selects reduced order by 99.5% energy,
    constructs reduced model, and evaluates its prediction performance.
    """
    print("\n" + "=" * 60)
    print("BALANCED TRUNCATION ANALYSIS")
    print("=" * 60)

    n = A.shape[0]
    # Output matrix: select glucose state (first coordinate)
    C = np.zeros((1, n))
    C[0, 0] = 1.0

    # Compute controllability Gramian Wc
    try:
        Wc = solve_discrete_lyapunov(A, B @ B.T)
    except Exception as e:
        print(f"WARNING: Could not solve controllability Lyapunov equation: {e}")
        print("Skipping balanced truncation.")
        return

    # Compute observability Gramian Wo
    try:
        Wo = solve_discrete_lyapunov(A.T, C.T @ C)
    except Exception as e:
        print(f"WARNING: Could not solve observability Lyapunov equation: {e}")
        print("Skipping balanced truncation.")
        return

    # Compute product and Hankel singular values
    M = Wc @ Wo
    eig_bal = np.linalg.eigvals(M)
    # Ensure real and non-negative (should be, but take sqrt of absolute to be safe)
    hsv = np.sqrt(np.abs(np.real(eig_bal)))
    hsv = np.sort(hsv)[::-1]  # descending

    print("\n--- Hankel Singular Values ---")
    print(f"Top 10: {hsv[:10]}")
    print(f"Bottom 5: {hsv[-5:] if len(hsv) >= 5 else hsv}")
    print(f"Ratio min/max: {hsv[-1]/hsv[0]:.2e}")

    # Cumulative energy and order selection (99.5% threshold)
    energy_cum = np.cumsum(hsv) / np.sum(hsv)
    r = np.searchsorted(energy_cum, 0.995) + 1
    r = min(r, n)  # ensure not exceeding n
    print(f"\nSelected reduced order (99.5% energy): r = {r}")

    # Balancing transformation via SVD
    # Compute square roots (Cholesky might fail if Gramians are ill-conditioned; use sqrtm as fallback)
    try:
        Lc = np.linalg.cholesky(Wc)
        Lo = np.linalg.cholesky(Wo)
    except np.linalg.LinAlgError:
        from scipy.linalg import sqrtm
        Lc = np.real(sqrtm(Wc))
        Lo = np.real(sqrtm(Wo))

    # Compute SVD of Lo @ Lc
    Ubal, s_bal, Vbal = svd(Lo.T @ Lc, full_matrices=False)

    # Balancing transformation
    T = Lc @ Vbal.T @ np.diag(1.0 / np.sqrt(s_bal))
    T_inv = np.diag(1.0 / np.sqrt(s_bal)) @ Ubal.T @ Lo.T

    # Transform and truncate
    A_bal = T_inv @ A @ T
    B_bal = T_inv @ B
    C_bal = C @ T

    # Truncate to r
    A_bal = A_bal[:r, :r]
    B_bal = B_bal[:r, :]
    C_bal = C_bal[:, :r]

    print(f"\nReduced model size: {r}")

    # Evaluate reduced model predictions for 2h, 6h, 12h
    print("\n--- Reduced Model Performance ---")

    # Prepare test data
    steps_2h = int(120 / dt)
    steps_6h = int(360 / dt)
    steps_12h = int(720 / dt)
    horizons = [steps_2h, steps_6h, steps_12h]
    labels = ['2h', '6h', '12h']
    rmse_red = []

    for steps, label in zip(horizons, labels):
        if steps <= X_te.shape[1] - 1:
            true_norm_horizon = np.concatenate([[X_te[0,0]], Xpr_te[0, :steps]])
            true = true_norm_horizon * std_g + mean_g
            U_seq = U_te[:, :steps]
            # For reduced model, initial lifted state phi0_test must be transformed to balanced coordinates
            # phi0_bal = T_inv @ phi0_test, but T_inv is full n x n, so we take first r rows after truncation
            # However, T_inv is n x n; we need to project phi0_test onto balanced coordinates.
            # Instead, use the full T_inv and then truncate to r rows.
            phi0_bal = T_inv @ phi0_test
            phi0_bal_r = phi0_bal[:r]
            pred_red = predict_multi_step(A_bal, B_bal, phi0_bal_r, U_seq, mean_g, std_g)
            rmse = compute_metrics(true, pred_red)[0]
            rmse_red.append(rmse)
            print(f"  Reduced model {label} RMSE: {rmse:.2f} mg/dL")
        else:
            print(f"  Not enough data for {label} prediction")
            rmse_red.append(np.nan)

    # Compute spectral radius and controllability of reduced model
    sr_red = spectral_radius(A_bal)
    print(f"  Reduced model spectral radius: {sr_red:.4f}")
    # Controllability analysis for reduced model
    controllability_analysis(A_bal, B_bal)   # this will print its own section

    print("\n--- Comparison with Full Model ---")
    # Full model RMSEs from earlier (we have rmse_final, but need 6h and 12h; we'll compute them now)
    # Compute full model predictions for the same horizons
    rmse_full = []
    for steps, label in zip(horizons, labels):
        if steps <= X_te.shape[1] - 1:
            true_norm_horizon = np.concatenate([[X_te[0,0]], Xpr_te[0, :steps]])
            true = true_norm_horizon * std_g + mean_g
            U_seq = U_te[:, :steps]
            pred_full = predict_multi_step(A, B, phi0_test, U_seq, mean_g, std_g)
            rmse = compute_metrics(true, pred_full)[0]
            rmse_full.append(rmse)
        else:
            rmse_full.append(np.nan)

    for i, label in enumerate(labels):
        if not np.isnan(rmse_full[i]) and not np.isnan(rmse_red[i]):
            print(f"  {label} RMSE: full={rmse_full[i]:.2f}, reduced={rmse_red[i]:.2f}")
    print(f"  Spectral radius: full={spectral_radius(A):.4f}, reduced={sr_red:.4f}")

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
    plt.title('Glucose Prediction')
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

def plot_long_horizon_errors(horizon_labels, errors, save_path):
    plt.figure(figsize=(6,4))
    plt.bar(horizon_labels, errors, color='steelblue')
    plt.ylabel('RMSE (mg/dL)')
    plt.title('Long‑horizon prediction error')
    plt.grid(alpha=0.3, axis='y')
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
        project_root = os.getcwd()
    if os.path.basename(project_root) == "notebooks":
        project_root = os.path.dirname(project_root)

    data_dir = os.path.join(project_root, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    results_dir = os.path.join(project_root, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir
    models_dir = os.path.join(project_root, args.models_dir) if not os.path.isabs(args.models_dir) else args.models_dir

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("Stability-Constrained eDMDc Pipeline")
    print("=" * 60)
    print(f"Patient: {args.patient}")
    print(f"Delays: {args.delays}")
    print(f"Physiological dictionary: {'disabled' if args.identity_only else 'enabled'}")
    print(f"RBF: {args.use_rbf and not args.identity_only}")
    print(f"Identity only: {args.identity_only}")
    print(f"Ridge lambda: {args.ridge_lambda}")
    print(f"Prediction horizon: {args.prediction_horizon} steps ({args.prediction_horizon*args.dt:.0f} min)")
    if args.fixed_rank:
        print(f"Fixed rank (overrides selection): {args.fixed_rank}")
    if args.spectral_radius_cap:
        print(f"Spectral radius cap: {args.spectral_radius_cap}")
    print("-" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    BG, insulin, meal, t_minutes = load_patient_data(args.patient, data_dir, args.dt)
    T_total = len(BG)
    print(f"Total samples: {T_total}")

    # ------------------------------------------------------------------
    # 2. Chronological split
    # ------------------------------------------------------------------
    train_end = int(args.train_frac * T_total)
    val_end = train_end + int(args.val_frac * T_total)
    if val_end > T_total:
        val_end = T_total

    BG_train, BG_val, BG_test = BG[:train_end], BG[train_end:val_end], BG[val_end:]
    ins_train, ins_val, ins_test = insulin[:train_end], insulin[train_end:val_end], insulin[val_end:]
    meal_train, meal_val, meal_test = meal[:train_end], meal[train_end:val_end], meal[val_end:]
    t_train, t_val, t_test = t_minutes[:train_end], t_minutes[train_end:val_end], t_minutes[val_end:]

    print(f"Train: {len(BG_train)} samples")
    print(f"Validation: {len(BG_val)} samples")
    print(f"Test: {len(BG_test)} samples")

    # ------------------------------------------------------------------
    # 3. Normalize
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
        N = Xg.shape[1]
        return np.vstack([Xg, Xi, Xm])

    Z_train = build_raw_state(BG_train_n, ins_train_n, meal_train_n, args.delays)
    Z_val   = build_raw_state(BG_val_n,   ins_val_n,   meal_val_n,   args.delays)
    Z_test  = build_raw_state(BG_test_n,  ins_test_n,  meal_test_n,  args.delays)

    def build_U(ins_n, meal_n, d):
        start = d - 1
        return np.vstack([ins_n[start:], meal_n[start:]])

    U_train = build_U(ins_train_n, meal_train_n, args.delays)
    U_val   = build_U(ins_val_n,   meal_val_n,   args.delays)
    U_test  = build_U(ins_test_n,  meal_test_n,  args.delays)

    # ------------------------------------------------------------------
    # 5. Nonlinear lifting
    # ------------------------------------------------------------------
    if args.identity_only:
        Phi_train, Phi_val, Phi_test = Z_train, Z_val, Z_test
    else:
        # Physiological dictionary
        Phi_train = build_physiological_features(Z_train)
        Phi_val   = build_physiological_features(Z_val)
        Phi_test  = build_physiological_features(Z_test)

        # If RBF requested, add it on top
        if args.use_rbf:
            Z_train_T = Z_train.T
            centers = rbf_centers_from_data(Z_train_T, args.rbf_centers)
            gamma = 1.0
            rbf_train = rbf_features(Z_train, centers, gamma)
            rbf_val   = rbf_features(Z_val,   centers, gamma)
            rbf_test  = rbf_features(Z_test,  centers, gamma)
            Phi_train = np.vstack([Phi_train, rbf_train])
            Phi_val   = np.vstack([Phi_val,   rbf_val])
            Phi_test  = np.vstack([Phi_test,  rbf_test])

    n_features = Phi_train.shape[0]
    print(f"Lifted feature dimension: {n_features}")

    # ------------------------------------------------------------------
    # 6. Build eDMDc matrices
    # ------------------------------------------------------------------
    X_tr, Xpr_tr, U_tr = build_edmdc_matrices(Phi_train, U_train)
    X_val, Xpr_val, U_val_m = build_edmdc_matrices(Phi_val, U_val)
    X_te, Xpr_te, U_te = build_edmdc_matrices(Phi_test, U_test)

    N_tr, N_val, N_te = X_tr.shape[1], X_val.shape[1], X_te.shape[1]
    print(f"Snapshots: train={N_tr}, val={N_val}, test={N_te}")

    # ------------------------------------------------------------------
    # 7. Ridge model (reference)
    # ------------------------------------------------------------------
    print("\n--- Training Ridge model ---")
    A_ridge, B_ridge = train_ridge(X_tr, Xpr_tr, U_tr, args.ridge_lambda)

    # ------------------------------------------------------------------
    # 8. Model identification (fixed rank / validation)
    # ------------------------------------------------------------------
    if args.fixed_rank is not None:
        # Safe rank constraint: rank <= n_features - 2
        max_safe_rank = n_features - 2
        if args.fixed_rank > max_safe_rank:
            print(f"WARNING: Requested fixed rank {args.fixed_rank} exceeds safe limit {max_safe_rank} (feature_dim - 2). Reducing to {max_safe_rank}.")
            rank_used = max_safe_rank
        else:
            rank_used = args.fixed_rank

        print(f"\n--- Fixed rank mode: using rank = {rank_used} ---")
        max_possible_rank = min(n_features + U_tr.shape[0], N_tr)
        if rank_used > max_possible_rank:
            print(f"WARNING: Requested rank {rank_used} exceeds max possible {max_possible_rank}. Using {max_possible_rank}.")
            rank_used = max_possible_rank

        A_svd, B_svd = train_truncated_svd(X_tr, Xpr_tr, U_tr, rank_used)
        best_rank = rank_used
        best_val_score = np.nan
    else:
        # Original rank selection with validation
        max_possible_rank = min(n_features + U_tr.shape[0], N_tr)
        if args.max_rank is None:
            max_rank = max_possible_rank
        else:
            max_rank = min(args.max_rank, max_possible_rank)
        rank_list = list(range(2, max_rank+1))
        print(f"\n--- Selecting rank via validation (ranks 2-{max_rank}) ---")
        horizons = (10, 20, 40)
        best_rank, A_svd, B_svd, best_val_score = select_rank_by_validation(
            X_tr, Xpr_tr, U_tr,
            X_val, Xpr_val, U_val_m,
            mean_g, std_g,
            rank_list, prediction_horizons=horizons, method='svd',
            stability_factors=(0.95, 0.99, 0.999),
            validation_stride=args.validation_stride
        )
        print(f"Best rank: {best_rank} with average validation RMSE = {best_val_score:.2f} mg/dL")

        if best_rank is None:
            print("WARNING: Rank selection did not produce a valid model. Falling back to default rank=10.")
            fallback_rank = min(10, max_possible_rank)
            A_tmp, B_tmp = train_truncated_svd(X_tr, Xpr_tr, U_tr, fallback_rank)
            A_svd, _ = stabilise_model(A_tmp, B_tmp, X_val, Xpr_val, U_val_m,
                                       mean_g, std_g, horizons,
                                       factors=(0.95, 0.99, 0.999),
                                       stride=args.validation_stride)
            B_svd = B_tmp
            best_rank = fallback_rank

    # ------------------------------------------------------------------
    # 9. Print singular values of A (for diagnostics)
    # ------------------------------------------------------------------
    print_singular_values_of_A(A_svd)

    # ------------------------------------------------------------------
    # 10. Optional spectral radius capping
    # ------------------------------------------------------------------
    if args.spectral_radius_cap is not None:
        rho = spectral_radius(A_svd)
        if rho > args.spectral_radius_cap:
            scale = args.spectral_radius_cap / rho
            A_svd = A_svd * scale
            print(f"Applied spectral radius cap: original rho={rho:.4f}, scaled by {scale:.4f}, new rho={spectral_radius(A_svd):.4f}")
        else:
            print(f"Spectral radius ({rho:.4f}) already below cap {args.spectral_radius_cap}, no scaling applied.")

    A_final, B_final = A_svd, B_svd

    # ------------------------------------------------------------------
    # 11. Test set evaluation (2-hour)
    # ------------------------------------------------------------------
    phi0_test = X_te[:, 0]
    steps_test = min(args.prediction_horizon, X_te.shape[1] - 1)
    U_seq_test = U_te[:, :steps_test]
    true_norm = np.concatenate([[X_te[0,0]], Xpr_te[0, :steps_test]])
    true_glucose = true_norm * std_g + mean_g

    pred_final = predict_multi_step(A_final, B_final, phi0_test, U_seq_test, mean_g, std_g)
    rmse_final, mae_final, nrmse_final = compute_metrics(true_glucose, pred_final)

    print("\n--- Test Set Results (2-hour prediction) ---")
    print(f"Stabilised model (rank={best_rank}): RMSE = {rmse_final:.2f} mg/dL, MAE = {mae_final:.2f}, NRMSE = {nrmse_final:.3f}")

    # ------------------------------------------------------------------
    # 12. Controllability analysis
    # ------------------------------------------------------------------
    controllability_analysis(A_final, B_final)

    # ------------------------------------------------------------------
    # 13. Spectral diagnostics
    # ------------------------------------------------------------------
    spectral_diagnostics(A_final, args.dt)

    # ------------------------------------------------------------------
    # 14. Modal Controllability Analysis
    # ------------------------------------------------------------------
    modal_controllability_analysis(A_final, B_final, args.dt)

    # ------------------------------------------------------------------
    # 15. Balanced Truncation Analysis (Control-aware reduction)
    # ------------------------------------------------------------------
    balanced_truncation_analysis(A_final, B_final, mean_g, std_g, phi0_test,
                                 U_te, X_te, Xpr_te, t_test, args.dt)

    # ------------------------------------------------------------------
    # 16. Long‑horizon stress test
    # ------------------------------------------------------------------
    print("\n--- Long‑Horizon Stress Test ---")
    long_horizons_min = [120, 360, 720]
    long_horizons_steps = [int(h/args.dt) for h in long_horizons_min]
    long_rmse = []
    feasible_horizons = []

    for steps, label in zip(long_horizons_steps, ['2h','6h','12h']):
        if steps <= X_te.shape[1] - 1:
            true_norm_horizon = np.concatenate([[X_te[0,0]], Xpr_te[0, :steps]])
            true = true_norm_horizon * std_g + mean_g
            U_seq = U_te[:, :steps]
            pred = predict_multi_step(A_final, B_final, phi0_test, U_seq, mean_g, std_g)
            rmse = compute_metrics(true, pred)[0]
            long_rmse.append(rmse)
            feasible_horizons.append(label)
            print(f"{label} prediction RMSE: {rmse:.2f} mg/dL")
        else:
            print(f"{label} not enough test data (need {steps+1} samples, have {X_te.shape[1]})")

    if long_rmse:
        plot_long_horizon_errors(feasible_horizons, long_rmse,
                                 os.path.join(results_dir, f"{args.patient}_long_horizon_errors.png"))

    # ------------------------------------------------------------------
    # 17. Generate and save standard plots
    # ------------------------------------------------------------------
    Omega_full = np.vstack([X_tr, U_tr])
    _, s, _ = svd(Omega_full, full_matrices=False)
    plot_singular_values(s, os.path.join(results_dir, f"{args.patient}_singular_values.png"))

    plot_eigenvalues(A_final, os.path.join(results_dir, f"{args.patient}_eigenvalues.png"))

    time_axis = t_test[0] + np.arange(steps_test+1) * args.dt
    plot_predictions(time_axis, true_glucose, pred_final,
                     os.path.join(results_dir, f"{args.patient}_prediction.png"))

    residuals = true_glucose - pred_final
    plot_residuals(time_axis, residuals,
                   os.path.join(results_dir, f"{args.patient}_residuals.png"))

    horizons = [10, 20, 30, 40, 50]
    errors = []
    for H in horizons:
        if H >= steps_test:
            break
        U_seq = U_te[:, :H]
        pred = predict_multi_step(A_final, B_final, phi0_test, U_seq, mean_g, std_g)
        true = true_glucose[:H+1]
        rmse = compute_metrics(true, pred)[0]
        errors.append(rmse)
    plot_error_growth(horizons[:len(errors)], errors,
                      os.path.join(results_dir, f"{args.patient}_error_growth.png"))

    print(f"\nPlots saved to {results_dir}/")

    # ------------------------------------------------------------------
    # 18. Save model
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
             rmse_test=rmse_final)
    print(f"Model saved to {model_path}")

    # ------------------------------------------------------------------
    # 19. Comparative Summary Block
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARATIVE MODEL SUMMARY")
    print("=" * 60)
    print(f"Feature dimension:            {n_features}")
    print(f"Rank used:                    {best_rank}")
    print(f"Spectral radius:               {spectral_radius(A_final):.4f}")
    n = A_final.shape[0]
    C = B_final
    Ai = np.eye(n)
    for i in range(1, n):
        Ai = Ai @ A_final
        C = np.hstack((C, Ai @ B_final))
    rank_c = np.linalg.matrix_rank(C)
    cond_c = np.linalg.cond(C)
    s_c = np.linalg.svd(C, compute_uv=False)
    s_sorted = np.sort(s_c)[::-1]
    print(f"Controllability rank:         {rank_c} / {n}")
    print(f"Controllability cond number:  {cond_c:.2e}")
    print(f"Controllability σ_min/σ_max:  {s_sorted[-1]/s_sorted[0]:.2e}")
    print(f"2h RMSE:                       {rmse_final:.2f} mg/dL")
    rmse_6h = long_rmse[1] if len(long_rmse) > 1 else np.nan
    rmse_12h = long_rmse[2] if len(long_rmse) > 2 else np.nan
    print(f"6h RMSE:                       {rmse_6h:.2f} mg/dL" if not np.isnan(rmse_6h) else "6h RMSE:                       N/A")
    print(f"12h RMSE:                      {rmse_12h:.2f} mg/dL" if not np.isnan(rmse_12h) else "12h RMSE:                      N/A")
    print("=" * 60)

if __name__ == "__main__":
    main()