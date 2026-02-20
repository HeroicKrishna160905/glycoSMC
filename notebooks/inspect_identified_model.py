#!/usr/bin/env python3
"""
inspect_identified_model.py

Load a trained Koopman/eDMDc model and visually/numerically inspect its properties.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, logm, svd, norm, inv

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PATIENT = "adolescent_001"          # change as needed
MODEL_PATH = "../models/edmdc_adolescent_001.npz"
DT = 3.0                              # sampling interval in minutes (should match model)

# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

data = np.load(MODEL_PATH)
A = data['A']
B = data['B']
mean_g = data['mean_g']
std_g = data['std_g']
mean_i = data['mean_i']
std_i = data['std_i']
mean_m = data['mean_m']
std_m = data['std_m']
delays = int(data['delays'])
dt = float(data['dt'])
best_rank = int(data['best_rank'])
rmse_test = float(data['rmse_test'])

# C matrix: selects glucose (first lifted state)
n = A.shape[0]
C = np.zeros((1, n))
C[0, 0] = 1.0

print("=" * 60)
print("IDENTIFIED KOOPMAN MODEL INSPECTION")
print("=" * 60)
print(f"Patient:           {PATIENT}")
print(f"Feature dimension: {n}")
print(f"Best rank:         {best_rank}")
print(f"Test RMSE (2h):    {rmse_test:.2f} mg/dL")
print(f"Sampling interval: {dt} min")
print("-" * 60)

# ----------------------------------------------------------------------
# Basic matrix properties
# ----------------------------------------------------------------------
print("\n--- Matrix Shapes ---")
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")

# Spectral radius
eigvals = np.linalg.eigvals(A)
spec_rad = np.max(np.abs(eigvals))
print(f"\n--- Spectral radius ---")
print(f"ρ(A) = {spec_rad:.6f}")

# Rank of A
rank_A = np.linalg.matrix_rank(A)
print(f"\n--- Rank of A ---")
print(f"rank(A) = {rank_A} / {n}")

# Controllability matrix and its properties
def controllability_matrix(A, B):
    n = A.shape[0]
    m = B.shape[1]
    Cmat = B.copy()
    Ai = np.eye(n)
    for i in range(1, n):
        Ai = Ai @ A
        Cmat = np.hstack((Cmat, Ai @ B))
    return Cmat

C_ctrb = controllability_matrix(A, B)
rank_ctrb = np.linalg.matrix_rank(C_ctrb)
cond_ctrb = np.linalg.cond(C_ctrb)
s_ctrb = svd(C_ctrb, compute_uv=False)
s_sorted = np.sort(s_ctrb)[::-1]
print(f"\n--- Controllability ---")
print(f"Controllability matrix shape: {C_ctrb.shape}")
print(f"rank = {rank_ctrb} / {n}  (full rank: {rank_ctrb == n})")
print(f"condition number = {cond_ctrb:.2e}")
print(f"σ_min / σ_max = {s_sorted[-1]/s_sorted[0]:.2e}")

# ----------------------------------------------------------------------
# Plot A matrix heatmap
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.imshow(A, cmap='bwr', aspect='auto', interpolation='none')
plt.colorbar(label='A(i,j)')
plt.title('Identified A Matrix')
plt.xlabel('Column index')
plt.ylabel('Row index')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Plot B matrix heatmap (two columns: insulin, meal)
# ----------------------------------------------------------------------
plt.figure(figsize=(6, 8))
plt.imshow(B, cmap='bwr', aspect='auto', interpolation='none')
plt.colorbar(label='B(i,j)')
plt.title('Input Matrix B (columns: insulin, meal)')
plt.xlabel('Input (0=insulin, 1=meal)')
plt.ylabel('Row index')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Eigenvalue plot
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(np.real(eigvals), np.imag(eigvals), c='b', marker='.', alpha=0.7)
# Unit circle
theta = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1)
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.axis('equal')
plt.xlabel('Real')
plt.ylabel('Imag')
plt.title(f'Eigenvalues of A (ρ={spec_rad:.4f})')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Continuous-time poles and time constants
# ----------------------------------------------------------------------
print("\n--- Continuous-time poles and time constants ---")
print("(Based on transformation s = log(λ) / dt)")
print("For stable modes (|λ|<1) time constant τ = -1/Re(s) minutes")
print("-" * 70)
print(f"{'Idx':>4} | {'λ (real)':>12} {'λ (imag)':>12} {'|λ|':>10} | {'Re(s)':>12} {'Im(s)':>12} {'τ (min)':>10}")
print("-" * 70)

# Sort eigenvalues by descending magnitude
order = np.argsort(np.abs(eigvals))[::-1]
dominant_modes = []
for idx_in_list, i in enumerate(order[:8]):   # top 8 by magnitude
    lam = eigvals[i]
    mag = np.abs(lam)
    s = np.log(lam) / dt   # principal branch; careful with branch cuts
    tau = -1.0 / np.real(s) if (np.real(s) < 0 and mag < 1) else np.inf
    dominant_modes.append((i, lam, mag, s, tau))
    print(f"{i:4d} | {lam.real:12.4f} {lam.imag:12.4f} {mag:10.4f} | {s.real:12.4f} {s.imag:12.4f} {tau:10.2f}")

# ----------------------------------------------------------------------
# Glucose participation in Koopman modes
# ----------------------------------------------------------------------
V = np.linalg.eig(A)[1]   # eigenvectors, columns are right eigenvectors
glucose_part = np.abs(V[0, :])  # first row corresponds to glucose

# Sort by participation
order_part = np.argsort(glucose_part)[::-1]
print("\n--- Top 5 modes by glucose participation ---")
print(f"{'Mode idx':>8} | {'Glucose part.':>14} | {'|λ|':>8}")
print("-" * 40)
for i in order_part[:5]:
    print(f"{i:8d} | {glucose_part[i]:14.4f} | {np.abs(eigvals[i]):8.4f}")

# Stem plot of glucose participation
plt.figure(figsize=(10, 4))
plt.stem(range(n), glucose_part, basefmt=' ')
plt.xlabel('Mode index')
plt.ylabel('|C * eigenvector|')
plt.title('Glucose Participation in Koopman Modes')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Dominant mode time evolution
# ----------------------------------------------------------------------
# Find eigenvalue with largest magnitude (dominant mode)
idx_dom = np.argmax(np.abs(eigvals))
lam_dom = eigvals[idx_dom]
print(f"\n--- Dominant mode (largest |λ| = {np.abs(lam_dom):.4f}) ---")
print(f"Eigenvalue: {lam_dom.real:.4f} + {lam_dom.imag:.4f}j")

# Simulate its evolution in lifted space: mode evolves as λ^k
# We'll plot the real part of λ^k for k from 0 to steps corresponding to 300 minutes
steps = int(300 / dt) + 1
k = np.arange(steps)
time = k * dt
mode_evolution = np.real(lam_dom ** k)

plt.figure(figsize=(10, 4))
plt.plot(time, mode_evolution, 'b-')
plt.xlabel('Time (minutes)')
plt.ylabel('Re(λ^k)')
plt.title('Dominant Mode Time Evolution')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Mode energy distribution (sorted |λ|)
# ----------------------------------------------------------------------
sorted_mag = np.sort(np.abs(eigvals))[::-1]
plt.figure(figsize=(8, 4))
plt.semilogy(sorted_mag, 'o-', markersize=3)
plt.xlabel('Mode index')
plt.ylabel('|λ|')
plt.title('Spectral decay (sorted magnitudes)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------
# Time constants for stable modes (|λ| < 1)
stable = np.abs(eigvals) < 1.0
if np.any(stable):
    s_stable = np.log(eigvals[stable]) / dt
    tau_stable = -1.0 / np.real(s_stable)
    tau_stable = tau_stable[np.isfinite(tau_stable)]   # remove inf if any
    slowest = np.max(tau_stable) if len(tau_stable) > 0 else np.nan
    fastest = np.min(tau_stable) if len(tau_stable) > 0 else np.nan
else:
    slowest = fastest = np.nan

n_slow = np.sum(np.abs(eigvals) > 0.95)
n_fast = np.sum(np.abs(eigvals) < 0.2)

print("\n--- Summary ---")
print(f"Slowest stable time constant: {slowest:.2f} min" if not np.isnan(slowest) else "Slowest stable time constant: N/A")
print(f"Fastest stable time constant: {fastest:.2f} min" if not np.isnan(fastest) else "Fastest stable time constant: N/A")
print(f"Modes with |λ| > 0.95 (slow): {n_slow}")
print(f"Modes with |λ| < 0.2  (fast): {n_fast}")

print("\n" + "=" * 60)
print("Inspection complete.")