# glycoSMC  
## Koopman-Based Identification of Glucose–Insulin Dynamics for Control

---

## Overview

**glycoSMC** implements a control-oriented Koopman operator identification pipeline for modeling glucose–insulin dynamics using Extended Dynamic Mode Decomposition with control (eDMDc).

The objective is to learn a **stable, interpretable, controller-ready surrogate model** directly from:

- BG (body glucose)
- Insulin input
- Meal input

The identified discrete-time lifted model has the form:

Φₖ₊₁ = A Φₖ + B uₖ  
yₖ = C Φₖ

Where:

- Φₖ = lifted state (delay embedding + nonlinear physiological features)
- uₖ = [insulin, meal]
- yₖ = BG (body glucose)
- A ∈ ℝⁿˣⁿ = intrinsic lifted dynamics
- B ∈ ℝⁿˣ² = input matrix

This model serves as a surrogate for nonlinear glucose–insulin physiology and is designed for downstream controller synthesis.

---

## Identification Pipeline

The pipeline includes:

### 1️ Data Processing
- Chronological train/validation/test split
- Normalization using training statistics
- Delay embedding of BG, insulin, and meal signals

### 2️ Nonlinear Lifting
- Identity delay states
- Glucose–insulin interaction terms
- Glucose–meal interaction terms
- Mild nonlinear saturation features

### 3️ Model Identification
- Ridge regression baseline
- Truncated SVD with rank selection
- Spectral radius stabilization
- Stability-constrained validation

### 4️ Structural Diagnostics
- Eigenvalue spectrum
- Continuous-time pole computation (s = log(λ)/dt)
- Time constant extraction
- Controllability analysis
- Modal controllability analysis
- Glucose participation in Koopman modes
- Balanced truncation and reduced-order comparison

### 5️ Long-Horizon Stress Testing
- 2-hour prediction
- 6-hour prediction
- 12-hour prediction

---

## What Was Identified?

From BG, insulin, and meal data, the pipeline extracts:

- A stable lifted linear surrogate of nonlinear glucose dynamics
- Dominant physiological time constants (~100 min insulin action, ~1100 min slow regulation)
- Oscillatory regulatory modes
- Structured insulin and meal actuation directions
- Effective dynamical order ≈ 3–5 dominant modes

The identified system is:

- Stable
- Structurally consistent across age groups
- Controllable in the dominant subspace
- Suitable for control design

---

## Multi-Patient Validation

Validated across:

- adolescent_001  
- adult_001  
- child_001  

Metrics compared:

- Selected rank
- Spectral radius
- 2h / 6h / 12h RMSE
- Controllability rank
- Modal controllability statistics

Structural consistency observed across patients.

---

## Repository Structure

glycoSMC/
│
├── data/ # Patient CSV files (BG, insulin, meal)
├── models/ # Saved identified models (.npz)
├── results/ # Plots and validation reports
├── notebooks/ # Jupyter exploration
│
├── edmdc_pipeline.py # Main identification pipeline
├── multi_patient_validation.py # Multi-patient comparison script
├── inspect_identified_model.py # Structural inspection script
└── README.md