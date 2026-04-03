"""
Autonomous Multiclass TRM — m3+m4 Unified Moving Bottleneck Extension
Dataset Generator
Author: Mingchen Yuan

3 classes: A (Trucks, m=0), Bf (Free Cars, m=1), Bs (Trapped Cars, m=2)
4-phase Lie-Trotter operator splitting:
  Phase 1: Exact Capture & Release  (analytical matrix exponential)
  Phase 2: Algebraic Projection + Implicit Kinematics  (Thomas algorithm)
  Phase 3: Lateral Lane-Changing  (Softplus + explicit Euler, Bs frozen)
  Phase 4: Spatial Advection  (FVM + global Godunov flux limiter)

Strictly follows Multi-class_TRM.tex equations and Benchmark Dataset.json parameters.
"""

import numpy as np
import h5py
import time
import os

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (Benchmark Dataset.json)
# ─────────────────────────────────────────────────────────────────────────────
X       = 150        # spatial cells
L       = 3          # lanes
N       = 15         # speed categories
M       = 3          # vehicle classes: 0=A (truck), 1=Bf (free car), 2=Bs (trapped car)

dx      = 20.0       # cell length [m]
dt      = 0.5        # time step [s]
T_STEPS = 500        # total time steps → 250 s simulation

# Speed spectrum v_i [m/s], i = 0..N-1
v = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
              18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0], dtype=np.float64)
v_max   = 30.0

# Macroscopic capacity & filter parameters
rho_max  = 0.15      # jam density [PCE/m]
R_supply = 0.035     # advection filter [PCE/m]
R_gap    = 0.025     # lateral gap filter [PCE/m]
R_c      = 0.05      # acceleration relaxation [PCE/m]
omega_sp = 20.0      # Softplus smoothing stiffness
eps      = 1.0e-8    # regularizer

# PCE weights: [A, Bf, Bs]  — Bf and Bs share same physical size
w = np.array([2.5, 1.0, 1.0], dtype=np.float64)

# Per-class kinematic parameters [A, Bf, Bs]
alpha   = np.array([0.35, 1.50, 1.50], dtype=np.float64)  # base acceleration [Hz]
eta_m   = np.array([4.5,  2.0,  2.0 ], dtype=np.float64)  # barrier stiffness exponent
omega_0 = np.array([0.05, 0.01, 0.01], dtype=np.float64)  # spontaneous anticipation [Hz]
kappa   = np.array([0.08, 0.60, 0.0 ], dtype=np.float64)  # lateral agility [Hz]; Bs=0

# Beta kinetic collision kernel (3×3): beta[m, n]  (with /rho_max in deceleration)
beta = np.array([
    [0.12, 0.08, 0.08],   # A  follows (A, Bf, Bs)
    [0.06, 0.03, 0.03],   # Bf follows (A, Bf, Bs)
    [0.06, 0.03, 0.03],   # Bs follows (A, Bf, Bs)
], dtype=np.float64)

# Moving bottleneck parameters
v_A_ff     = 14.0   # truck free-flow speed limit [m/s] → i_thr = 6
i_thr      = int(np.searchsorted(v, v_A_ff, side='right') - 1)  # = 6 (v[6]=14.0)
eta_lat    = 2.0    # escape gate exponent
omega_0_BA = 0.05   # kinetic exposure spontaneous rate [Hz]
beta_BA    = 0.06   # kinetic exposure rate [m^-1]
sigma_0    = 0.5    # base capture rate [Hz]
mu_0       = 0.3    # base release rate [Hz]
xi         = 0.5    # lateral awareness weight
R_A        = 0.05   # truck dispersal scale [PCE/m]

# CFL validation (Phase 4)
cfl = dt * v_max / dx
assert cfl <= 1.0, f"CFL violated: {cfl:.4f} > 1.0"
print(f"CFL = {cfl:.4f}  ✓   i_thr = {i_thr}  (v[i_thr]={v[i_thr]} m/s ≤ v_A_ff={v_A_ff})")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: effective occupancy  Ω_{x,l} = Σ_m Σ_i w^(m) f_{i,x,l}^(m)  (eq.1)
# ─────────────────────────────────────────────────────────────────────────────
def compute_omega(f):
    """f: (M, N, X, L)  →  omega: (X, L)"""
    return (w[:, None, None, None] * f).sum(axis=(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION  (Benchmark Dataset.json benchmark scenario)
# ─────────────────────────────────────────────────────────────────────────────
def initialize_state():
    """Return f of shape (M=3, N, X, L)."""
    f = np.full((M, N, X, L), 1.0e-5, dtype=np.float64)

    # Class A (trucks, m=0): bottleneck at cells 74-79, min speed (i=0, v=2m/s)
    # Ω = 2.5 × 0.058 = 0.145 ≈ ρ_max  → extreme stiffness
    f[0, 0, 74:80, :] = 0.058

    # Class Bf (free cars, m=1): injection at cells 59-69, max speed (i=14, v=30m/s)
    f[1, 14, 59:70, :] = 0.060

    # Class Bs (trapped cars, m=2): starts at zero everywhere
    f[2, :, :, :] = 0.0

    f = np.clip(f, 0.0, rho_max)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: safe φ(z) = (1 − e^{−z}) / z,  φ(0) = 1  (Phase 1 exact integrator)
# ─────────────────────────────────────────────────────────────────────────────
def safe_phi(z):
    """Numerically stable φ(z)=(1-e^{-z})/z.  For z<1e-12 returns 1.0."""
    return np.where(z < 1.0e-12, 1.0, -np.expm1(-z) / z)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Exact Capture & Release  (analytical matrix exponential)
# ─────────────────────────────────────────────────────────────────────────────
def phase1_capture_release(f, omega):
    """
    Implements Phase 1: Bf ↔ Bs reactions at the SAME speed index i.

    Kinetic exposure A^(i)_{x,l}  (eq.8):
      A[i,x,l] = Σ_{k≤i_thr} 1_{v_i≥v_k} · (ω_0_BA + β_BA·(v_i−v_k)) · w^(A)·f^(A)_{k} / ρ_max

    Escape gate G_{x,l→l'}  (eq.9):
      G = (Ω_{x,l'} / ρ_max)^{η_lat};  boundary G = 1

    Entrapment factor E = G_left · G_right

    Capture rate σ^(i)  (eq.10):
      σ = σ_0 · E · A^(i) · [1 + ξ · Σ_{l'} A^(i)_{x,l'}] · B(Ω)

    Release rate μ  (eq.11-12):
      S̃ = E · S · [1 + ξ · Σ_{l'} S_{x,l'}];  μ = μ_0 · exp(−S̃/R_A)

    Exact integration (eq.18-19):
      F = f_Bf + f_Bs
      f_Bf* = f_Bf·exp(−S·dt) + μ·F·dt·φ(S·dt)
      f_Bs* = F − f_Bf*

    Returns: f_new, sigma (N,X,L), mu (X,L), E_trap (X,L)
    """
    f_new = f.copy()

    # ── Kinetic exposure A^(i)_{x,l}  (eq.8) ────────────────────────────────
    # indicator[i, k] = 1 if v[i] >= v[k],  k = 0..i_thr
    indicator = (v[:, None] >= v[None, :i_thr + 1]).astype(np.float64)  # (N, i_thr+1)
    # Speed difference term: max(0, v_i - v_k)
    speed_diff = np.maximum(0.0, v[:, None] - v[None, :i_thr + 1])       # (N, i_thr+1)
    # Weight per (i, k) pair
    weight_ik = indicator * (omega_0_BA + beta_BA * speed_diff)           # (N, i_thr+1)
    # Weighted truck density per speed k: w^(A) * f^(A)_k / ρ_max
    fA_trucks = w[0] * f[0, :i_thr + 1, :, :] / rho_max                  # (i_thr+1, X, L)
    # A[i, x, l] = Σ_k weight_ik[i,k] * fA_trucks[k,x,l]
    A_exposure = np.einsum('ik,kxl->ixl', weight_ik, fA_trucks)          # (N, X, L)

    # Lateral sum of A (adjacent lane pressure term)
    A_lat_sum = np.zeros((N, X, L), dtype=np.float64)
    A_lat_sum[:, :, 1:]  += A_exposure[:, :, :-1]   # from left neighbor
    A_lat_sum[:, :, :-1] += A_exposure[:, :, 1:]    # from right neighbor

    # ── Escape gate G  (eq.9) ────────────────────────────────────────────────
    # G_{x,l→l'} = (clip(Ω_{x,l'}, 0, ρ_max) / ρ_max)^{η_lat}
    omega_norm = np.clip(omega, 0.0, rho_max) / rho_max    # (X, L) in [0,1]

    # G_right[x, l] = G_{x,l→l+1}: for l=0..L-2 use omega_norm[:, 1:]
    # Boundary: G_right[x, L-1] = 1 (no right neighbor → treat as fully blocked)
    G_right = np.ones((X, L), dtype=np.float64)
    G_right[:, :L - 1] = omega_norm[:, 1:] ** eta_lat

    # G_left[x, l] = G_{x,l→l-1}: for l=1..L-1 use omega_norm[:, :-1]
    # Boundary: G_left[x, 0] = 1 (no left neighbor → fully blocked)
    G_left = np.ones((X, L), dtype=np.float64)
    G_left[:, 1:] = omega_norm[:, :-1] ** eta_lat

    # Entrapment factor E_{x,l} = G_left · G_right  ∈ [0,1]
    E_trap = G_left * G_right                                              # (X, L)

    # ── Singular barrier for Bf (η^(Bf) = eta_m[1])  (eq.4) ─────────────────
    pressure_Bf = (rho_max / np.maximum(eps, rho_max - omega)
                   ) ** eta_m[1]                                           # (X, L)

    # ── Capture rate σ^(i)_{x,l}  (eq.10) ───────────────────────────────────
    sigma = (sigma_0
             * E_trap[None, :, :]
             * A_exposure
             * (1.0 + xi * A_lat_sum)
             * pressure_Bf[None, :, :])                                    # (N, X, L)
    sigma = np.maximum(sigma, 0.0)

    # ── Truck presence S_{x,l}  (eq.11) ─────────────────────────────────────
    S_truck = (w[0] * f[0, :i_thr + 1, :, :] / rho_max).sum(axis=0)      # (X, L)
    S_lat_sum = np.zeros((X, L), dtype=np.float64)
    S_lat_sum[:, 1:]  += S_truck[:, :-1]
    S_lat_sum[:, :-1] += S_truck[:, 1:]

    S_tilde = E_trap * S_truck * (1.0 + xi * S_lat_sum)                   # (X, L)

    # ── Release rate μ_{x,l}  (eq.12) ───────────────────────────────────────
    mu = mu_0 * np.exp(-S_tilde / R_A)                                    # (X, L)

    # ── Exact matrix exponential integration  (eq.18-19) ────────────────────
    # Total reaction rate per (i, x, l): S^(i) = σ^(i) + μ
    S_total = sigma + mu[None, :, :]                                       # (N, X, L)

    # Total B cars at speed i: F = f_Bf + f_Bs
    F_total = f[1] + f[2]                                                  # (N, X, L)

    # φ(S·dt) safe at S→0
    phi_z = safe_phi(S_total * dt)                                         # (N, X, L)

    # Exact update
    f_new[1] = f[1] * np.exp(-S_total * dt) + mu[None, :, :] * F_total * dt * phi_z
    f_new[2] = F_total - f_new[1]

    # Positivity guard
    f_new[1] = np.maximum(f_new[1], 0.0)
    f_new[2] = np.maximum(f_new[2], 0.0)

    return f_new, sigma, mu, E_trap


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Algebraic Projection + Semi-implicit Kinematics  (Thomas algorithm)
# ─────────────────────────────────────────────────────────────────────────────
def phase2_kinematics(f, omega):
    """
    Step A — Algebraic projection for Bs  (eq.20):
      Find κ*_{x,l} = highest i ≤ i_thr where f^(A)[i,x,l] > 0
      Move all f^(Bs)[j > κ*] to f^(Bs)[κ*], zero out j > κ*

    Step B — Semi-implicit Thomas for all 3 classes:
      λ_acc  (eq.3):  uses η^(m) per-class  (not global η)
      λ_dec  (eq.4-5): includes /ρ_max normalization in collision term
      Bs constraints: λ_acc^(Bs)[i ≥ i_thr] = 0  (Acceleration Blockade)

    Returns: f, lambda_acc (M,N,X,L), lambda_dec (M,N,X,L)
    """
    f_new = f.copy()

    # ── Step A: Algebraic projection ─────────────────────────────────────────
    thresh = 1.0e-10
    truck_present = f[0, :i_thr + 1, :, :] > thresh              # (i_thr+1, X, L)

    # κ* = highest i ≤ i_thr where truck is present
    # Flip speed axis, argmax finds first True from high end
    truck_flipped = truck_present[::-1, :, :]                     # (i_thr+1, X, L)
    argmax_flip   = np.argmax(truck_flipped, axis=0)              # (X, L)
    kappa_star    = i_thr - argmax_flip                            # (X, L)
    any_truck     = truck_present.any(axis=0)                     # (X, L)
    kappa_star    = np.where(any_truck, kappa_star, 0)            # default 0 if no trucks

    # Mask for speed indices above κ*: high_mask[i, x, l] = (i > κ*[x,l])
    speed_idx     = np.arange(N)
    high_mask     = speed_idx[:, None, None] > kappa_star[None, :, :]  # (N, X, L)

    # Mass to redistribute: trapped cars going faster than κ*
    excess        = f_new[2] * high_mask                          # (N, X, L)
    excess_total  = excess.sum(axis=0)                            # (X, L)

    # One-hot mask for κ* position
    kappa_mask    = speed_idx[:, None, None] == kappa_star[None, :, :]  # (N, X, L)

    f_new[2] = f_new[2] - excess + kappa_mask * excess_total[None, :, :]
    f_new[2] = np.maximum(f_new[2], 0.0)

    # ── Step B: Kinematic rates ───────────────────────────────────────────────
    # Acceleration rates λ_acc^(m,i)  (eq.3): uses η^(m) per class
    supply_kin   = np.maximum(0.0, rho_max - omega)              # (X, L)
    acc_filter   = 1.0 - np.exp(-supply_kin / R_c)              # (X, L)

    # speed_factor[m, i] = (1 - v[i]/v_max)^{η^(m)}
    speed_factor = (1.0 - v[None, :] / v_max) ** eta_m[:, None] # (M, N)

    lambda_acc = (alpha[:, None, None, None]
                  * speed_factor[:, :, None, None]
                  * acc_filter[None, None, :, :])                # (M, N, X, L)
    # Dirichlet upper bound: no acceleration from top speed bin
    lambda_acc[:, N - 1, :, :] = 0.0
    # Bs Acceleration Blockade: λ_acc^(Bs)[i ≥ i_thr] = 0  (eq.2)
    lambda_acc[2, i_thr:, :, :] = 0.0

    # Deceleration rates λ_dec^(m,i)  (eq.4-5):
    # Interaction sum: Σ_n Σ_{k<i} β^(m,n)·(v_i−v_k)·w^(n)·f_k^(n) / ρ_max
    # Uses prefix-sum trick (O(N) instead of O(N²))
    beta_w = beta * w[None, :]                                   # (M, M) = β[m,n]*w[n]
    # bwf[m, k, x, l] = Σ_n β_w[m,n] * f[n,k,x,l] / ρ_max
    bwf = np.einsum('mn,nkxl->mkxl', beta_w, f_new) / rho_max  # (M, N, X, L)

    # Prefix sums (shifted right so cum_bwf[m,i] = Σ_{k<i} bwf[m,k])
    cum_bwf  = np.zeros_like(bwf)
    cum_vbwf = np.zeros_like(bwf)
    cum_bwf [:, 1:, :, :] = np.cumsum(bwf [:, :-1, :, :], axis=1)
    cum_vbwf[:, 1:, :, :] = np.cumsum(
        v[None, :-1, None, None] * bwf[:, :-1, :, :], axis=1)

    interaction = v[None, :, None, None] * cum_bwf - cum_vbwf   # (M, N, X, L)

    # Singular pressure barrier B(Ω)^{η^(m)}  (eq.4)
    pressure = (rho_max / np.maximum(eps, rho_max - omega[None, None, :, :])
                ) ** eta_m[:, None, None, None]                  # (M, N, X, L)

    lambda_dec = (omega_0[:, None, None, None] + interaction) * pressure
    lambda_dec = np.maximum(lambda_dec, 0.0)
    # Dirichlet lower bound: no deceleration below min speed
    lambda_dec[:, 0, :, :] = 0.0

    # ── Thomas algorithm: solve (I − Δt·A)·f_new = f_old ────────────────────
    # Tridiagonal for speed axis (size N) per (m, x, l):
    #   a[i] = −Δt · λ_acc[m, i-1]   sub-diagonal  (inflow from i-1 via acc)
    #   b[i] =  1 + Δt·(λ_acc[i] + λ_dec[i])
    #   c[i] = −Δt · λ_dec[m, i+1]   super-diagonal (inflow from i+1 via dec)
    #   d[i] = f[m, i, x, l]  (rhs)

    a = np.zeros((M, N, X, L), dtype=np.float64)
    b = 1.0 + dt * (lambda_acc + lambda_dec)
    c = np.zeros((M, N, X, L), dtype=np.float64)
    d = f_new.copy()

    a[:, 1:,  :, :] = -dt * lambda_acc[:, :-1, :, :]
    c[:, :-1, :, :] = -dt * lambda_dec[:, 1:,  :, :]

    # Forward sweep
    c_p = np.zeros_like(c)
    d_p = np.zeros_like(d)
    c_p[:, 0, :, :] = c[:, 0, :, :] / b[:, 0, :, :]
    d_p[:, 0, :, :] = d[:, 0, :, :] / b[:, 0, :, :]
    for i in range(1, N):
        denom = b[:, i, :, :] - a[:, i, :, :] * c_p[:, i - 1, :, :]
        denom = np.maximum(denom, eps)
        c_p[:, i, :, :] = c[:, i, :, :] / denom
        d_p[:, i, :, :] = (d[:, i, :, :] - a[:, i, :, :] * d_p[:, i - 1, :, :]) / denom

    # Back substitution
    f_out = np.zeros_like(f_new)
    f_out[:, N - 1, :, :] = d_p[:, N - 1, :, :]
    for i in range(N - 2, -1, -1):
        f_out[:, i, :, :] = d_p[:, i, :, :] - c_p[:, i, :, :] * f_out[:, i + 1, :, :]

    f_out = np.maximum(f_out, 0.0)
    return f_out, lambda_acc, lambda_dec, kappa_star


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Lateral Lane-Changing  (Softplus, Bs frozen)
# ─────────────────────────────────────────────────────────────────────────────
def phase3_lateral(f, omega):
    """
    Softplus lateral rates for A (m=0) and Bf (m=1).
    Bs (m=2): gamma = 0 everywhere (Absolute Lateral Immobilization, eq.3).
    Dirichlet: gamma_{l→0} = gamma_{l→L+1} = 0.

    Rate (eq.6):
      γ_{l→l'} = κ^(m)/ω·ln[1+exp(ω·y)] · [1−exp(−max(0,ρ_max−Ω_{l'})/R_gap)]
      where y = (Ω_{x,l} − Ω_{x,l'}) / ρ_max
    """
    def softplus(y):
        return np.where(y > 20.0,
                        y,
                        np.log1p(np.exp(np.minimum(omega_sp * y, 500.0))) / omega_sp)

    # ── Rightward: l → l+1 ───────────────────────────────────────────────────
    y_right        = (omega[:, :-1] - omega[:, 1:]) / rho_max              # (X, L-1)
    gap_right      = np.maximum(0.0, rho_max - omega[:, 1:])
    gap_f_right    = 1.0 - np.exp(-gap_right / R_gap)                      # (X, L-1)
    sp_right       = softplus(y_right)

    gr_internal    = (kappa[:, None, None, None]
                      * sp_right[None, None, :, :]
                      * gap_f_right[None, None, :, :])                     # (M, N, X, L-1)
    gamma_right    = np.zeros((M, N, X, L), dtype=np.float64)
    gamma_right[:, :, :, :L - 1] = gr_internal
    # Bs immobilization
    gamma_right[2, :, :, :] = 0.0

    # ── Leftward: l → l-1 ────────────────────────────────────────────────────
    y_left         = (omega[:, 1:] - omega[:, :-1]) / rho_max              # (X, L-1)
    gap_left       = np.maximum(0.0, rho_max - omega[:, :-1])
    gap_f_left     = 1.0 - np.exp(-gap_left / R_gap)
    sp_left        = softplus(y_left)

    gl_internal    = (kappa[:, None, None, None]
                      * sp_left[None, None, :, :]
                      * gap_f_left[None, None, :, :])
    gamma_left     = np.zeros((M, N, X, L), dtype=np.float64)
    gamma_left[:, :, :, 1:] = gl_internal
    gamma_left[2, :, :, :] = 0.0

    # ── Explicit Euler update ─────────────────────────────────────────────────
    loss = (gamma_right + gamma_left) * f
    gain = np.zeros_like(f)
    gain[:, :, :, 1:]   += gamma_right[:, :, :, :L - 1] * f[:, :, :, :L - 1]
    gain[:, :, :, :L-1] += gamma_left[:, :, :, 1:]      * f[:, :, :, 1:]

    f_new = f + dt * (gain - loss)
    f_new = np.maximum(f_new, 0.0)
    return f_new, gamma_left, gamma_right


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Spatial Advection  (FVM + Global Godunov Flux Limiter)
# ─────────────────────────────────────────────────────────────────────────────
def phase4_advection(f, omega):
    """
    All classes advect at their actual speed v_i (eq.13).

    Demand Ψ^(m)  (eq.13):
      Ψ_{i,x→x+1,l}^(m) = v_i · f_{i,x,l}^(m) · [1−exp(−max(0,ρ_max−Ω_{x+1,l})/R_supply)]

    Global aggregate demand at face x→x+1  (eq.14):
      D_{x→x+1,l} = dt/dx · Σ_{m,i} w^(m) · Ψ^(m)_{i,x,l}

    Global Godunov flux limiter α  (eq.14):
      α_{x→x+1,l} = min(1, max(0, ρ_max−Ω_{x+1,l}) / D)  if D > 0,  else 1

    Final flux  (eq.14):
      Φ^(m) = α · Ψ^(m)

    Returns f_new (M,N,X,L), phi (M,N,X+1,L)
    """
    phi = np.zeros((M, N, X + 1, L), dtype=np.float64)

    # ── Supply filter at downstream cell (faces 1..X-1) ─────────────────────
    supply_arg    = np.maximum(0.0, rho_max - omega[1:X, :])           # (X-1, L)
    supply_factor = 1.0 - np.exp(-supply_arg / R_supply)               # (X-1, L)

    # ── Demand Ψ at internal faces 1..X-1 ────────────────────────────────────
    Psi_internal = (v[None, :, None, None]
                    * f[:, :, :X - 1, :]
                    * supply_factor[None, None, :, :])                  # (M, N, X-1, L)

    # ── Global aggregate demand D_{face, l} ──────────────────────────────────
    # D = dt/dx * Σ_{m,i} w[m] * Psi[m,i,face,l]
    D = ((dt / dx)
         * (w[:, None, None, None] * Psi_internal).sum(axis=(0, 1)))   # (X-1, L)

    # ── Available space at downstream cell ───────────────────────────────────
    available = np.maximum(0.0, rho_max - omega[1:X, :])               # (X-1, L)

    # ── Global Godunov limiter α ──────────────────────────────────────────────
    alpha_g = np.where(D > eps, np.minimum(1.0, available / D), 1.0)  # (X-1, L)

    # ── Apply α to all classes simultaneously ────────────────────────────────
    phi[:, :, 1:X, :] = Psi_internal * alpha_g[None, None, :, :]

    # ── Right boundary: free outflow (no downstream constraint) ─────────────
    phi[:, :, X, :] = v[None, :, None] * f[:, :, X - 1, :]

    # ── FVM update ────────────────────────────────────────────────────────────
    f_new = f + (dt / dx) * (phi[:, :, :X, :] - phi[:, :, 1:X + 1, :])
    f_new = np.maximum(f_new, 0.0)
    return f_new, phi


# ─────────────────────────────────────────────────────────────────────────────
# MACROSCOPIC DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_macroscopic(f):
    """rho_macro, q_macro, u_macro each of shape (M, X, L)."""
    rho_macro = f.sum(axis=1)                                           # (M, X, L)
    q_macro   = (v[None, :, None, None] * f).sum(axis=1)               # (M, X, L)
    u_macro   = q_macro / np.maximum(eps, rho_macro)                   # (M, X, L)
    return rho_macro, q_macro, u_macro


# ─────────────────────────────────────────────────────────────────────────────
# MASS CONSERVATION CHECK (Class B = Bf + Bs)
# ─────────────────────────────────────────────────────────────────────────────
def check_mass(f_old, f_new, phi):
    """
    Verify ΔP^(B) ≈ net boundary flux for combined B class (Bf+Bs).
    Capture/release is internal zero-sum (proven in §7 Theorem).
    """
    # Combined B mass (Bf + Bs)
    mass_B_old = ((f_old[1] + f_old[2]) * w[1]).sum() * dx
    mass_B_new = ((f_new[1] + f_new[2]) * w[1]).sum() * dx
    # Net outflow through boundaries (right face X, left face 0)
    net_flux = ((w[1] * phi[1, :, X, :] + w[2] * phi[2, :, X, :]).sum()
                - (w[1] * phi[1, :, 0, :] + w[2] * phi[2, :, 0, :]).sum()) * dt
    residual = abs((mass_B_new - mass_B_old) + net_flux)
    rel_err  = residual / (mass_B_old + eps)
    return rel_err


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup_hdf5(filepath, T):
    """Pre-allocate all datasets. No compression for exact ~500 MB target."""
    hf = h5py.File(filepath, 'w')

    # ── Parameters ───────────────────────────────────────────────────────────
    pg = hf.create_group('parameters')
    pg.attrs['model']         = 'Autonomous Multiclass TRM m3+m4'
    pg.attrs['M']             = M
    pg.attrs['N']             = N
    pg.attrs['X']             = X
    pg.attrs['L']             = L
    pg.attrs['dx_m']          = dx
    pg.attrs['dt_s']          = dt
    pg.attrs['T_steps']       = T
    pg.attrs['v_max_mps']     = v_max
    pg.attrs['rho_max']       = rho_max
    pg.attrs['R_supply']      = R_supply
    pg.attrs['R_gap']         = R_gap
    pg.attrs['R_c']           = R_c
    pg.attrs['omega_sp']      = omega_sp
    pg.attrs['eps']           = eps
    pg.attrs['v_A_ff_mps']    = v_A_ff
    pg.attrs['i_thr']         = i_thr
    pg.attrs['eta_lat']       = eta_lat
    pg.attrs['sigma_0']       = sigma_0
    pg.attrs['mu_0']          = mu_0
    pg.attrs['xi']            = xi
    pg.attrs['R_A']           = R_A
    pg.attrs['omega_0_BA']    = omega_0_BA
    pg.attrs['beta_BA']       = beta_BA
    pg.attrs['CFL']           = cfl
    pg.attrs['operator_split'] = '4-phase: Capture/Release -> Kinematics -> Lateral -> Advection'

    pg['v_mps']      = v
    pg['w_PCE']      = w
    pg['alpha_hz']   = alpha
    pg['eta_m']      = eta_m
    pg['omega_0_hz'] = omega_0
    pg['kappa_hz']   = kappa
    pg['beta_matrix'] = beta

    # ── Data datasets ─────────────────────────────────────────────────────────
    dg = hf.create_group('data')

    def mk(name, shape_t, dtype=np.float64):
        full  = (T,) + shape_t
        chunk = (1,) + shape_t
        dg.create_dataset(name, shape=full, dtype=dtype, chunks=chunk)

    mk('f',           (M, N, X, L))     # primary state
    mk('omega',       (X, L))           # effective occupancy
    mk('phi',         (M, N, X + 1, L))# spatial fluxes
    mk('lambda_acc',  (M, N, X, L))    # acceleration rates
    mk('lambda_dec',  (M, N, X, L))    # deceleration rates
    mk('gamma_left',  (M, N, X, L))    # lateral rate l→l-1
    mk('gamma_right', (M, N, X, L))    # lateral rate l→l+1
    mk('sigma',       (N, X, L))       # capture rate (Phase 1)
    mk('mu',          (X, L))          # release rate (Phase 1)
    mk('E_trap',      (X, L))          # entrapment factor
    mk('kappa_star',  (X, L))          # truck speed threshold per cell (int stored as float)
    mk('rho_macro',   (M, X, L))       # macroscopic density
    mk('q_macro',     (M, X, L))       # macroscopic flow
    mk('u_macro',     (M, X, L))       # macroscopic speed

    dg.create_dataset('time_s', data=np.arange(T) * dt)
    hf.create_dataset('diagnostics/mass_rel_error_B', shape=(T,), dtype=np.float64)
    return hf


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run(output_path):
    print("=" * 66)
    print("  Autonomous Multiclass TRM — m3+m4 Moving Bottleneck Extension")
    print(f"  Grid: {X}×{L} cells/lanes, {N} speeds, M={M} classes")
    print(f"  Steps: {T_STEPS}  dt={dt}s  →  {T_STEPS*dt:.0f}s simulation")
    print(f"  i_thr={i_thr}  (v_A_ff={v_A_ff} m/s)")
    print(f"  4-Phase: CaptureRelease → Kinematics → Lateral → Advection")
    print(f"  Output: {output_path}")
    print("=" * 66)

    # Estimate file size
    bytes_per_step = (
        M*N*X*L*8 +       # f
        X*L*8 +           # omega
        M*N*(X+1)*L*8 +   # phi
        M*N*X*L*8*2 +     # lambda_acc + lambda_dec
        M*N*X*L*8*2 +     # gamma_left + gamma_right
        N*X*L*8 +         # sigma
        X*L*8*2 +         # mu + E_trap + kappa_star
        X*L*8 +
        M*X*L*8*3         # rho, q, u macro
    )
    print(f"  Estimated size: {T_STEPS * bytes_per_step / 1024**2:.0f} MB\n")

    f      = initialize_state()
    f_old  = f.copy()
    hf     = setup_hdf5(output_path, T_STEPS)
    dg     = hf['data']
    diag   = hf['diagnostics/mass_rel_error_B']

    # Upstream inflow: Bf cars at max speed into all lanes
    INFLOW_BF = 0.001   # veh/m·s

    t_wall = time.perf_counter()

    for t in range(T_STEPS):
        t0 = time.perf_counter()

        omega = compute_omega(f)

        # Phase 1: Exact Capture & Release
        f, sigma, mu, E_trap = phase1_capture_release(f, omega)

        # Phase 2: Algebraic Projection + Thomas Kinematics
        omega = compute_omega(f)
        f, lambda_acc, lambda_dec, kappa_star = phase2_kinematics(f, omega)

        # Phase 3: Lateral Lane-Changing (Bs frozen)
        omega = compute_omega(f)
        f, gamma_left, gamma_right = phase3_lateral(f, omega)

        # Phase 4: Spatial Advection (global Godunov)
        omega = compute_omega(f)
        f, phi = phase4_advection(f, omega)

        # Upstream inflow boundary (Bf at max speed)
        f[1, N - 1, 0, :] += INFLOW_BF * dt
        f = np.minimum(f, rho_max)

        # Final omega
        omega_final = compute_omega(f)

        # Diagnostics
        rho_m, q_m, u_m = compute_macroscopic(f)
        rel_err = check_mass(f_old, f, phi)
        f_old = f.copy()

        # Write to HDF5
        dg['f'][t]            = f
        dg['omega'][t]        = omega_final
        dg['phi'][t]          = phi
        dg['lambda_acc'][t]   = lambda_acc
        dg['lambda_dec'][t]   = lambda_dec
        dg['gamma_left'][t]   = gamma_left
        dg['gamma_right'][t]  = gamma_right
        dg['sigma'][t]        = sigma
        dg['mu'][t]           = mu
        dg['E_trap'][t]       = E_trap
        dg['kappa_star'][t]   = kappa_star.astype(np.float64)
        dg['rho_macro'][t]    = rho_m
        dg['q_macro'][t]      = q_m
        dg['u_macro'][t]      = u_m
        diag[t]               = rel_err

        if t % 50 == 0 or t == T_STEPS - 1:
            step_ms = (time.perf_counter() - t0) * 1000
            eta_s   = (time.perf_counter() - t_wall) / max(t + 1, 1) * (T_STEPS - t - 1)
            o_peak  = omega_final.max()
            Bs_tot  = f[2].sum()
            print(f"  step {t:4d}/{T_STEPS}  |  {step_ms:5.1f}ms  |"
                  f"  ETA {eta_s:5.0f}s  |"
                  f"  Ω_peak={o_peak:.4f}  |"
                  f"  f_Bs_total={Bs_tot:.4f}  |"
                  f"  mass_err_B={rel_err:.2e}")

        if t % 100 == 0:
            hf.flush()

    hf.attrs['description'] = (
        'm3+m4 Autonomous Multiclass TRM benchmark dataset. '
        'Classes: A (trucks), Bf (free cars), Bs (trapped cars). '
        '4-phase Lie-Trotter: Capture/Release (exact exp) -> '
        'Kinematics+Projection (Thomas) -> Lateral (Softplus) -> '
        'Advection (global Godunov). '
        'Benchmark: A bottleneck cells 74-79, Bf injection cells 59-69.'
    )
    hf.close()

    size_mb = os.path.getsize(output_path) / 1024**2
    total_s = time.perf_counter() - t_wall
    print("=" * 66)
    print(f"  Done!  Wall time: {total_s:.1f}s")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Simulation duration: {T_STEPS * dt:.0f} s")
    print("=" * 66)


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(__file__),
                       'multiclass_trm_benchmark_500mb.h5')
    run(out)
