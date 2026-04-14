"""
Autonomous Multiclass TRM — m3+m4 Probabilistic Blocking Extension
Dataset Generator
Author: Mingchen Yuan

3 classes: A (Trucks, m=0), Bf (Free Cars, m=1), Bs (Trapped Cars, m=2)
3-phase Lie-Trotter operator splitting:
  Phase 1: Exact Capture & Release  (analytical matrix exponential, P_block-based)
  Phase 2: Algebraic Projection + Implicit Kinematics  (Thomas algorithm)
  Phase 3: Spatial Advection  (FVM + global Godunov flux limiter)

NOTE: Lateral lane-changing (old Phase 3) is REMOVED.
      Each lane is solved as an independent 1D system.
      P_block(x) = (Omega/rho_max)^eta_block replaces E_trap/G escape gates.

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
L       = 3          # lanes (independent 1D systems)
N       = 16         # speed categories (v=0 added per supervisor: stationary traffic class)
M       = 3          # vehicle classes: 0=A (truck), 1=Bf (free car), 2=Bs (trapped car)

dx      = 20.0       # cell length [m]
dt      = 0.5        # time step [s]
T_STEPS = 500        # total time steps → 250 s simulation

# Speed spectrum v_i [m/s], i = 0..N-1
# v[0]=0: completely stationary (jam), v[1..15]=2..30 m/s free-flow spectrum
v = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
              18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0], dtype=np.float64)
v_max   = 30.0

# Macroscopic capacity & filter parameters
rho_max  = 0.15      # jam density [PCE/m]
R_supply = 0.035     # advection filter [PCE/m]
R_c      = 0.05      # acceleration relaxation [PCE/m]
eps      = 1.0e-8    # regularizer

# PCE weights: [A, Bf, Bs]  — Bf and Bs share same physical size
w = np.array([2.5, 1.0, 1.0], dtype=np.float64)

# Per-class kinematic parameters [A, Bf, Bs]
alpha   = np.array([0.35, 1.50, 1.50], dtype=np.float64)  # base acceleration [Hz]
eta_m   = np.array([4.5,  2.0,  2.0 ], dtype=np.float64)  # barrier stiffness exponent
omega_0 = np.array([0.05, 0.01, 0.01], dtype=np.float64)  # spontaneous anticipation [Hz]

# Beta kinetic collision kernel (3×3): beta[m, n]  (with /rho_max in deceleration)
beta = np.array([
    [0.12, 0.08, 0.08],   # A  follows (A, Bf, Bs)
    [0.06, 0.03, 0.03],   # Bf follows (A, Bf, Bs)
    [0.06, 0.03, 0.03],   # Bs follows (A, Bf, Bs)
], dtype=np.float64)

# Moving bottleneck parameters
v_A_ff     = 14.0   # truck free-flow speed limit [m/s] → i_thr = 6
i_thr      = int(np.searchsorted(v, v_A_ff, side='right') - 1)  # = 6 (v[6]=14.0)
eta_block  = 2.0    # probabilistic blocking exponent (Eq. 8)
omega_0_BA = 0.05   # kinetic exposure spontaneous rate [Hz]
beta_BA    = 0.06   # kinetic exposure rate [m^-1]
sigma_0    = 0.8    # base capture rate [Hz]
mu_0       = 0.3    # base release rate [Hz]
R_A        = 0.05   # truck dispersal scale [PCE/m]

# CFL validation (Phase 3)
cfl = dt * v_max / dx
assert cfl <= 1.0, f"CFL violated: {cfl:.4f} > 1.0"
print(f"CFL = {cfl:.4f}  OK   i_thr = {i_thr}  (v[i_thr]={v[i_thr]} m/s <= v_A_ff={v_A_ff})")

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

    # Class Bf (free cars, m=1): uniform upstream (x=0-73, v=30m/s, ρ=0.020)
    # Left half of ring = sustained free-flow state → classic Riemann problem setup.
    # Estimated shock speed ≈ -4.6 m/s → shock travels ~58 cells in 250 s (clearly visible).
    f[1, 14, 0:74, :] = 0.035

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
# PHASE 1 — Exact Capture & Release  (P_block-based, analytical matrix exponential)
# ─────────────────────────────────────────────────────────────────────────────
def phase1_capture_release(f, omega):
    """
    Implements Phase 1: Bf <-> Bs reactions at the SAME speed index i.

    Probabilistic Blocking Factor P_block(x)  (eq.8):
      P_block = (clip(Omega, 0, rho_max) / rho_max) ^ eta_block

    Kinetic exposure A^(i)_{x,l}  (eq.7):
      A[i,x,l] = sum_{k<=i_thr} 1_{v_i>=v_k} * (omega_0_BA + beta_BA*(v_i-v_k))
                 * w^(A)*f^(A)_{k} / rho_max

    Capture rate sigma^(i)  (eq.9):
      sigma = sigma_0 * P_block * A^(i) * B(Omega)   [no E_trap, no xi]

    Truck footprint theta_A  (eq.10):
      theta_A = sum_{k<=i_thr} w^(A)*f^(A)_{k} / rho_max

    Effective bottleneck pressure S_tilde  (eq.11):
      S_tilde = P_block * theta_A

    Release rate mu  (eq.12):
      mu = mu_0 * exp(-S_tilde / R_A)

    Exact integration:
      F = f_Bf + f_Bs
      f_Bf* = f_Bf*exp(-S*dt) + mu*F*dt*phi(S*dt)
      f_Bs* = F - f_Bf*

    Returns: f_new, sigma (N,X,L), mu (X,L), P_block (X,L)
    """
    f_new = f.copy()

    # ── Probabilistic Blocking Factor P_block  (eq.8) ────────────────────────
    omega_norm = np.clip(omega, 0.0, rho_max) / rho_max        # (X, L) in [0,1]
    P_block    = omega_norm ** eta_block                        # (X, L) in [0,1]

    # ── Kinetic exposure A^(i)_{x,l}  (eq.7) ─────────────────────────────────
    # indicator[i, k] = 1 if v[i] >= v[k],  k = 0..i_thr
    indicator = (v[:, None] >= v[None, :i_thr + 1]).astype(np.float64)  # (N, i_thr+1)
    # Speed difference term: max(0, v_i - v_k)
    speed_diff = np.maximum(0.0, v[:, None] - v[None, :i_thr + 1])       # (N, i_thr+1)
    # Weight per (i, k) pair
    weight_ik = indicator * (omega_0_BA + beta_BA * speed_diff)           # (N, i_thr+1)
    # Weighted truck density per speed k: w^(A) * f^(A)_k / rho_max
    fA_trucks = w[0] * f[0, :i_thr + 1, :, :] / rho_max                  # (i_thr+1, X, L)
    # A[i, x, l] = sum_k weight_ik[i,k] * fA_trucks[k,x,l]
    A_exposure = np.einsum('ik,kxl->ixl', weight_ik, fA_trucks)          # (N, X, L)

    # ── Singular barrier for Bf (eta^(Bf) = eta_m[1])  (eq.4) ───────────────
    pressure_Bf = (rho_max / np.maximum(eps, rho_max - omega)
                   ) ** eta_m[1]                                           # (X, L)

    # ── Capture rate sigma^(i)_{x,l}  (eq.9) ─────────────────────────────────
    # sigma = sigma_0 * P_block * A^(i) * B(Omega)   [no E_trap, no xi]
    sigma = (sigma_0
             * P_block[None, :, :]
             * A_exposure
             * pressure_Bf[None, :, :])                                    # (N, X, L)
    sigma = np.maximum(sigma, 0.0)

    # ── Truck footprint theta_A  (eq.10) ─────────────────────────────────────
    theta_A = (w[0] * f[0, :i_thr + 1, :, :] / rho_max).sum(axis=0)      # (X, L)

    # ── Effective bottleneck pressure S_tilde  (eq.11) ───────────────────────
    S_tilde = P_block * theta_A                                            # (X, L)

    # ── Release rate mu_{x,l}  (eq.12) ──────────────────────────────────────
    mu = mu_0 * np.exp(-S_tilde / R_A)                                    # (X, L)

    # ── Exact matrix exponential integration ─────────────────────────────────
    # Total reaction rate per (i, x, l): S^(i) = sigma^(i) + mu
    S_total = sigma + mu[None, :, :]                                       # (N, X, L)

    # Total B cars at speed i: F = f_Bf + f_Bs
    F_total = f[1] + f[2]                                                  # (N, X, L)

    # phi(S*dt) safe at S->0
    phi_z = safe_phi(S_total * dt)                                         # (N, X, L)

    # Exact update
    f_new[1] = f[1] * np.exp(-S_total * dt) + mu[None, :, :] * F_total * dt * phi_z
    f_new[2] = F_total - f_new[1]

    # Positivity guard
    f_new[1] = np.maximum(f_new[1], 0.0)
    f_new[2] = np.maximum(f_new[2], 0.0)

    return f_new, sigma, mu, P_block


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Algebraic Projection + Semi-implicit Kinematics  (Thomas algorithm)
# ─────────────────────────────────────────────────────────────────────────────
def phase2_kinematics(f, omega):
    """
    Step A — Algebraic projection for Bs:
      Find kappa*_{x,l} = highest i <= i_thr where f^(A)[i,x,l] > 0
      Move all f^(Bs)[j > kappa*] to f^(Bs)[kappa*], zero out j > kappa*

    Step B — Semi-implicit Thomas for all 3 classes:
      lambda_acc  (eq.3):  uses eta^(m) per-class
      lambda_dec  (eq.4-5): includes /rho_max normalization in collision term
      Bs constraints: lambda_acc^(Bs)[i >= i_thr] = 0  (Acceleration Blockade)

    Returns: f, lambda_acc (M,N,X,L), lambda_dec (M,N,X,L), kappa_star (X,L)
    """
    f_new = f.copy()

    # ── Step A: Algebraic projection ─────────────────────────────────────────
    thresh = 1.0e-10
    truck_present = f[0, :i_thr + 1, :, :] > thresh              # (i_thr+1, X, L)

    # kappa* = highest i <= i_thr where truck is present
    truck_flipped = truck_present[::-1, :, :]                     # (i_thr+1, X, L)
    argmax_flip   = np.argmax(truck_flipped, axis=0)              # (X, L)
    kappa_star    = i_thr - argmax_flip                            # (X, L)

    any_truck     = truck_present.any(axis=0)                     # (X, L)
    kappa_star    = np.where(any_truck, kappa_star, 0)            # default 0 if no trucks

    # Mask for speed indices above kappa*: high_mask[i, x, l] = (i > kappa*[x,l])
    speed_idx     = np.arange(N)
    high_mask     = speed_idx[:, None, None] > kappa_star[None, :, :]  # (N, X, L)

    # Mass to redistribute: trapped cars going faster than kappa*
    excess        = f_new[2] * high_mask                          # (N, X, L)
    excess_total  = excess.sum(axis=0)                            # (X, L)

    # One-hot mask for kappa* position
    kappa_mask    = speed_idx[:, None, None] == kappa_star[None, :, :]  # (N, X, L)

    f_new[2] = f_new[2] - excess + kappa_mask * excess_total[None, :, :]
    f_new[2] = np.maximum(f_new[2], 0.0)

    # ── Step B: Kinematic rates ───────────────────────────────────────────────
    # Acceleration rates lambda_acc^(m,i)  (eq.3): uses eta^(m) per class
    supply_kin   = np.maximum(0.0, rho_max - omega)              # (X, L)
    acc_filter   = 1.0 - np.exp(-supply_kin / R_c)              # (X, L)

    # speed_factor[m, i] = (1 - v[i]/v_max)^{eta^(m)}
    speed_factor = (1.0 - v[None, :] / v_max) ** eta_m[:, None] # (M, N)

    lambda_acc = (alpha[:, None, None, None]
                  * speed_factor[:, :, None, None]
                  * acc_filter[None, None, :, :])                # (M, N, X, L)
    # Dirichlet upper bound: no acceleration from top speed bin
    lambda_acc[:, N - 1, :, :] = 0.0
    # Bs Acceleration Blockade: lambda_acc^(Bs)[i >= i_thr] = 0
    lambda_acc[2, i_thr:, :, :] = 0.0

    # Deceleration rates lambda_dec^(m,i):
    # Interaction sum: sum_n sum_{k<i} beta^(m,n)*(v_i-v_k)*w^(n)*f_k^(n) / rho_max
    beta_w = beta * w[None, :]                                   # (M, M) = beta[m,n]*w[n]
    # bwf[m, k, x, l] = sum_n beta_w[m,n] * f[n,k,x,l] / rho_max
    bwf = np.einsum('mn,nkxl->mkxl', beta_w, f_new) / rho_max  # (M, N, X, L)

    # Prefix sums (shifted right so cum_bwf[m,i] = sum_{k<i} bwf[m,k])
    cum_bwf  = np.zeros_like(bwf)
    cum_vbwf = np.zeros_like(bwf)
    cum_bwf [:, 1:, :, :] = np.cumsum(bwf [:, :-1, :, :], axis=1)
    cum_vbwf[:, 1:, :, :] = np.cumsum(
        v[None, :-1, None, None] * bwf[:, :-1, :, :], axis=1)

    interaction = v[None, :, None, None] * cum_bwf - cum_vbwf   # (M, N, X, L)

    # Singular pressure barrier B(Omega)^{eta^(m)}
    pressure = (rho_max / np.maximum(eps, rho_max - omega[None, None, :, :])
                ) ** eta_m[:, None, None, None]                  # (M, N, X, L)

    lambda_dec = (omega_0[:, None, None, None] + interaction) * pressure
    lambda_dec = np.maximum(lambda_dec, 0.0)
    # Dirichlet lower bound: no deceleration below min speed
    lambda_dec[:, 0, :, :] = 0.0

    # ── Thomas algorithm: solve (I - Dt*A)*f_new = f_old ────────────────────
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
# PHASE 3 — Spatial Advection  (FVM + Global Godunov Flux Limiter)
# ─────────────────────────────────────────────────────────────────────────────
def phase3_advection(f, omega):
    """
    All classes advect at their actual speed v_i.

    Demand Psi^(m):
      Psi_{i,x->x+1,l}^(m) = v_i * f_{i,x,l}^(m) * [1-exp(-max(0,rho_max-Omega_{x+1,l})/R_supply)]

    Global aggregate demand at face x->x+1:
      D_{x->x+1,l} = dt/dx * sum_{m,i} w^(m) * Psi^(m)_{i,x,l}

    Global Godunov flux limiter alpha:
      alpha_{x->x+1,l} = min(1, max(0, rho_max-Omega_{x+1,l}) / D)  if D > 0,  else 1

    Final flux:
      Phi^(m) = alpha * Psi^(m)

    Returns f_new (M,N,X,L), phi (M,N,X+1,L)
    """
    phi = np.zeros((M, N, X + 1, L), dtype=np.float64)

    # ── Supply filter at downstream cell (faces 1..X-1) ─────────────────────
    supply_arg    = np.maximum(0.0, rho_max - omega[1:X, :])           # (X-1, L)
    supply_factor = 1.0 - np.exp(-supply_arg / R_supply)               # (X-1, L)

    # ── Demand Psi at internal faces 1..X-1 ──────────────────────────────────
    Psi_internal = (v[None, :, None, None]
                    * f[:, :, :X - 1, :]
                    * supply_factor[None, None, :, :])                  # (M, N, X-1, L)

    # ── Global aggregate demand D_{face, l} ──────────────────────────────────
    D = ((dt / dx)
         * (w[:, None, None, None] * Psi_internal).sum(axis=(0, 1)))   # (X-1, L)

    # ── Available space at downstream cell ───────────────────────────────────
    available = np.maximum(0.0, rho_max - omega[1:X, :])               # (X-1, L)

    # ── Global Godunov limiter alpha ──────────────────────────────────────────
    alpha_g = np.where(D > eps, np.minimum(1.0, available / D), 1.0)  # (X-1, L)

    # ── Apply alpha to all classes simultaneously ────────────────────────────
    phi[:, :, 1:X, :] = Psi_internal * alpha_g[None, None, :, :]

    # ── Ring road periodic boundary ──────────────────────────────────────────
    # face X: vehicles leaving cell X-1 enter cell 0 (periodic).
    # Apply supply filter and Godunov limiter using omega[0] as downstream.
    supply_arg_X    = np.maximum(0.0, rho_max - omega[0, :])            # (L,)
    supply_factor_X = 1.0 - np.exp(-supply_arg_X / R_supply)           # (L,)
    Psi_X = (v[None, :, None] * f[:, :, X - 1, :]
             * supply_factor_X[None, None, :])                          # (M, N, L)
    D_X     = (dt / dx) * (w[:, None, None] * Psi_X).sum(axis=(0, 1)) # (L,)
    avail_X = np.maximum(0.0, rho_max - omega[0, :])                   # (L,)
    alpha_X = np.where(D_X > eps,
                       np.minimum(1.0, avail_X / D_X), 1.0)            # (L,)
    phi[:, :, X, :] = Psi_X * alpha_X[None, None, :]

    # face 0: what enters cell 0 from the "left" = what left cell X-1 (periodic)
    phi[:, :, 0, :] = phi[:, :, X, :]

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
    Verify DeltaP^(B) approx net boundary flux for combined B class (Bf+Bs).
    Capture/release is internal zero-sum (proven in theorem).
    """
    mass_B_old = ((f_old[1] + f_old[2]) * w[1]).sum() * dx
    mass_B_new = ((f_new[1] + f_new[2]) * w[1]).sum() * dx
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
    pg.attrs['model']         = 'Autonomous Multiclass TRM m3+m4 P_block'
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
    pg.attrs['R_c']           = R_c
    pg.attrs['eps']           = eps
    pg.attrs['v_A_ff_mps']    = v_A_ff
    pg.attrs['i_thr']         = i_thr
    pg.attrs['eta_block']     = eta_block
    pg.attrs['sigma_0']       = sigma_0
    pg.attrs['mu_0']          = mu_0
    pg.attrs['R_A']           = R_A
    pg.attrs['omega_0_BA']    = omega_0_BA
    pg.attrs['beta_BA']       = beta_BA
    pg.attrs['CFL']           = cfl
    pg.attrs['operator_split'] = '3-phase: Capture/Release (P_block) -> Kinematics -> Advection'

    pg['v_mps']      = v
    pg['w_PCE']      = w
    pg['alpha_hz']   = alpha
    pg['eta_m']      = eta_m
    pg['omega_0_hz'] = omega_0
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
    mk('P_block',     (X, L))          # probabilistic blocking factor (Phase 1)
    mk('sigma',       (N, X, L))       # capture rate (Phase 1)
    mk('mu',          (X, L))          # release rate (Phase 1)
    mk('kappa_star',  (X, L))          # truck speed threshold per cell (int stored as float)
    mk('rho_macro',        (M, X, L))  # macroscopic density
    mk('q_macro',         (M, X, L))  # macroscopic flow
    mk('u_macro',         (M, X, L))  # macroscopic speed
    mk('omega_pre_phase3', (X, L))    # omega immediately before Phase 3 (exact Godunov reference)

    dg.create_dataset('time_s', data=np.arange(T) * dt)
    hf.create_dataset('diagnostics/mass_rel_error_B', shape=(T,), dtype=np.float64)
    return hf


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run(output_path):
    print("=" * 66)
    print("  Autonomous Multiclass TRM -- m3+m4 Probabilistic Blocking")
    print(f"  Grid: {X}x{L} cells/lanes, {N} speeds, M={M} classes")
    print(f"  Steps: {T_STEPS}  dt={dt}s  ->  {T_STEPS*dt:.0f}s simulation")
    print(f"  i_thr={i_thr}  (v_A_ff={v_A_ff} m/s)  eta_block={eta_block}")
    print(f"  3-Phase: P_block CaptureRelease -> Kinematics -> Advection")
    print(f"  Output: {output_path}")
    print("=" * 66)

    # Estimate file size (no gamma datasets, added P_block)
    bytes_per_step = (
        M*N*X*L*8 +       # f
        X*L*8 +           # omega
        M*N*(X+1)*L*8 +   # phi
        M*N*X*L*8*2 +     # lambda_acc + lambda_dec
        X*L*8 +           # P_block
        N*X*L*8 +         # sigma
        X*L*8*2 +         # mu + kappa_star
        M*X*L*8*3         # rho, q, u macro
    )
    print(f"  Estimated size: {T_STEPS * bytes_per_step / 1024**2:.0f} MB\n")

    f      = initialize_state()
    f_old  = f.copy()
    hf     = setup_hdf5(output_path, T_STEPS)
    dg     = hf['data']
    diag   = hf['diagnostics/mass_rel_error_B']

    t_wall = time.perf_counter()

    for t in range(T_STEPS):
        t0 = time.perf_counter()

        omega = compute_omega(f)

        # Phase 1: Exact Capture & Release (P_block-based)
        f, sigma, mu, P_block = phase1_capture_release(f, omega)

        # Phase 2: Algebraic Projection + Thomas Kinematics
        omega = compute_omega(f)
        f, lambda_acc, lambda_dec, kappa_star = phase2_kinematics(f, omega)

        # Phase 3: Spatial Advection (global Godunov)
        omega = compute_omega(f)
        omega_pre3 = omega.copy()          # exact reference for V3-e Godunov check
        f, phi = phase3_advection(f, omega)

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
        dg['P_block'][t]      = P_block
        dg['sigma'][t]        = sigma
        dg['mu'][t]           = mu
        dg['kappa_star'][t]   = kappa_star.astype(np.float64)
        dg['rho_macro'][t]        = rho_m
        dg['q_macro'][t]          = q_m
        dg['u_macro'][t]          = u_m
        dg['omega_pre_phase3'][t] = omega_pre3
        diag[t]               = rel_err

        if t % 50 == 0 or t == T_STEPS - 1:
            step_ms = (time.perf_counter() - t0) * 1000
            eta_s   = (time.perf_counter() - t_wall) / max(t + 1, 1) * (T_STEPS - t - 1)
            o_peak  = omega_final.max()
            Bs_tot  = f[2].sum()
            print(f"  step {t:4d}/{T_STEPS}  |  {step_ms:5.1f}ms  |"
                  f"  ETA {eta_s:5.0f}s  |"
                  f"  Omega_peak={o_peak:.4f}  |"
                  f"  f_Bs_total={Bs_tot:.4f}  |"
                  f"  mass_err_B={rel_err:.2e}")

        if t % 100 == 0:
            hf.flush()

    hf.attrs['description'] = (
        'm3+m4 Autonomous Multiclass TRM benchmark dataset (P_block version). '
        'Classes: A (trucks), Bf (free cars), Bs (trapped cars). '
        '3-phase Lie-Trotter: Capture/Release (P_block exact exp) -> '
        'Kinematics+Projection (Thomas) -> Advection (global Godunov). '
        'No lateral phase. Lanes independent. '
        'Ring road (periodic BC): A bottleneck cells 74-79, Bf uniform upstream x=0-73 (rho=0.020, v=30m/s). No external inflow. Classic Riemann IC: left=free flow, right=sparse.'
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
