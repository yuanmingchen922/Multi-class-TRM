"""
Autonomous Multiclass TRM — Benchmark Dataset Generator
Author: Mingchen Yuan

Generates ~500 MB HDF5 dataset by simulating the Autonomous Multiclass TRM
for 800 timesteps using the three-phase Lie-Trotter operator splitting scheme:
  Phase 1: Explicit Spatial Advection (FVM, positivity-preserving)
  Phase 2: Lateral Lane-Changing (Softplus, explicit Euler)
  Phase 3: Implicit Kinematics (Semi-implicit Thomas algorithm)

All parameters strictly follow Benchmark_Dataset.json and Multi-class_TRM.tex.
"""

import numpy as np
import h5py
import time
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (strictly from Benchmark Dataset.json)
# ─────────────────────────────────────────────────────────────────────────────
X       = 150        # spatial cells
L       = 3          # lanes
N       = 15         # speed categories
M       = 2          # vehicle classes  (0=PC, 1=HDT)

dx      = 20.0       # cell length [m]
dt      = 0.5        # time step [s]
T_STEPS = 800        # total time steps → ~400 s simulation

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
eta_g    = 2.0       # global acceleration decay exponent

# Per-class parameters: index 0 = PC, index 1 = HDT
w       = np.array([1.0,  2.5 ])   # PCE spatial weight
alpha   = np.array([1.50, 0.35])   # base acceleration rate [Hz]
eta_m   = np.array([2.0,  4.5 ])   # singular barrier stiffness exponent
omega_0 = np.array([0.01, 0.05])   # spontaneous anticipation rate [Hz]
kappa   = np.array([0.60, 0.08])   # lateral agility [Hz]

# Beta kinetic collision kernel (2×2), beta[m, n]
beta = np.array([[0.03, 0.06],
                 [0.08, 0.12]], dtype=np.float64)  # [m^-1]

# CFL validation
cfl = dt * v_max / dx
assert cfl <= 1.0, f"CFL violated: {cfl:.4f} > 1.0"
print(f"CFL = {cfl:.4f}  ✓  (dt={dt}, dx={dx}, v_max={v_max})")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: effective occupancy  Ω_{x,l} = Σ_m Σ_i  w^(m) f_{i,x,l}^(m)
# ─────────────────────────────────────────────────────────────────────────────
def compute_omega(f):
    """f: (M, N, X, L)  →  omega: (X, L)"""
    return (w[:, None, None, None] * f).sum(axis=(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION  (benchmark scenario from JSON)
# ─────────────────────────────────────────────────────────────────────────────
def initialize_state():
    """Return initial f of shape (M, N, X, L)."""
    f = np.full((M, N, X, L), 1.0e-5, dtype=np.float64)

    # Downstream bottleneck (cells 74-79, 0-indexed): HDT at minimum speed
    # Class HDT = index 1,  speed index 0 (v=2 m/s)
    # Density 0.058 veh/m → Ω = 2.5 * 0.058 = 0.145 ≈ ρ_max   (extreme stiffness)
    f[1, 0, 74:80, :] = 0.058

    # Upstream injection zone (cells 59-69): PC at maximum speed
    # Class PC = index 0,  speed index 14 (v=30 m/s)
    f[0, 14, 59:70, :] = 0.060

    # Clip to physical bounds
    f = np.clip(f, 0.0, rho_max)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Explicit Spatial Advection  (FVM, positivity-preserving)
# ─────────────────────────────────────────────────────────────────────────────
def phase1_advection(f, omega):
    """
    Compute inter-cell fluxes and update f explicitly.

    Flux formula (eq. 4):
      Φ_{i,x→x+1,l}^(m) = v_i · f_{i,x,l}^(m) · [1 - exp(-max(0, ρ_max - Ω_{x+1,l}) / R_supply)]

    Returns updated f (M,N,X,L) and phi (M,N,X+1,L).
    """
    # phi[:, :, face, :] is the flux across the face between cell (face-1) and face
    phi = np.zeros((M, N, X + 1, L), dtype=np.float64)

    # Internal faces 1 .. X-1:  upstream cell = face-1,  downstream cell = face
    # Downstream supply factor from omega[face, :], face in 1..X-1 → omega[1:X, :]
    supply_arg    = np.maximum(0.0, rho_max - omega[1:X, :])          # (X-1, L)
    supply_factor = 1.0 - np.exp(-supply_arg / R_supply)              # (X-1, L)

    # Upwind flux: velocity × density × supply
    phi[:, :, 1:X, :] = (v[None, :, None, None]
                          * f[:, :, 0:X-1, :]
                          * supply_factor[None, None, :, :])

    # Right boundary face X: free outflow (no downstream restriction)
    # f[:, :, X-1, :] has shape (M, N, L) → use 3D broadcast (1, N, 1)
    phi[:, :, X, :] = v[None, :, None] * f[:, :, X-1, :]
    # Left boundary face 0: zero (no upstream inflow from outside)
    # phi[:, :, 0, :] = 0  (already zero)

    # ── Positivity-preserving flux limiter ──
    # Max outgoing flux from cell x: f[m,i,x,l] * dx / dt
    # This is the maximum that can leave without going negative.
    available = f * (dx / dt)                                         # (M, N, X, L)
    phi_out   = phi[:, :, 1:X+1, :]                                   # outflow per cell (face x+1)
    ratio     = np.where(phi_out > available,
                         available / np.maximum(phi_out, eps),
                         1.0)
    # Scale the outgoing face and the same face used as inflow for the next cell
    phi[:, :, 1:X+1, :] = phi[:, :, 1:X+1, :] * ratio
    # Inflow limiter: also scale face x (the same physical face) viewed as inflow
    phi[:, :, 1:X,   :] = np.minimum(phi[:, :, 1:X, :],
                                       phi[:, :, 1:X, :])   # no-op, already limited above

    # FVM update: f_new[x] = f[x] + dt/dx * (flux_in[x] - flux_out[x])
    f_new = f + (dt / dx) * (phi[:, :, 0:X, :] - phi[:, :, 1:X+1, :])
    f_new = np.maximum(f_new, 0.0)
    return f_new, phi


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Lateral Lane-Changing  (Softplus, explicit Euler)
# ─────────────────────────────────────────────────────────────────────────────
def phase2_lateral(f, omega):
    """
    Compute smoothly differentiable lateral transition rates and update f.

    Rate formula (eq. 6):
      γ_{l→l',i}^(m) = κ^(m) · (1/ω) · ln[1+exp(ω·y)] · [1-exp(-max(0,ρ_max-Ω_{l'})/R_gap)]
      where y = (Ω_{x,l} - Ω_{x,l'}) / ρ_max

    Dirichlet BCs: γ_{1→0} = γ_{L→L+1} = 0.

    Returns updated f, gamma_left (M,N,X,L), gamma_right (M,N,X,L).
    """
    def softplus(y):
        # Numerically stable softplus: avoids overflow for large y
        return np.where(y > 20.0,
                        y,
                        np.log1p(np.exp(np.minimum(omega_sp * y, 500.0))) / omega_sp)

    # ── Rightward rates: lane l → l+1,  for l = 0..L-2 ──
    y_right = (omega[:, :-1] - omega[:, 1:]) / rho_max                # (X, L-1)
    gap_arg_right   = np.maximum(0.0, rho_max - omega[:, 1:])          # (X, L-1)
    gap_factor_right = 1.0 - np.exp(-gap_arg_right / R_gap)           # (X, L-1)
    sp_right = softplus(y_right)                                        # (X, L-1)
    # gamma for lane l (source), shape broadcast over (M, N, X, L-1)
    gr_internal = (kappa[:, None, None, None]
                   * sp_right[None, None, :, :]
                   * gap_factor_right[None, None, :, :])

    gamma_right = np.zeros((M, N, X, L), dtype=np.float64)
    gamma_right[:, :, :, 0:L-1] = gr_internal                        # lane L-1 cannot go right

    # ── Leftward rates: lane l → l-1,  for l = 1..L-1 ──
    y_left = (omega[:, 1:] - omega[:, :-1]) / rho_max                 # (X, L-1)
    gap_arg_left    = np.maximum(0.0, rho_max - omega[:, :-1])         # (X, L-1)
    gap_factor_left = 1.0 - np.exp(-gap_arg_left / R_gap)            # (X, L-1)
    sp_left = softplus(y_left)                                         # (X, L-1)
    gl_internal = (kappa[:, None, None, None]
                   * sp_left[None, None, :, :]
                   * gap_factor_left[None, None, :, :])

    gamma_left = np.zeros((M, N, X, L), dtype=np.float64)
    gamma_left[:, :, :, 1:L] = gl_internal                           # lane 0 cannot go left

    # ── Explicit Euler update ──
    # Loss from lane l: (γ_right[l] + γ_left[l]) * f[l]
    loss = (gamma_right + gamma_left) * f                             # (M, N, X, L)

    # Gain at lane l from right-moving neighbor l-1 and left-moving neighbor l+1
    gain = np.zeros_like(f)
    gain[:, :, :, 1:L]   += gamma_right[:, :, :, 0:L-1] * f[:, :, :, 0:L-1]  # l-1 → l
    gain[:, :, :, 0:L-1] += gamma_left[:, :, :, 1:L]    * f[:, :, :, 1:L]    # l+1 → l

    f_new = f + dt * (gain - loss)
    f_new = np.maximum(f_new, 0.0)
    return f_new, gamma_left, gamma_right


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Implicit Kinematics  (Thomas algorithm, semi-implicit)
# ─────────────────────────────────────────────────────────────────────────────
def phase3_kinematics(f, omega):
    """
    Resolve extreme ODE stiffness using semi-implicit (linearly implicit) Euler.
    Rates are evaluated at current f (frozen), then the linear tridiagonal system
    is solved by the Thomas algorithm for each (m, x, l) independently.

    Acceleration rate (eq. 2):
      λ_{i→i+1}^(m) = α^(m) · (1 - v_i/v_max)^η · [1-exp(-max(0,ρ_max-Ω)/R_c)]
      Dirichlet: λ_{N→N+1} = 0

    Deceleration rate (eq. 3):
      λ_{i→i-1}^(m) = (ω_0^(m) + Σ_n Σ_{k<i} β^(m,n)·(v_i-v_k)·w^(n)·f_{k}^(n))
                       · (ρ_max / max(ε, ρ_max-Ω))^{η^(m)}
      Dirichlet: λ_{1→0} = 0

    Returns updated f, lambda_acc (M,N,X,L), lambda_dec (M,N,X,L).
    """
    # ── Acceleration rates ──
    speed_factor = (1.0 - v / v_max) ** eta_g                        # (N,)
    supply_kin   = np.maximum(0.0, rho_max - omega)                   # (X, L)
    acc_filter   = 1.0 - np.exp(-supply_kin / R_c)                    # (X, L)

    lambda_acc = (alpha[:, None, None, None]
                  * speed_factor[None, :, None, None]
                  * acc_filter[None, None, :, :])                     # (M, N, X, L)
    lambda_acc[:, N-1, :, :] = 0.0    # Dirichlet: no acc from top bin

    # ── Deceleration rates ──
    # Interaction sum: Σ_n Σ_{k<i} β[m,n]·(v[i]-v[k])·w[n]·f[n,k,x,l]
    # Use prefix-sum trick to avoid O(N²) inner loop per (m,x,l).
    #   bwf[m,k,x,l] = Σ_n β[m,n]·w[n]·f[n,k,x,l]
    #   interaction[m,i,x,l] = v[i]·Σ_{k<i} bwf[m,k] - Σ_{k<i} v[k]·bwf[m,k]
    #                        = v[i]·cum_bwf[m,i] - cum_vbwf[m,i]
    #   where cum_bwf[m,i] = Σ_{k=0}^{i-1} bwf[m,k]

    # beta_w: (M, M) element-wise with w: beta[m,n]*w[n]
    beta_w = beta * w[None, :]                                         # (M, M)
    # Sum over source class n: bwf[m, k, x, l] = Σ_n beta_w[m,n]*f[n,k,x,l]
    # f has shape (M, N, X, L); sum over axis 0 with beta_w[m, n] weight
    bwf = np.einsum('mn,nkxl->mkxl', beta_w, f)                      # (M, N, X, L)

    # Prefix sums (shift by 1 so cum_bwf[m, i] = sum_{k<i})
    cum_bwf   = np.zeros_like(bwf)
    cum_vbwf  = np.zeros_like(bwf)
    cum_bwf [:, 1:, :, :] = np.cumsum(bwf [:, :-1, :, :], axis=1)
    cum_vbwf[:, 1:, :, :] = np.cumsum((v[None, :-1, None, None] * bwf[:, :-1, :, :]), axis=1)

    interaction = (v[None, :, None, None] * cum_bwf - cum_vbwf)       # (M, N, X, L)

    # Singular pressure barrier
    pressure = (rho_max / np.maximum(eps, rho_max - omega[None, None, :, :])
                ) ** eta_m[:, None, None, None]                        # (M, N, X, L)

    lambda_dec = (omega_0[:, None, None, None] + interaction) * pressure  # (M, N, X, L)
    lambda_dec = np.maximum(lambda_dec, 0.0)                          # physical positivity
    lambda_dec[:, 0, :, :] = 0.0     # Dirichlet: no dec below min bin

    # ── Thomas algorithm: solve (I - dt·A)·f_new = f_old ──
    # Tridiagonal coefficients for speed-bin system of each (m,x,l):
    #   sub-diagonal a[i]  = -dt * lambda_acc[m, i-1, x, l]  (inflow from i-1 via acc)
    #   diagonal     b[i]  =  1 + dt * (lambda_acc[m, i] + lambda_dec[m, i])
    #   super-diag   c[i]  = -dt * lambda_dec[m, i+1, x, l]  (inflow from i+1 via dec)
    #   rhs          d[i]  = f[m, i, x, l]

    a = np.zeros((M, N, X, L), dtype=np.float64)
    b = np.zeros((M, N, X, L), dtype=np.float64)
    c = np.zeros((M, N, X, L), dtype=np.float64)
    d = f.copy()

    a[:, 1:,  :, :] = -dt * lambda_acc[:, :-1, :, :]   # sub-diagonal
    b              = 1.0 + dt * (lambda_acc + lambda_dec)
    c[:, :-1, :, :] = -dt * lambda_dec[:, 1:,  :, :]   # super-diagonal

    # Boundary rows already handled (a[:,0,:,:]=0, c[:,N-1,:,:]=0)

    # Forward sweep
    c_p = np.zeros_like(c)
    d_p = np.zeros_like(d)
    c_p[:, 0, :, :] = c[:, 0, :, :] / b[:, 0, :, :]
    d_p[:, 0, :, :] = d[:, 0, :, :] / b[:, 0, :, :]
    for i in range(1, N):
        denom = b[:, i, :, :] - a[:, i, :, :] * c_p[:, i-1, :, :]
        denom = np.maximum(denom, eps)                                # guard singular pivot
        c_p[:, i, :, :] = c[:, i, :, :] / denom
        d_p[:, i, :, :] = (d[:, i, :, :] - a[:, i, :, :] * d_p[:, i-1, :, :]) / denom

    # Back substitution
    f_new = np.zeros_like(f)
    f_new[:, N-1, :, :] = d_p[:, N-1, :, :]
    for i in range(N - 2, -1, -1):
        f_new[:, i, :, :] = d_p[:, i, :, :] - c_p[:, i, :, :] * f_new[:, i+1, :, :]

    f_new = np.maximum(f_new, 0.0)
    return f_new, lambda_acc, lambda_dec


# ─────────────────────────────────────────────────────────────────────────────
# MACROSCOPIC DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_macroscopic(f):
    """
    Returns:
      rho_macro (M, X, L): class density = Σ_i f[m,i,x,l]
      q_macro   (M, X, L): class flow    = Σ_i v_i · f[m,i,x,l]
      u_macro   (M, X, L): class speed   = q / max(ε, ρ)
    """
    rho_macro = f.sum(axis=1)                                          # (M, X, L)
    q_macro   = (v[None, :, None, None] * f).sum(axis=1)              # (M, X, L)
    u_macro   = q_macro / np.maximum(eps, rho_macro)                  # (M, X, L)
    return rho_macro, q_macro, u_macro


# ─────────────────────────────────────────────────────────────────────────────
# MASS CONSERVATION CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_mass(f_old, f_new, phi):
    """
    Verifies that Δ(total PCE mass) equals the net boundary flux.
    Returns relative error.
    """
    mass_old = (w[:, None, None, None] * f_old).sum() * dx
    mass_new = (w[:, None, None, None] * f_new).sum() * dx
    # Net mass flux out through right boundary (face X) – inflow at left (face 0)
    net_flux = ((w[:, None, None] * phi[:, :, X, :]).sum()
                - (w[:, None, None] * phi[:, :, 0, :]).sum()) * dt
    residual = abs((mass_new - mass_old) + net_flux)
    rel_err  = residual / (mass_old + eps)
    return rel_err


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup_hdf5(filepath, T):
    """Pre-allocate all datasets. No compression → guarantees ~500 MB on disk."""
    hf = h5py.File(filepath, 'w')

    # ── Parameters group ──
    pg = hf.create_group('parameters')
    pg.attrs['model']       = 'Autonomous Multiclass TRM'
    pg.attrs['X']           = X
    pg.attrs['L']           = L
    pg.attrs['N']           = N
    pg.attrs['M']           = M
    pg.attrs['dx_m']        = dx
    pg.attrs['dt_s']        = dt
    pg.attrs['T_steps']     = T
    pg.attrs['v_max_mps']   = v_max
    pg.attrs['rho_max']     = rho_max
    pg.attrs['R_supply']    = R_supply
    pg.attrs['R_gap']       = R_gap
    pg.attrs['R_c']         = R_c
    pg.attrs['omega_sp']    = omega_sp
    pg.attrs['eps']         = eps
    pg.attrs['eta_global']  = eta_g
    pg.attrs['CFL']         = cfl
    pg['v_mps']             = v
    pg['w_PCE']             = w
    pg['alpha_hz']          = alpha
    pg['eta_m']             = eta_m
    pg['omega_0_hz']        = omega_0
    pg['kappa_hz']          = kappa
    pg['beta_matrix']       = beta

    # ── Data group: pre-allocate with chunking per time step ──
    dg = hf.create_group('data')

    def mk(name, shape_per_t, dtype=np.float64):
        full  = (T,) + shape_per_t
        chunk = (1,) + shape_per_t
        dg.create_dataset(name, shape=full, dtype=dtype, chunks=chunk)

    mk('f',            (M, N, X, L))      # primary state
    mk('omega',        (X, L))            # effective occupancy
    mk('phi',          (M, N, X+1, L))   # spatial fluxes (X+1 faces)
    mk('lambda_acc',   (M, N, X, L))      # acceleration rates
    mk('lambda_dec',   (M, N, X, L))      # deceleration rates
    mk('gamma_left',   (M, N, X, L))      # lateral rate  l → l-1
    mk('gamma_right',  (M, N, X, L))      # lateral rate  l → l+1
    mk('rho_macro',    (M, X, L))         # macroscopic class density
    mk('q_macro',      (M, X, L))         # macroscopic class flow
    mk('u_macro',      (M, X, L))         # macroscopic class speed

    # Time vector
    dg.create_dataset('time_s', data=np.arange(T) * dt)

    # Mass conservation log
    hf.create_dataset('diagnostics/mass_rel_error', shape=(T,), dtype=np.float64)

    return hf


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run(output_path):
    print("=" * 62)
    print("  Autonomous Multiclass TRM — Dataset Generator")
    print(f"  Grid: {X} cells × {L} lanes × {N} speeds × {M} classes")
    print(f"  Steps: {T_STEPS}  dt={dt}s  dx={dx}m  → {T_STEPS*dt:.0f}s simulation")
    print(f"  Output: {output_path}")
    print("=" * 62)

    # Estimated uncompressed size
    bytes_per_step = (M*N*X*L + X*L + M*N*(X+1)*L
                      + M*N*X*L*4 + M*X*L*3) * 8
    est_mb = T_STEPS * bytes_per_step / 1024**2
    print(f"  Estimated uncompressed size: {est_mb:.0f} MB")

    f     = initialize_state()
    f_old = f.copy()

    hf   = setup_hdf5(output_path, T_STEPS)
    dg   = hf['data']
    diag = hf['diagnostics/mass_rel_error']

    # Small upstream inflow: PC entering at max speed into lane 1 (index 0)
    # Applied after Phase 1 to represent open-highway upstream demand
    INFLOW_PC_DENSITY = 0.001   # veh/m per second (gentle free-flow injection)

    t_wall_start = time.perf_counter()

    for t in range(T_STEPS):
        t_step_start = time.perf_counter()

        # ── Effective occupancy before Phase 1 ──
        omega = compute_omega(f)

        # ── Phase 1: Spatial Advection ──
        f, phi = phase1_advection(f, omega)

        # Open upstream boundary: inject PC at max speed into all lanes
        f[0, N-1, 0, :] += INFLOW_PC_DENSITY * dt
        f = np.minimum(f, rho_max)     # hard cap (rare, safety guard)

        # ── Update omega after advection ──
        omega = compute_omega(f)

        # ── Phase 2: Lateral Lane-Changing ──
        f, gamma_left, gamma_right = phase2_lateral(f, omega)

        # ── Update omega after lateral ──
        omega = compute_omega(f)

        # ── Phase 3: Implicit Kinematics ──
        f, lambda_acc, lambda_dec = phase3_kinematics(f, omega)

        # Final omega for storage
        omega_final = compute_omega(f)

        # ── Macroscopic diagnostics ──
        rho_m, q_m, u_m = compute_macroscopic(f)

        # ── Mass conservation ──
        rel_err = check_mass(f_old, f, phi)
        f_old   = f.copy()

        # ── Write to HDF5 ──
        dg['f'][t]           = f
        dg['omega'][t]       = omega_final
        dg['phi'][t]         = phi
        dg['lambda_acc'][t]  = lambda_acc
        dg['lambda_dec'][t]  = lambda_dec
        dg['gamma_left'][t]  = gamma_left
        dg['gamma_right'][t] = gamma_right
        dg['rho_macro'][t]   = rho_m
        dg['q_macro'][t]     = q_m
        dg['u_macro'][t]     = u_m
        diag[t]              = rel_err

        # ── Progress log every 50 steps ──
        if t % 50 == 0 or t == T_STEPS - 1:
            step_ms = (time.perf_counter() - t_step_start) * 1000
            eta_s   = (time.perf_counter() - t_wall_start) / max(t+1, 1) * (T_STEPS - t - 1)
            peak_O  = omega_final.max()
            print(f"  step {t:4d}/{T_STEPS}  |  wall {step_ms:5.1f}ms  |"
                  f"  ETA {eta_s:5.0f}s  |  Ω_peak={peak_O:.4f}  |  mass_err={rel_err:.2e}")

        # Flush HDF5 every 100 steps
        if t % 100 == 0:
            hf.flush()

    # ── Finalise ──
    hf.attrs['description'] = (
        'Autonomous Multiclass TRM benchmark dataset. '
        '3-phase Lie-Trotter operator splitting. '
        'Strict kinematic tridiagonal semi-implicit Phase 3 (Thomas algorithm). '
        'Spatial FVM with positivity-preserving flux limiter. '
        'Softplus lateral lane-change dynamics. '
        'Benchmark scenario: HDT bottleneck (cells 74-79) + PC injection (cells 59-69).'
    )
    hf.close()

    file_size_mb = os.path.getsize(output_path) / 1024**2
    total_s = time.perf_counter() - t_wall_start
    print("=" * 62)
    print(f"  Done!  Wall time: {total_s:.1f}s")
    print(f"  File size on disk: {file_size_mb:.1f} MB")
    print(f"  Timesteps stored: {T_STEPS}")
    print(f"  Simulation duration: {T_STEPS * dt:.0f} s")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    output = os.path.join(os.path.dirname(__file__),
                          'multiclass_trm_benchmark_500mb.h5')
    run(output)
