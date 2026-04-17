"""
Autonomous Multiclass TRM — No-Bottleneck Baseline
Free-flow Riemann Problem: pure acceleration / deceleration test

IC: Classic Riemann problem (no trucks)
    x=0-74  : sparse free flow, Bf at v=30 m/s, rho=0.010 PCE/m
    x=75-149: dense jam,        Bf at v=2  m/s, rho=0.070 PCE/m

Expected dynamics:
  - Shock at x=75 propagates BACKWARD at ~-2.7 m/s (congestion spreading upstream)
  - Rarefaction at downstream jam edge: slow cars exit jam and ACCELERATE
  - No capture/release (sigma=0 everywhere, no trucks)
  - Demonstrates Phase 2 kinematics working correctly without moving bottleneck

Physics: IDENTICAL to generate_dataset.py (same 3-phase splitting, same parameters).
         Only initialize_state() is different.
"""

import numpy as np
import h5py
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  (same as Benchmark Dataset.json)
# ─────────────────────────────────────────────────────────────────────────────
X       = 150
L       = 3
N       = 15
M       = 3

dx      = 20.0
dt      = 0.5
T_STEPS = 500        # 250 s simulation

v = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
              18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0], dtype=np.float64)
v_max   = 30.0

rho_max  = 0.15
R_supply = 0.035
R_c      = 0.05
eps      = 1.0e-8

w     = np.array([2.5, 1.0, 1.0], dtype=np.float64)
alpha = np.array([0.35, 1.50, 1.50], dtype=np.float64)
eta_m = np.array([4.5,  2.0,  2.0 ], dtype=np.float64)
omega_0 = np.array([0.05, 0.01, 0.01], dtype=np.float64)

beta = np.array([
    [0.12, 0.08, 0.08],
    [0.06, 0.03, 0.03],
    [0.06, 0.03, 0.03],
], dtype=np.float64)

v_A_ff     = 14.0
i_thr      = int(np.searchsorted(v, v_A_ff, side='right') - 1)
eta_block  = 2.0
omega_0_BA = 0.05
beta_BA    = 0.06
sigma_0    = 0.5
mu_0       = 0.3
R_A        = 0.05

cfl = dt * v_max / dx
assert cfl <= 1.0, f"CFL violated: {cfl:.4f}"
print(f"CFL = {cfl:.4f}  OK   i_thr = {i_thr}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def compute_omega(f):
    return (w[:, None, None, None] * f).sum(axis=(0, 1))


def safe_phi(z):
    """Piecewise φ(z)=(1-e^{-z})/z — matches tex Phase-1 eq. eps_phi=1e-5."""
    eps_phi = 1.0e-5
    taylor  = 1.0 - z / 2.0 + z ** 2 / 6.0
    exact   = -np.expm1(-z) / np.where(z > eps_phi, z, 1.0)
    return np.where(z > eps_phi, exact, taylor)


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION — Riemann IC, NO TRUCKS
# ─────────────────────────────────────────────────────────────────────────────
def initialize_state():
    """
    Riemann problem without moving bottleneck.

    Left  (x=0-74):   sparse free flow  — Bf at v=30 m/s, rho=0.010
    Right (x=75-149): dense jam          — Bf at v=2  m/s, rho=0.070

    Estimated shock speed (Rankine-Hugoniot):
      Q_L = 0.010 x 30 = 0.300,  Q_R = 0.070 x 2 = 0.140
      v_shock = (0.140-0.300)/(0.070-0.010) = -2.67 m/s  (backward ~33 cells in 250s)
    """
    f = np.full((M, N, X, L), 1.0e-5, dtype=np.float64)

    # Class A (trucks): NONE — no moving bottleneck
    # f[0, ...] stays at background noise (effectively zero)

    # Class Bf: Riemann discontinuity at x=75
    f[1, 14, 0:75,  :] = 0.010   # upstream: sparse, v=30 m/s
    f[1,  0, 75:150, :] = 0.070  # downstream: dense jam, v=2 m/s

    # Class Bs: zero
    f[2, :, :, :] = 0.0

    f = np.clip(f, 0.0, rho_max)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Exact Capture & Release  (identical physics, but sigma→0 since A=0)
# ─────────────────────────────────────────────────────────────────────────────
def phase1_capture_release(f, omega):
    f_new = f.copy()

    omega_norm = np.clip(omega, 0.0, rho_max) / rho_max
    P_block    = omega_norm ** eta_block

    indicator  = (v[:, None] >= v[None, :i_thr + 1]).astype(np.float64)
    speed_diff = np.maximum(0.0, v[:, None] - v[None, :i_thr + 1])
    weight_ik  = indicator * (omega_0_BA + beta_BA * speed_diff)
    fA_trucks  = w[0] * f[0, :i_thr + 1, :, :] / rho_max           # ≈ 0 (no trucks)
    A_exposure = np.einsum('ik,kxl->ixl', weight_ik, fA_trucks)     # ≈ 0

    # sigma = sigma_0 * P_block * A^(i)  [tex Eq.10 — no B(Omega_x)]
    sigma = sigma_0 * P_block[None, :, :] * A_exposure              # ≈ 0 (no trucks)
    sigma = np.maximum(sigma, 0.0)

    theta_A = (w[0] * f[0, :i_thr + 1, :, :] / rho_max).sum(axis=0)  # ≈ 0
    S_tilde = P_block * theta_A                                          # ≈ 0
    mu      = mu_0 * np.exp(-S_tilde / R_A)                             # ≈ mu_0

    S_total = sigma + mu[None, :, :]
    F_total = f[1] + f[2]
    phi_z   = safe_phi(S_total * dt)

    f_new[1] = f[1] * np.exp(-S_total * dt) + mu[None, :, :] * F_total * dt * phi_z
    f_new[2] = F_total - f_new[1]
    f_new[1] = np.maximum(f_new[1], 0.0)
    f_new[2] = np.maximum(f_new[2], 0.0)

    return f_new, sigma, mu, P_block


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Kinematics  (Thomas algorithm — identical)
# ─────────────────────────────────────────────────────────────────────────────
def phase2_kinematics(f, omega):
    f_new = f.copy()

    thresh      = 1.0e-10
    truck_present = f[0, :i_thr + 1, :, :] > thresh
    truck_flipped = truck_present[::-1, :, :]
    argmax_flip   = np.argmax(truck_flipped, axis=0)
    kappa_star    = i_thr - argmax_flip
    any_truck     = truck_present.any(axis=0)
    kappa_star    = np.where(any_truck, kappa_star, i_thr)   # default i_thr if no trucks

    speed_idx  = np.arange(N)
    high_mask  = speed_idx[:, None, None] > kappa_star[None, :, :]
    excess     = f_new[2] * high_mask
    excess_total = excess.sum(axis=0)
    kappa_mask = speed_idx[:, None, None] == kappa_star[None, :, :]

    f_new[2] = f_new[2] - excess + kappa_mask * excess_total[None, :, :]
    f_new[2] = np.maximum(f_new[2], 0.0)

    # Acceleration: downstream Omega_{x+1} (ring road wrap for last cell)
    omega_down   = np.roll(omega, -1, axis=0)
    supply_kin   = np.maximum(0.0, rho_max - omega_down)
    acc_filter   = 1.0 - np.exp(-supply_kin / R_c)
    speed_factor = (1.0 - v[None, :] / v_max) ** eta_m[:, None]

    lambda_acc = (alpha[:, None, None, None]
                  * speed_factor[:, :, None, None]
                  * acc_filter[None, None, :, :])
    lambda_acc[:, N - 1, :, :] = 0.0
    lambda_acc[2, i_thr:, :, :] = 0.0

    beta_w = beta * w[None, :]
    bwf    = np.einsum('mn,nkxl->mkxl', beta_w, f_new) / rho_max

    cum_bwf  = np.zeros_like(bwf)
    cum_vbwf = np.zeros_like(bwf)
    cum_bwf [:, 1:, :, :] = np.cumsum(bwf [:, :-1, :, :], axis=1)
    cum_vbwf[:, 1:, :, :] = np.cumsum(
        v[None, :-1, None, None] * bwf[:, :-1, :, :], axis=1)

    interaction = v[None, :, None, None] * cum_bwf - cum_vbwf

    pressure = (rho_max / np.maximum(eps, rho_max - omega[None, None, :, :])
                ) ** eta_m[:, None, None, None]

    lambda_dec = (omega_0[:, None, None, None] + interaction) * pressure
    lambda_dec = np.maximum(lambda_dec, 0.0)
    lambda_dec[:, 0, :, :] = 0.0

    a = np.zeros((M, N, X, L), dtype=np.float64)
    b = 1.0 + dt * (lambda_acc + lambda_dec)
    c = np.zeros((M, N, X, L), dtype=np.float64)
    d = f_new.copy()

    a[:, 1:,  :, :] = -dt * lambda_acc[:, :-1, :, :]
    c[:, :-1, :, :] = -dt * lambda_dec[:, 1:,  :, :]

    c_p = np.zeros_like(c)
    d_p = np.zeros_like(d)
    c_p[:, 0, :, :] = c[:, 0, :, :] / b[:, 0, :, :]
    d_p[:, 0, :, :] = d[:, 0, :, :] / b[:, 0, :, :]
    for i in range(1, N):
        denom = b[:, i, :, :] - a[:, i, :, :] * c_p[:, i - 1, :, :]
        denom = np.maximum(denom, eps)
        c_p[:, i, :, :] = c[:, i, :, :] / denom
        d_p[:, i, :, :] = (d[:, i, :, :] - a[:, i, :, :] * d_p[:, i - 1, :, :]) / denom

    f_out = np.zeros_like(f_new)
    f_out[:, N - 1, :, :] = d_p[:, N - 1, :, :]
    for i in range(N - 2, -1, -1):
        f_out[:, i, :, :] = d_p[:, i, :, :] - c_p[:, i, :, :] * f_out[:, i + 1, :, :]

    f_out = np.maximum(f_out, 0.0)
    return f_out, lambda_acc, lambda_dec, kappa_star


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Advection  (identical — ring road periodic BC)
# ─────────────────────────────────────────────────────────────────────────────
def phase3_advection(f, omega):
    phi = np.zeros((M, N, X + 1, L), dtype=np.float64)

    eps_tol = 1.0e-12
    # Pure demand Psi = v_i * f  [tex Eq.13 — no supply pre-filter]
    Psi_internal  = v[None, :, None, None] * f[:, :, :X - 1, :]
    D         = ((dt / dx) * (w[:, None, None, None] * Psi_internal).sum(axis=(0, 1)))
    available = np.maximum(0.0, rho_max - omega[1:X, :])
    alpha_g   = np.minimum(1.0, (available + eps_tol) / (D + eps_tol))
    phi[:, :, 1:X, :] = Psi_internal * alpha_g[None, None, :, :]

    # Ring road face X — pure demand from cell X-1, downstream = cell 0
    Psi_X   = v[None, :, None] * f[:, :, X - 1, :]
    D_X     = (dt / dx) * (w[:, None, None] * Psi_X).sum(axis=(0, 1))
    avail_X = np.maximum(0.0, rho_max - omega[0, :])
    alpha_X = np.minimum(1.0, (avail_X + eps_tol) / (D_X + eps_tol))
    phi[:, :, X, :] = Psi_X * alpha_X[None, None, :]
    phi[:, :, 0, :] = phi[:, :, X, :]

    f_new = f + (dt / dx) * (phi[:, :, :X, :] - phi[:, :, 1:X + 1, :])
    f_new = np.maximum(f_new, 0.0)
    return f_new, phi


def compute_macroscopic(f):
    rho_macro = f.sum(axis=1)
    q_macro   = (v[None, :, None, None] * f).sum(axis=1)
    u_macro   = q_macro / np.maximum(eps, rho_macro)
    return rho_macro, q_macro, u_macro


def check_mass(f_old, f_new, phi):
    mass_B_old = ((f_old[1] + f_old[2]) * w[1]).sum() * dx
    mass_B_new = ((f_new[1] + f_new[2]) * w[1]).sum() * dx
    net_flux = ((w[1] * phi[1, :, X, :] + w[2] * phi[2, :, X, :]).sum()
                - (w[1] * phi[1, :, 0, :] + w[2] * phi[2, :, 0, :]).sum()) * dt
    residual = abs((mass_B_new - mass_B_old) + net_flux)
    return residual / (mass_B_old + eps)


# ─────────────────────────────────────────────────────────────────────────────
# HOVMÖLLER VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def plot_hovmoller(h5_path, fig_path):
    with h5py.File(h5_path, 'r') as hf:
        T    = hf['parameters'].attrs['T_steps']
        time_s = hf['data/time_s'][:]
        omega_all = hf['data/omega'][:]        # (T, X, L)
        u_all     = hf['data/u_macro'][:, 1, :, :]  # Bf mean speed (T, X, L)

    # Lane-average
    omega_avg = omega_all.mean(axis=2)    # (T, X)
    u_avg     = u_all.mean(axis=2)        # (T, X)

    x_cells = np.arange(X)   # cell index 0–149
    t_axis  = time_s          # [s]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('No-Bottleneck Baseline — Riemann IC\n'
                 r'Sparse free flow (cell 0–74) | Dense jam (cell 75–149)',
                 fontsize=12, fontweight='bold')

    x_interface = 75          # cell index
    interface_color = 'cyan'

    # Panel 1: Occupancy Hovmöller
    ax = axes[0]
    im = ax.pcolormesh(x_cells, t_axis, omega_avg,
                       cmap='turbo', vmin=0, vmax=rho_max, shading='auto')
    plt.colorbar(im, ax=ax, label=r'$\Omega$ [PCE/m]')
    ax.axvline(x_interface, color=interface_color, ls='--', lw=1.5,
               label='Initial interface (cell 75)')
    # Theoretical shock: v_shock = -2.67 m/s → -2.67/dx cells/s from cell 75
    t_arr   = np.array([0, T_STEPS * dt])
    x_shock = x_interface + (-2.67 / dx) * t_arr   # [cells]
    ax.plot(x_shock, t_arr, color='white', lw=1.5, alpha=0.9, label='Theoretical shock')
    ax.set_xlabel('Cell x')
    ax.set_ylabel('Time [s]')
    ax.set_title(r'Occupancy $\Omega(x,t)$  [Hovmöller]')
    ax.legend(fontsize=8, loc='upper right')

    # Panel 2: Mean speed Hovmöller
    ax = axes[1]
    im2 = ax.pcolormesh(x_cells, t_axis, u_avg,
                        cmap='RdYlGn', vmin=0, vmax=v_max, shading='auto')
    plt.colorbar(im2, ax=ax, label='Mean speed [m/s]')
    ax.axvline(x_interface, color=interface_color, ls='--', lw=1.5,
               label='Initial interface (cell 75)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlabel('Cell x')
    ax.set_ylabel('Time [s]')
    ax.set_title(r'Bf Mean Speed $u^{(Bf)}(x,t)$')

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Hovmöller saved: {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run(output_path):
    print("=" * 66)
    print("  Multiclass TRM — No-Bottleneck Baseline (Riemann IC)")
    print(f"  Grid: {X}x{L}, {N} speeds, M={M} classes")
    print(f"  Steps: {T_STEPS}  dt={dt}s  ->  {T_STEPS*dt:.0f}s")
    print(f"  IC: Bf sparse (rho=0.010, v=30m/s) | Bf dense (rho=0.070, v=2m/s)")
    print(f"  No trucks. Periodic (ring road) BC.")
    print(f"  Output: {output_path}")
    print("=" * 66)

    # Minimal HDF5 — store only f, omega, u_macro
    f     = initialize_state()
    f_old = f.copy()

    hf = h5py.File(output_path, 'w')
    pg = hf.create_group('parameters')
    pg.attrs['T_steps'] = T_STEPS
    pg.attrs['dt_s']    = dt
    pg.attrs['dx_m']    = dx
    pg.attrs['X']       = X
    pg.attrs['L']       = L
    pg.attrs['N']       = N
    pg.attrs['rho_max'] = rho_max
    pg.attrs['ic']      = 'Riemann: Bf sparse x=0-74 (rho=0.010,v=30) | dense x=75-149 (rho=0.070,v=2). No trucks.'

    dg = hf.create_group('data')
    shape_f = (T_STEPS, M, N, X, L)
    shape_o = (T_STEPS, X, L)
    shape_u = (T_STEPS, M, X, L)

    ds_f = dg.create_dataset('f',       shape_f, dtype=np.float64, chunks=(1, M, N, X, L))
    ds_o = dg.create_dataset('omega',   shape_o, dtype=np.float64, chunks=(1, X, L))
    ds_u = dg.create_dataset('u_macro', shape_u, dtype=np.float64, chunks=(1, M, X, L))
    dg.create_dataset('time_s', data=np.arange(T_STEPS) * dt)

    t_wall = time.perf_counter()

    for t in range(T_STEPS):
        omega = compute_omega(f)

        # Phase 1 (sigma≈0 since no trucks)
        f, sigma, mu, P_block = phase1_capture_release(f, omega)

        # Phase 2 (kinematics — acceleration/deceleration)
        omega = compute_omega(f)
        f, lambda_acc, lambda_dec, kappa_star = phase2_kinematics(f, omega)

        # Phase 3 (advection)
        omega = compute_omega(f)
        f, phi = phase3_advection(f, omega)

        omega_final = compute_omega(f)
        rho_m, q_m, u_m = compute_macroscopic(f)
        f_old = f.copy()

        ds_f[t] = f
        ds_o[t] = omega_final
        ds_u[t] = u_m

        if t % 50 == 0 or t == T_STEPS - 1:
            eta_s = ((time.perf_counter() - t_wall) / max(t + 1, 1)
                     * (T_STEPS - t - 1))
            u_Bf_mean = float(u_m[1].mean())
            print(f"  step {t:4d}/{T_STEPS}  |  Omega_peak={omega_final.max():.4f}"
                  f"  |  u_Bf_mean={u_Bf_mean:.2f} m/s  |  ETA {eta_s:.0f}s")

        if t % 100 == 0:
            hf.flush()

    hf.close()
    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"\n  Done!  Wall time: {time.perf_counter()-t_wall:.1f}s  |  File: {size_mb:.1f} MB")
    print("=" * 66)


if __name__ == '__main__':
    code_dir = os.path.dirname(os.path.abspath(__file__))
    out_h5   = os.path.join(code_dir, 'multiclass_trm_nobot.h5')
    out_fig  = os.path.join(code_dir, 'figures', 'V_nobot_hovmoller.png')

    run(out_h5)
    plot_hovmoller(out_h5, out_fig)
