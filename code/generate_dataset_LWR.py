"""
LWR Multiclass Variant — First-order model for isolating Phase 1 dynamics.

Purpose (per supervisor request):
  Strip out the second-order speed dynamics (Phase 2 kinematics) and the
  PCE-weighted supply filter (Phase 3), replacing them with:
    • Instantaneous speed = v(ρ) via triangular fundamental diagram
    • Demand–supply advection with per-class proportional-to-density allocation
  Keep Phase 1 (σ capture, μ release, P_block) identical to the kinetic model.

This isolates the multi-class transition dynamics (Bf ↔ Bs) from the kinetic
speed relaxation, so Phase 1 effects can be analysed directly.

IC: identical to the main benchmark (generate_dataset.py).
    Class A (trucks):      cells 74-79, ρ = 0.040  (Ω = 0.100)
    Class Bf (free cars):  cells 0-73,  ρ = 0.020
    Class Bs (trapped):    zero everywhere

Output:
    multiclass_trm_LWR.h5   — full simulation history
    figures/v_lwr.png       — 2×3 comparison with kinetic model
"""

import os
import time
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS (same as main benchmark)
# ─────────────────────────────────────────────────────────────────────────────
X, L, M = 150, 3, 3
dx, dt  = 20.0, 0.5
T_STEPS = 500                         # 250 s

rho_max   = 0.15
v_f_car   = 30.0                      # V — free-flow speed of non-stuck cars (Bf)
v_f_truck = 14.0                      # U — free-flow speed of trucks (A) and stuck (Bs)
w_back    = 4.62                      # W — backward wave speed (from R-H estimate)
rho_c     = rho_max * w_back / (v_f_car + w_back)   # ≈ 0.020 — critical density
Q_cap     = v_f_car * rho_c           # ≈ 0.60 PCE/m/s — capacity flow

# Phase 1 parameters (identical to kinetic model)
sigma_0    = 0.5
mu_0       = 0.3
R_A        = 0.05
eta_block  = 2.0
w_A_PCE    = 2.5                      # truck PCE weight

cfl_car   = v_f_car   * dt / dx       # 0.75  OK (< 1)
cfl_truck = v_f_truck * dt / dx       # 0.35  OK
assert cfl_car <= 1.0, f"CFL violated: {cfl_car:.4f}"
print(f"CFL_car = {cfl_car:.4f}, CFL_truck = {cfl_truck:.4f}")
print(f"rho_c  = {rho_c:.4f} PCE/m, Q_cap = {Q_cap:.4f} PCE/m/s")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def safe_phi(z):
    eps_phi = 1.0e-5
    taylor  = 1.0 - z / 2.0 + z ** 2 / 6.0
    exact   = -np.expm1(-z) / np.where(z > eps_phi, z, 1.0)
    return np.where(z > eps_phi, exact, taylor)


def compute_omega(rho):
    """PCE occupancy Ω = w_A·ρ_A + ρ_Bf + ρ_Bs."""
    return w_A_PCE * rho[0] + rho[1] + rho[2]


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION — identical IC to main benchmark (no speed classes needed)
# ─────────────────────────────────────────────────────────────────────────────
def initialize_state():
    rho = np.zeros((M, X, L), dtype=np.float64)
    rho[0, 74:80, :] = 0.040          # Trucks at bottleneck
    rho[1, 0:74,  :] = 0.020          # Bf upstream
    # rho[2] = 0 (Bs starts empty — must be created by Phase 1)
    return rho


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Capture & Release (identical physics to kinetic model)
# ─────────────────────────────────────────────────────────────────────────────
def phase1_capture_release(rho):
    omega = compute_omega(rho)                                    # (X, L)
    omega_norm = np.clip(omega, 0.0, rho_max) / rho_max
    P_block = omega_norm ** eta_block                             # (X, L)

    # Truck exposure (aggregate — no speed-class resolution in LWR)
    A_exposure = w_A_PCE * rho[0] / rho_max                       # (X, L)

    # Capture rate σ and release rate μ
    sigma = sigma_0 * P_block * A_exposure                        # (X, L)
    mu    = mu_0    * np.exp(-P_block * A_exposure / R_A)         # (X, L)

    # Exact 2×2 matrix exponential for [Bf, Bs] ODE
    S_total = sigma + mu
    F_total = rho[1] + rho[2]
    phi_z   = safe_phi(S_total * dt)

    rho_new = rho.copy()
    rho_new[1] = rho[1] * np.exp(-S_total * dt) + mu * F_total * dt * phi_z
    rho_new[2] = F_total - rho_new[1]
    rho_new[1] = np.maximum(rho_new[1], 0.0)
    rho_new[2] = np.maximum(rho_new[2], 0.0)

    return rho_new, sigma, mu, P_block


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — LWR advection with triangular FD, demand-supply, density sharing
# ─────────────────────────────────────────────────────────────────────────────
def phase3_LWR_advection(rho):
    # Per-class demand per cell
    D_A  = np.minimum(v_f_truck * rho[0], Q_cap)                  # trucks
    D_Bf = np.minimum(v_f_car   * rho[1], Q_cap)                  # free cars
    D_Bs = np.minimum(v_f_truck * rho[2], Q_cap)                  # stuck cars
    D_tot = D_A + D_Bf + D_Bs                                     # (X, L)

    # Per-cell supply (triangular FD downstream side)
    # Supervisor spec: S = min(Q_cap, W·(ρ_max − ρ_total)) — unweighted total density
    rho_tot = rho[0] + rho[1] + rho[2]                            # unweighted total density
    S_cell = np.minimum(Q_cap, w_back * np.maximum(0.0, rho_max - rho_tot))

    # Face fluxes with ring-road periodicity.
    # Face i separates cell (i-1) (upstream) and cell i (downstream).
    # Array phi[face, lane] for face = 0..X; ring: face X ≡ face 0.
    phi_A  = np.zeros((X + 1, L))
    phi_Bf = np.zeros((X + 1, L))
    phi_Bs = np.zeros((X + 1, L))

    # Vectorised face loop:
    # Upstream cell for internal face i (i=1..X-1) is (i-1); downstream is i.
    # For ring wrap, face 0's upstream is cell X-1.
    up_idx = np.arange(-1, X)    # shape (X+1,), up_idx[i] = i-1 mod X, but uses -1 → X-1
    up_idx = up_idx % X
    dn_idx = np.arange(0, X + 1) % X

    D_A_up  = D_A [up_idx, :]    # (X+1, L)
    D_Bf_up = D_Bf[up_idx, :]
    D_Bs_up = D_Bs[up_idx, :]
    D_tot_up = D_A_up + D_Bf_up + D_Bs_up
    S_dn = S_cell[dn_idx, :]

    # Upstream per-class densities (for density-proportional allocation)
    rho_A_up  = rho[0, up_idx, :]
    rho_Bf_up = rho[1, up_idx, :]
    rho_Bs_up = rho[2, up_idx, :]
    rho_tot_up = rho_A_up + rho_Bf_up + rho_Bs_up

    # Density-proportional capacity allocation (supervisor spec):
    #   If total demand ≤ supply → each class gets its demand (free flow)
    #   If total demand > supply → each class gets (ρ_m / ρ_total) × supply
    rho_safe = np.where(rho_tot_up > 1e-12, rho_tot_up, 1.0)
    share_A  = np.where(rho_tot_up > 1e-12, rho_A_up  / rho_safe, 0.0)
    share_Bf = np.where(rho_tot_up > 1e-12, rho_Bf_up / rho_safe, 0.0)
    share_Bs = np.where(rho_tot_up > 1e-12, rho_Bs_up / rho_safe, 0.0)

    # Per-class flux = min(own demand, density share × supply)
    phi_A  = np.minimum(D_A_up,  share_A  * S_dn)
    phi_Bf = np.minimum(D_Bf_up, share_Bf * S_dn)
    phi_Bs = np.minimum(D_Bs_up, share_Bs * S_dn)

    # Ring road: explicitly enforce face 0 = face X
    phi_A [0, :] = phi_A [X, :]
    phi_Bf[0, :] = phi_Bf[X, :]
    phi_Bs[0, :] = phi_Bs[X, :]

    # FVM update
    rho_new = rho.copy()
    rho_new[0] = rho[0] + (dt / dx) * (phi_A [:X, :] - phi_A [1:X + 1, :])
    rho_new[1] = rho[1] + (dt / dx) * (phi_Bf[:X, :] - phi_Bf[1:X + 1, :])
    rho_new[2] = rho[2] + (dt / dx) * (phi_Bs[:X, :] - phi_Bs[1:X + 1, :])
    rho_new = np.maximum(rho_new, 0.0)
    return rho_new


# ─────────────────────────────────────────────────────────────────────────────
# SPEED COMPUTATION (instantaneous triangular-FD)
# ─────────────────────────────────────────────────────────────────────────────
def compute_speeds(rho):
    """Return u[M, X, L]: per-class speed = min(v_f, W·(ρ_max − ρ_total)/ρ_total).

    Supervisor spec: all quantities use ρ_total (unweighted), not Ω (PCE-weighted).
    """
    rho_tot = rho[0] + rho[1] + rho[2]                            # unweighted total

    # Congestion speed (common to all when jammed): W·(ρ_max − ρ_tot) / ρ_tot
    v_jammed = np.where(rho_tot > 1e-10,
                        w_back * np.maximum(0.0, rho_max - rho_tot) / np.maximum(rho_tot, 1e-10),
                        v_f_car)
    u = np.zeros((M, X, L))
    u[0] = np.minimum(v_f_truck, v_jammed)                        # trucks
    u[1] = np.minimum(v_f_car,   v_jammed)                        # Bf
    u[2] = np.minimum(v_f_truck, v_jammed)                        # Bs
    return u


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run():
    rho = initialize_state()

    rho_hist = np.zeros((T_STEPS, M, X, L), dtype=np.float32)
    u_hist   = np.zeros((T_STEPS, M, X, L), dtype=np.float32)
    omega_hist = np.zeros((T_STEPS, X, L),  dtype=np.float32)

    t_wall = time.time()
    for t in range(T_STEPS):
        rho_hist[t]   = rho
        u_hist[t]     = compute_speeds(rho)
        omega_hist[t] = compute_omega(rho)

        rho, sigma, mu, P_block = phase1_capture_release(rho)
        rho = phase3_LWR_advection(rho)

        if t % 50 == 0 or t == T_STEPS - 1:
            omega_now = compute_omega(rho)
            bs_total  = float(rho[2].sum())
            print(f"  step {t:4d}/{T_STEPS}  |  Ω_peak={float(omega_now.max()):.4f}  "
                  f"|  Bs_total={bs_total:.4f}")

    print(f"  Done! Wall time: {time.time()-t_wall:.1f}s")
    return rho_hist, u_hist, omega_hist


# ─────────────────────────────────────────────────────────────────────────────
# SAVE HDF5
# ─────────────────────────────────────────────────────────────────────────────
def save_hdf5(rho_hist, u_hist, omega_hist, out_path):
    with h5py.File(out_path, 'w') as hf:
        pg = hf.create_group('parameters')
        for k, v in dict(X=X, L=L, M=M, dx_m=dx, dt_s=dt, T_steps=T_STEPS,
                         rho_max=rho_max, v_f_car=v_f_car, v_f_truck=v_f_truck,
                         w_back=w_back, rho_c=rho_c, Q_cap=Q_cap,
                         sigma_0=sigma_0, mu_0=mu_0, R_A=R_A,
                         eta_block=eta_block, w_A_PCE=w_A_PCE,
                         model='LWR Multiclass Variant (first-order)').items():
            pg.attrs[k] = v
        dg = hf.create_group('data')
        dg.create_dataset('rho',   data=rho_hist,   compression='gzip', compression_opts=4)
        dg.create_dataset('u',     data=u_hist,     compression='gzip', compression_opts=4)
        dg.create_dataset('omega', data=omega_hist, compression='gzip', compression_opts=4)
        dg.create_dataset('time_s', data=np.arange(T_STEPS) * dt)
    print(f"  Saved: {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON FIGURE — 2×3 Hovmöller panels (kinetic top, LWR bottom)
# ─────────────────────────────────────────────────────────────────────────────
def make_comparison_figure(lwr_rho, lwr_u, lwr_omega, kinetic_h5, out_png):
    with h5py.File(kinetic_h5, 'r') as hf:
        T_k   = int(hf['parameters'].attrs['T_steps'])
        dt_k  = float(hf['parameters'].attrs['dt_s'])
        # Kinetic stores f[T, M, N, X, L]; derive rho and u per class.
        rho_k_all = hf['data/rho_macro'][:]           # (T, M, X, L)
        u_k_all   = hf['data/u_macro'][:]             # (T, M, X, L)
        omega_k   = hf['data/omega'][:]               # (T, X, L)

    step_hov = max(1, T_STEPS // 100)

    def hov(arr, lane_avg=True):
        """(T, X, L) → (T_reduced, X) by sampling + lane averaging."""
        samp = arr[::step_hov]
        return samp.mean(axis=-1) if lane_avg else samp

    # Kinetic Hovmöllers (sub-sample same rate)
    Ω_k     = hov(omega_k)
    u_Bf_k  = hov(u_k_all[:, 1, :, :])
    rho_Bs_k = hov(rho_k_all[:, 2, :, :])

    # LWR Hovmöllers
    Ω_l     = hov(lwr_omega)
    u_Bf_l  = hov(lwr_u[:, 1, :, :])
    rho_Bs_l = hov(lwr_rho[:, 2, :, :])

    t_axis = np.arange(0, T_STEPS, step_hov) * dt

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Multiclass TRM: Kinetic (second-order) vs LWR (first-order) — same IC',
                 fontsize=14, fontweight='bold', y=0.995)

    row_labels = ['Kinetic\n(current model)', 'LWR variant\n(triangular FD)']

    def plot_hov(ax, data, cmap, vmin, vmax, title, cb_label):
        im = ax.pcolormesh(np.arange(X), t_axis, data[:len(t_axis)],
                           cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, label=cb_label)
        ax.axvline(74, color='white', ls='--', lw=1, alpha=0.8)
        ax.axvline(79, color='white', ls='--', lw=1, alpha=0.8)
        ax.set_xlabel('Cell x')
        ax.set_ylabel('Time [s]')
        ax.set_title(title)

    # Row 0: Kinetic
    plot_hov(axes[0, 0], Ω_k,      'turbo',   0.010, rho_max,
             f'{row_labels[0]}\n$\\Omega(x,t)$', r'$\Omega$ [PCE/m]')
    plot_hov(axes[0, 1], u_Bf_k,   'RdYlGn',  0,     v_f_car,
             f'{row_labels[0]}\n$u^{{(Bf)}}(x,t)$', 'm/s')
    plot_hov(axes[0, 2], rho_Bs_k, 'hot_r',   0,     0.03,
             f'{row_labels[0]}\n$\\rho^{{(Bs)}}(x,t)$ — trapped cars', 'veh/m')

    # Row 1: LWR
    plot_hov(axes[1, 0], Ω_l,      'turbo',   0.010, rho_max,
             f'{row_labels[1]}\n$\\Omega(x,t)$', r'$\Omega$ [PCE/m]')
    plot_hov(axes[1, 1], u_Bf_l,   'RdYlGn',  0,     v_f_car,
             f'{row_labels[1]}\n$u^{{(Bf)}}(x,t)$', 'm/s')
    plot_hov(axes[1, 2], rho_Bs_l, 'hot_r',   0,     0.03,
             f'{row_labels[1]}\n$\\rho^{{(Bs)}}(x,t)$ — trapped cars', 'veh/m')

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))
    out_h5  = os.path.join(BASE, 'multiclass_trm_LWR.h5')
    out_png = os.path.join(BASE, 'figures', 'v_lwr.png')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    kinetic_h5 = os.path.join(BASE, 'multiclass_trm_benchmark_500mb.h5')

    print('=' * 66)
    print('  LWR Multiclass Variant — first-order simulation')
    print('  (same IC as main benchmark, Phase 1 preserved, Phase 2/3 replaced)')
    print('=' * 66)

    rho_h, u_h, om_h = run()
    save_hdf5(rho_h, u_h, om_h, out_h5)

    print('-' * 66)
    print('  Generating comparison figure (Kinetic vs LWR)...')
    make_comparison_figure(rho_h, u_h, om_h, kinetic_h5, out_png)
    print('=' * 66)
