"""
Comparison: Version A (downstream Omega_{x+1}) vs Version B (local Omega_x)
for Eq.(4) acceleration supply filter.

Plots:
  Row 1: Hovmöller of truck mean speed  u^(A)(x,t)
  Row 2: Hovmöller of Bs density        rho^(Bs)(x,t)
  Row 3: Time series of f_Bs_total and Omega_peak
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

CODE_DIR = os.path.dirname(__file__)

# ── Load both datasets ────────────────────────────────────────────────────────
def load_h5(tag):
    path = os.path.join(CODE_DIR, f'multiclass_trm_{tag}.h5')
    hf = h5py.File(path, 'r')
    T = hf['parameters'].attrs['T_steps']
    dt = hf['parameters'].attrs['dt_s']
    X  = hf['parameters'].attrs['X']
    t_arr = np.arange(T) * dt                              # (T,) seconds

    rho_macro = hf['data/rho_macro'][:]                    # (T, M, X, L)
    u_macro   = hf['data/u_macro'][:]                      # (T, M, X, L)
    omega     = hf['data/omega'][:]                        # (T, X, L)

    # Average over lanes
    u_truck   = u_macro[:, 0, :, :].mean(axis=-1)         # (T, X)  truck mean speed
    rho_Bs    = rho_macro[:, 2, :, :].mean(axis=-1)        # (T, X)  Bs density
    omega_mean= omega.mean(axis=-1)                        # (T, X)

    # Scalars per timestep
    Bs_total  = rho_macro[:, 2, :, :].sum(axis=(1, 2))    # (T,)
    omega_peak= omega.max(axis=(1, 2))                     # (T,)

    hf.close()
    return dict(t=t_arr, X=X, u_truck=u_truck, rho_Bs=rho_Bs,
                omega_mean=omega_mean, Bs_total=Bs_total, omega_peak=omega_peak)

print("Loading vB …")
vB = load_h5('vB')
print("Loading vA …")
vA = load_h5('vA')

X   = vB['X']
t   = vB['t']
x   = np.arange(X)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 13))
fig.suptitle('Eq.(4) Acceleration + Eq.(13) Pure Demand $\\Psi$:\n'
             'Version B (local $\\Omega_x$)  vs  Version A (downstream $\\Omega_{x+1}$)',
             fontsize=12, fontweight='bold', y=0.99)

cols   = ['vB — local $\\Omega_x$', 'vA — downstream $\\Omega_{x+1}$']
dsets  = [vB, vA]
tags   = ['vB', 'vA']

# ── Row 0: Truck mean speed Hovmöller ─────────────────────────────────────────
for j, (ds, col) in enumerate(zip(dsets, cols)):
    ax = axes[0, j]
    im = ax.pcolormesh(x, t, ds['u_truck'], cmap='RdYlGn',
                       vmin=0, vmax=14, shading='auto')
    ax.axvline(74, color='white', ls='--', lw=1.0, alpha=0.8, label='Truck zone')
    ax.axvline(79, color='white', ls='--', lw=1.0, alpha=0.8)
    ax.set_xlabel('Cell index x')
    ax.set_ylabel('Time [s]')
    ax.set_title(f'Truck mean speed $u^{{(A)}}$ [m/s]\n{col}')
    plt.colorbar(im, ax=ax, label='m/s')
    ax.legend(fontsize=8, loc='upper right')

# ── Row 1: Bs density Hovmöller ───────────────────────────────────────────────
vmax_Bs = max(vB['rho_Bs'].max(), vA['rho_Bs'].max())
for j, (ds, col) in enumerate(zip(dsets, cols)):
    ax = axes[1, j]
    im = ax.pcolormesh(x, t, ds['rho_Bs'], cmap='hot_r',
                       vmin=0, vmax=vmax_Bs, shading='auto')
    ax.axvline(74, color='cyan', ls='--', lw=1.0, alpha=0.8, label='Truck zone')
    ax.axvline(79, color='cyan', ls='--', lw=1.0, alpha=0.8)
    ax.set_xlabel('Cell index x')
    ax.set_ylabel('Time [s]')
    ax.set_title(f'Trapped-car density $\\rho^{{(Bs)}}$ [veh/m]\n{col}')
    plt.colorbar(im, ax=ax, label='veh/m')
    ax.legend(fontsize=8, loc='upper right')

# ── Row 2: Time series ────────────────────────────────────────────────────────
ax_bs  = axes[2, 0]
ax_om  = axes[2, 1]

ax_bs.plot(t, vB['Bs_total'], color='tab:red',  lw=2.0, label='vB (local $\\Omega_x$)')
ax_bs.plot(t, vA['Bs_total'], color='tab:blue', lw=2.0, ls='--',
           label='vA (downstream $\\Omega_{x+1}$)')
ax_bs.set_xlabel('Time [s]')
ax_bs.set_ylabel('Total Bs mass [veh]')
ax_bs.set_title('Total trapped-car population $\\sum f^{(Bs)}$')
ax_bs.legend()
ax_bs.grid(alpha=0.3)

ax_om.plot(t, vB['omega_peak'], color='tab:red',  lw=2.0, label='vB')
ax_om.plot(t, vA['omega_peak'], color='tab:blue', lw=2.0, ls='--', label='vA')
ax_om.axhline(0.15, color='k', ls=':', lw=1.0, label='$\\rho_{\\max}=0.15$')
ax_om.set_xlabel('Time [s]')
ax_om.set_ylabel('Peak occupancy $\\Omega$ [PCE/m]')
ax_om.set_title('Peak occupancy across domain')
ax_om.legend()
ax_om.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(CODE_DIR, 'figures', 'compare_vA_vB.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")

# ── Text summary ──────────────────────────────────────────────────────────────
print("\n=== Quantitative Summary ===")
print(f"{'Metric':<30} {'vB (local)':>14} {'vA (downstream)':>16}")
print("-" * 62)
print(f"{'Peak Bs_total':<30} {vB['Bs_total'].max():>14.4f} {vA['Bs_total'].max():>16.4f}")
print(f"{'Mean Bs_total (t>50s)':<30} {vB['Bs_total'][vB['t']>50].mean():>14.4f} {vA['Bs_total'][vA['t']>50].mean():>16.4f}")
print(f"{'Peak Omega_peak':<30} {vB['omega_peak'].max():>14.4f} {vA['omega_peak'].max():>16.4f}")
print(f"{'Mean Omega_peak (t>50s)':<30} {vB['omega_peak'][vB['t']>50].mean():>14.4f} {vA['omega_peak'][vA['t']>50].mean():>16.4f}")
print(f"{'Bs_total at t=125s':<30} {vB['Bs_total'][vB['t']>=125][0]:>14.4f} {vA['Bs_total'][vA['t']>=125][0]:>16.4f}")
print(f"{'Bs_total at t=175s':<30} {vB['Bs_total'][vB['t']>=175][0]:>14.4f} {vA['Bs_total'][vA['t']>=175][0]:>16.4f}")
