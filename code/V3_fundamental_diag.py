"""
V3_fundamental_diag — Complete Fundamental Diagram in PCE-aggregate space.

Panels:
  Left : PCE-aggregate FD  (Ω vs Q_PCE), coloured by road region
  Right: Per-class FD       (ρ^(m) vs q^(m) for A, Bf, Bs)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

BASE   = os.path.dirname(os.path.abspath(__file__))
HDF5   = os.path.join(BASE, 'multiclass_trm_benchmark_500mb.h5')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

with h5py.File(HDF5, 'r') as hf:
    rho_max = float(hf['parameters'].attrs['rho_max'])
    v_max   = float(hf['parameters'].attrs['v_max_mps'])
    T       = int(hf['parameters'].attrs['T_steps'])
    X       = int(hf['parameters'].attrs['X'])
    L       = int(hf['parameters'].attrs['L'])
    M       = int(hf['parameters'].attrs['M'])
    w       = hf['parameters/w_PCE'][:]          # (M,)
    dt      = float(hf['parameters'].attrs['dt_s'])

    step_s = max(1, T // 60)   # ~60 timestep samples

    omega_list, Qpce_list, region_list = [], [], []
    rho_cls = [[] for _ in range(M)]
    q_cls   = [[] for _ in range(M)]

    for t_i in range(0, T, step_s):
        omega_t = hf['data/omega'][t_i]        # (X, L)
        q_t     = hf['data/q_macro'][t_i]      # (M, X, L)
        rho_t   = hf['data/rho_macro'][t_i]    # (M, X, L)

        Q_pce_t = (w[:, None, None] * q_t).sum(axis=0)   # (X, L)

        for x in range(X):
            for l in range(L):
                om = float(omega_t[x, l])
                qp = float(Q_pce_t[x, l])
                if om < 1e-7 and qp < 1e-7:
                    continue           # skip truly empty cells
                omega_list.append(om)
                Qpce_list.append(qp)
                if x < 74:
                    region_list.append(0)    # upstream
                elif x <= 79:
                    region_list.append(1)    # bottleneck
                else:
                    region_list.append(2)    # downstream

                for m in range(M):
                    rho_cls[m].append(float(rho_t[m, x, l]))
                    q_cls[m].append(float(q_t[m, x, l]))

omega_arr  = np.array(omega_list)
Qpce_arr   = np.array(Qpce_list)
region_arr = np.array(region_list)
rho_c = [np.array(rho_cls[m]) for m in range(M)]
q_c   = [np.array(q_cls[m])   for m in range(M)]

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Fundamental Diagram — Multiclass TRM\n'
             r'Left: PCE-aggregate $(\Omega, Q_\mathrm{PCE})$  |  '
             r'Right: per-class $(\rho^{(m)}, q^{(m)})$',
             fontsize=12, fontweight='bold')

# ── Left: PCE aggregate FD ───────────────────────────────────────────────────
ax = axes[0]
region_cfg = [
    (0, 'C0', 'Upstream  (x < 74)'),
    (1, 'C3', 'Bottleneck (x = 74–79)'),
    (2, 'C2', 'Downstream (x > 79)'),
]
for r_idx, col, lbl in region_cfg:
    mask = region_arr == r_idx
    ax.scatter(omega_arr[mask], Qpce_arr[mask],
               s=3, alpha=0.30, color=col, label=lbl)

# Free-flow reference slope  Q = v_max · Ω
rho_line = np.linspace(0, rho_max, 300)
ax.plot(rho_line, v_max * rho_line, 'k--', lw=1.5, alpha=0.7,
        label=f'$Q = v_{{\\max}}\\cdot\\Omega$  ({v_max:.0f} m/s)')
ax.axvline(rho_max, color='grey', ls=':', lw=1.2, alpha=0.8,
           label=r'$\rho_{\max} = $' + f'{rho_max}')

ax.set_xlabel(r'PCE Occupancy $\Omega$ [PCE/m]', fontsize=11)
ax.set_ylabel(r'PCE Flow $Q_\mathrm{PCE}$ [PCE/m/s]', fontsize=11)
ax.set_title('PCE-Aggregate Fundamental Diagram\n(free-flow branch + congested branch)',
             fontsize=10)
ax.legend(fontsize=8, markerscale=4)
ax.grid(alpha=0.3)

# ── Right: per-class FD ───────────────────────────────────────────────────────
ax = axes[1]
cls_cfg = [
    (0, 'C1', 'A — Trucks (PCE = 2.5)'),
    (1, 'C0', 'Bf — Free cars (PCE = 1)'),
    (2, 'C2', 'Bs — Trapped cars (PCE = 1)'),
]
for m, col, lbl in cls_cfg:
    mask = rho_c[m] > 1e-6
    ax.scatter(rho_c[m][mask], q_c[m][mask],
               s=3, alpha=0.30, color=col, label=lbl)

ax.set_xlabel(r'Class Density $\rho^{(m)}$ [veh/m]', fontsize=11)
ax.set_ylabel(r'Class Flow $q^{(m)}$ [veh/m/s]', fontsize=11)
ax.set_title('Per-class Fundamental Diagram\n(vehicle units per class)',
             fontsize=10)
ax.legend(fontsize=8, markerscale=4)
ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(FIGDIR, 'V3_fundamental_diag.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")
