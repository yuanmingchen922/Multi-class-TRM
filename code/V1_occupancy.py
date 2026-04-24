"""
V1 — 有效占用密度约束验证  (m3+m4 升级版)
对应 .tex §2 eq.1:  Ω_{x,l} = Σ_{m∈{A,Bf,Bs}} Σ_i w^(m) f_{i,x,l}^(m)
PCE 权重: w = [2.5, 1.0, 1.0]  (Bf 和 Bs 共享 w=1.0)

检验项:
  [V1-a] 物理上界: max(Ω) ≤ ρ_max (全时域，允许 2% 边界注入瞬态)
  [V1-b] 物理下界: min(Ω) ≥ 0
  [V1-c] 初始瓶颈: t=0, cells 74-79, Ω ≈ 0.100 (Class A: 2.5×0.040)
  [V1-d] PCE 加权一致性: 存储 omega 与从 f 重算误差 < 1e-10
  [V1-e] Bs 独立性: w^(Bf) = w^(Bs) = 1.0 确保 Bf↔Bs 互换不改变 Ω
"""

import os
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

# Font setup for Chinese text rendering in figures
if platform.system() == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS',
                                        'STHeiti', 'sans-serif']
elif platform.system() == 'Linux':
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC',
                                        'DejaVu Sans', 'sans-serif']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BASE   = os.path.dirname(os.path.abspath(__file__))
HDF5   = os.path.join(BASE, 'multiclass_trm_benchmark_500mb.h5')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

PASS = '\033[92m✓ PASS\033[0m'
FAIL = '\033[91m✗ FAIL\033[0m'


def run():
    results = {'module': 'V1_occupancy', 'checks': {}, 'figures': []}

    with h5py.File(HDF5, 'r') as hf:
        rho_max = float(hf['parameters'].attrs['rho_max'])
        M       = int(hf['parameters'].attrs['M'])
        X       = int(hf['parameters'].attrs['X'])
        L       = int(hf['parameters'].attrs['L'])
        T       = int(hf['parameters'].attrs['T_steps'])
        w       = hf['parameters/w_PCE'][:]          # (M,) = [2.5, 1.0, 1.0]
        dt      = float(hf['parameters'].attrs['dt_s'])

        print(f"\n{'='*60}")
        print(f"  V1 — 有效占用密度约束验证  (M={M} 类)")
        print(f"  w = {w},  ρ_max = {rho_max},  T = {T}")
        print(f"{'='*60}")

        omega_ds = hf['data/omega']   # (T, X, L)
        f_ds     = hf['data/f']       # (T, M, N, X, L)
        time_s   = hf['data/time_s'][:]

        # ── [V1-a] 物理上界 ─────────────────────────────────────────────────
        omega_max_per_t = np.zeros(T)
        for t in range(T):
            omega_max_per_t[t] = omega_ds[t].max()

        global_max = omega_max_per_t.max()
        TOLERANCE_A = rho_max * 0.02   # 允许 2% 上游注入瞬态超调
        violations_a = int((omega_max_per_t > rho_max + TOLERANCE_A).sum())
        passed_a = violations_a == 0
        results['checks']['V1-a'] = {
            'desc': 'max(Ω) ≤ ρ_max (全时域)',
            'passed': passed_a,
            'value': float(global_max),
            'threshold': rho_max,
            'violations': violations_a
        }
        tag = PASS if passed_a else FAIL
        print(f"  [V1-a] {tag}  max(Ω) = {global_max:.6f}  ≤  ρ_max={rho_max}"
              f"  (违规步数: {violations_a})")

        # ── [V1-b] 物理下界 ─────────────────────────────────────────────────
        omega_min_per_t = np.zeros(T)
        for t in range(T):
            omega_min_per_t[t] = omega_ds[t].min()
        global_min = omega_min_per_t.min()
        violations_b = int((omega_min_per_t < -1e-10).sum())
        passed_b = violations_b == 0
        results['checks']['V1-b'] = {
            'desc': 'min(Ω) ≥ 0', 'passed': passed_b,
            'value': float(global_min), 'violations': violations_b
        }
        tag = PASS if passed_b else FAIL
        print(f"  [V1-b] {tag}  min(Ω) = {global_min:.2e}  (违规步数: {violations_b})")

        # ── [V1-c] 初始瓶颈 ─────────────────────────────────────────────────
        # Class A (m=0, w=2.5) at cells 74-79, i=0, density=0.040
        # Expected Ω = 2.5 × 0.040 = 0.100
        omega_t0 = omega_ds[0]                    # (X, L)
        bott_omega = omega_t0[74:80, :].mean()
        expected   = w[0] * 0.040                 # = 0.100
        err_c = abs(bott_omega - expected)
        passed_c = err_c < 0.005
        results['checks']['V1-c'] = {
            'desc': 't=0 瓶颈 Ω ≈ 0.100 (Class A: 2.5×0.040)',
            'passed': passed_c, 'value': float(bott_omega),
            'expected': float(expected), 'abs_error': float(err_c)
        }
        tag = PASS if passed_c else FAIL
        print(f"  [V1-c] {tag}  瓶颈 Ω(t=0,x=74-79) = {bott_omega:.4f}"
              f"  ≈ 期望 {expected:.4f}  (误差: {err_c:.2e})")

        # ── [V1-d] PCE 加权一致性 (eq.1) ────────────────────────────────────
        max_err_d = 0.0
        for t in range(0, T, 50):
            f_t   = f_ds[t]                                   # (M, N, X, L)
            om_t  = omega_ds[t]                               # (X, L)
            om_re = (w[:, None, None, None] * f_t).sum(axis=(0, 1))
            max_err_d = max(max_err_d, float(np.abs(om_re - om_t).max()))
        passed_d = max_err_d < 1e-10
        results['checks']['V1-d'] = {
            'desc': 'omega 与 eq.1 重算一致 < 1e-10',
            'passed': passed_d, 'max_error': float(max_err_d)
        }
        tag = PASS if passed_d else FAIL
        print(f"  [V1-d] {tag}  Ω 一致性最大误差 = {max_err_d:.2e}")

        # ── [V1-e] Bf↔Bs 权重相等性 ─────────────────────────────────────────
        # w^(Bf) = w^(Bs): 若将所有 Bf 换成 Bs (反之亦然)，Ω 不变
        passed_e = bool(abs(w[1] - w[2]) < 1e-12)
        results['checks']['V1-e'] = {
            'desc': 'w^(Bf) = w^(Bs) = 1.0  (Bf↔Bs 互换不改变 Ω)',
            'passed': passed_e, 'w_Bf': float(w[1]), 'w_Bs': float(w[2])
        }
        tag = PASS if passed_e else FAIL
        print(f"  [V1-e] {tag}  w^(Bf)={w[1]:.1f} = w^(Bs)={w[2]:.1f}  "
              f"({'相等' if passed_e else '不等'})")

        # ═══════════════════════════════ 图表 ═══════════════════════════════
        fig, axes = plt.subplots(1, 4, figsize=(26, 5))

        # 图1: max(Ω) 时序
        ax = axes[0]
        ax.plot(time_s, omega_max_per_t, 'C0-', lw=1.5, label=r'$\max_{x,l}\Omega(t)$')
        ax.axhline(rho_max, color='red', ls='--', lw=1.5, label=r'$\rho_{\max}=0.15$')
        ax.set_xlabel('Time [s]'); ax.set_ylabel(r'$\Omega$ [PCE/m]')
        ax.set_title('[V1-a] Global Max Occupancy Density (M=3 classes)')
        ax.legend(); ax.grid(alpha=0.3)
        ax.annotate('Initial Class A Truck\nBottleneck, \u03a9\u22480.100',
                    xy=(0, omega_max_per_t[0]), xytext=(20, 0.075),
                    arrowprops=dict(arrowstyle='->', color='orange'),
                    fontsize=9, color='orange')

        # 图2: t=0 空间占用热图 + 各类贡献
        ax2 = axes[1]
        # Decompose Ω into A / Bf / Bs contributions at t=0
        f0 = f_ds[0]   # (M, N, X, L)
        omega_A  = (w[0] * f0[0]).sum(axis=0)[:, 1]   # lane 1 (middle lane)
        omega_Bf = (w[1] * f0[1]).sum(axis=0)[:, 1]
        omega_Bs = (w[2] * f0[2]).sum(axis=0)[:, 1]
        x_cells  = np.arange(X)

        ax2.fill_between(x_cells, 0, omega_A,  alpha=0.7, color='C1', label='Class A (Trucks)')
        ax2.fill_between(x_cells, omega_A, omega_A+omega_Bf, alpha=0.7, color='C0', label='Class Bf (Free Cars)')
        ax2.fill_between(x_cells, omega_A+omega_Bf, omega_A+omega_Bf+omega_Bs, alpha=0.5, color='C2', label='Class Bs (Trapped)')
        ax2.axhline(rho_max, color='red', ls='--', lw=1.5, label=r'$\rho_{\max}$')
        ax2.axvspan(74, 79, alpha=0.1, color='red')
        ax2.axvspan(0, 73, alpha=0.04, color='blue')
        ax2.set_xlabel('Cell x'); ax2.set_ylabel(r'$\Omega$ [PCE/m]')
        ax2.set_title('[V1-c] t=0 Per-class Occupancy Contribution (Middle Lane)')
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        # Helper: draw a per-class stacked snapshot at step t_snap
        def _draw_snapshot(ax, t_snap, title):
            f_s  = f_ds[t_snap]
            om_A  = (w[0] * f_s[0]).sum(axis=0)[:, 1]
            om_Bf = (w[1] * f_s[1]).sum(axis=0)[:, 1]
            om_Bs = (w[2] * f_s[2]).sum(axis=0)[:, 1]
            ax.fill_between(x_cells, 0, om_A, alpha=0.7, color='C1',
                            label='Class A (Trucks)')
            ax.fill_between(x_cells, om_A, om_A + om_Bf, alpha=0.7, color='C0',
                            label='Class Bf (Free Cars)')
            ax.fill_between(x_cells, om_A + om_Bf,
                            om_A + om_Bf + om_Bs, alpha=0.7, color='C2',
                            label='Class Bs (Trapped)')
            ax.axhline(rho_max, color='red', ls='--', lw=1.5, label=r'$\rho_{\max}$')
            ax.axvspan(74, 79, alpha=0.1, color='red')
            ax.set_xlabel('Cell x'); ax.set_ylabel(r'$\Omega$ [PCE/m]')
            ax.set_title(title)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # 图3: t=100s snapshot — 瓶颈拥堵积累期
        t_mid = min(200, T - 1)   # 100 s
        _draw_snapshot(axes[2], t_mid,
                       f'[V1] t={t_mid*0.5:.0f}s Snapshot — Bottleneck congestion peak')

        # 图4: t=250s snapshot — 环路均化后期
        t_end = T - 1             # 250 s
        _draw_snapshot(axes[3], t_end,
                       f'[V1] t={t_end*0.5:.0f}s Snapshot — Ring road late-time equilibration')

        plt.tight_layout()
        fig_path = os.path.join(FIGDIR, 'V1_occupancy.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        results['figures'].append(fig_path)
        print(f"\n  图表: {fig_path}")

    passed_all = all(v['passed'] for v in results['checks'].values())
    n_p = sum(1 for v in results['checks'].values() if v['passed'])
    results['summary'] = f"{'PASSED' if passed_all else 'FAILED'} {n_p}/{len(results['checks'])}"
    print(f"  V1 汇总: {results['summary']}\n")
    return results


if __name__ == '__main__':
    run()
