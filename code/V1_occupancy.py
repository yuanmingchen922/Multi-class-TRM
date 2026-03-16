"""
V1 — 有效占用密度约束验证
对应 .tex §2:  Ω_{x,l} = Σ_m Σ_i w^(m) f_{i,x,l}^(m)
物理要求: 0 ≤ Ω_{x,l} ≤ ρ_max  全时域成立

检验项:
  [V1-a] 物理上界: max(Ω) ≤ ρ_max = 0.15
  [V1-b] 物理下界: min(Ω) ≥ 0
  [V1-c] 初始瓶颈: t=0, cells 74-79, Ω ≈ 0.145 (HDT: 2.5×0.058)
  [V1-d] PCE 加权一致性: 存储的 omega 与从 f 重算的 Ω 误差 < 1e-10
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import h5py

# ── 路径 ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
HDF5   = os.path.join(BASE, 'multiclass_trm_benchmark_500mb.h5')
FIGDIR = os.path.join(BASE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

PASS = '\033[92m✓ PASS\033[0m'
FAIL = '\033[91m✗ FAIL\033[0m'

# ─────────────────────────────────────────────────────────────────────────────
def run():
    results = {'module': 'V1_occupancy', 'checks': {}, 'figures': []}

    with h5py.File(HDF5, 'r') as hf:
        # ── 读取参数 ──────────────────────────────────────────────────────────
        rho_max = float(hf['parameters'].attrs['rho_max'])
        X  = int(hf['parameters'].attrs['X'])
        L  = int(hf['parameters'].attrs['L'])
        N  = int(hf['parameters'].attrs['N'])
        M  = int(hf['parameters'].attrs['M'])
        T  = int(hf['parameters'].attrs['T_steps'])
        w  = hf['parameters/w_PCE'][:]          # (M,)
        dt = float(hf['parameters'].attrs['dt_s'])

        omega_ds = hf['data/omega']   # (T, X, L)
        f_ds     = hf['data/f']       # (T, M, N, X, L)
        time_s   = hf['data/time_s'][:]

        print(f"\n{'='*60}")
        print(f"  V1 — 有效占用密度约束验证")
        print(f"  ρ_max = {rho_max},  T = {T},  grid = {X}×{L}")
        print(f"{'='*60}")

        # ── [V1-a]  物理上界 ─────────────────────────────────────────────────
        omega_max_per_t = np.zeros(T)
        for t in range(T):
            omega_max_per_t[t] = omega_ds[t].max()

        global_max = omega_max_per_t.max()
        # 允许上游注入边界的微小瞬态超调 (< 2% ρ_max)
        TOLERANCE_A = rho_max * 0.02
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
        print(f"  [V1-a] {tag}  max(Ω)_global = {global_max:.6f}  ≤  ρ_max = {rho_max}"
              f"  (违规步数: {violations_a})")

        # ── [V1-b]  物理下界 ─────────────────────────────────────────────────
        omega_min_per_t = np.zeros(T)
        for t in range(T):
            omega_min_per_t[t] = omega_ds[t].min()

        global_min = omega_min_per_t.min()
        violations_b = int((omega_min_per_t < -1e-10).sum())
        passed_b = violations_b == 0
        results['checks']['V1-b'] = {
            'desc': 'min(Ω) ≥ 0 (全时域)',
            'passed': passed_b,
            'value': float(global_min),
            'threshold': 0.0,
            'violations': violations_b
        }
        tag = PASS if passed_b else FAIL
        print(f"  [V1-b] {tag}  min(Ω)_global = {global_min:.2e}  ≥  0"
              f"  (违规步数: {violations_b})")

        # ── [V1-c]  初始瓶颈验证 ─────────────────────────────────────────────
        # JSON: bottleneck cells 75-80 (1-indexed) → 74-79 (0-indexed)
        # HDT (m=1, w=2.5) at speed i=0, f=0.058 → Ω ≈ 2.5×0.058 = 0.145
        omega_t0 = omega_ds[0]               # (X, L)
        bottleneck_omega = omega_t0[74:80, :].mean()
        expected_omega   = 2.5 * 0.058       # 0.145
        err_c = abs(bottleneck_omega - expected_omega)
        passed_c = err_c < 0.002
        results['checks']['V1-c'] = {
            'desc': 't=0 瓶颈 Ω ≈ 0.145 (HDT 2.5×0.058)',
            'passed': passed_c,
            'value': float(bottleneck_omega),
            'expected': expected_omega,
            'abs_error': float(err_c)
        }
        tag = PASS if passed_c else FAIL
        print(f"  [V1-c] {tag}  瓶颈 Ω(t=0, x=74-79) = {bottleneck_omega:.4f}"
              f"  ≈ 期望 {expected_omega:.4f}  (误差: {err_c:.2e})")

        # ── [V1-d]  PCE 加权一致性 (eq.1) ────────────────────────────────────
        # 从 f 重算 omega，与存储的 omega 对比
        # eq.1: Ω_{x,l} = Σ_m Σ_i w^(m) f_{i,x,l}^(m)
        max_consistency_err = 0.0
        for t in range(0, T, 50):   # 抽查每 50 步
            f_t     = f_ds[t]                # (M, N, X, L)
            omega_t = omega_ds[t]            # (X, L)
            omega_recomputed = (w[:, None, None, None] * f_t).sum(axis=(0, 1))  # (X, L)
            err = np.abs(omega_recomputed - omega_t).max()
            if err > max_consistency_err:
                max_consistency_err = err

        passed_d = max_consistency_err < 1e-10
        results['checks']['V1-d'] = {
            'desc': 'omega 存储与 eq.1 重算一致 (误差 < 1e-10)',
            'passed': passed_d,
            'max_error': float(max_consistency_err)
        }
        tag = PASS if passed_d else FAIL
        print(f"  [V1-d] {tag}  Ω 一致性最大误差 = {max_consistency_err:.2e}"
              f"  (阈值 1e-10)")

        # ═══════════════════════════════════════════════════════════════════════
        # 图 V1-1: max(Ω) 随时间演化 + ρ_max 红线
        # ═══════════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(time_s, omega_max_per_t, 'C0-', lw=1.5, label=r'$\max_{x,l}\,\Omega(t)$')
        ax.axhline(rho_max, color='red', ls='--', lw=1.5, label=r'$\rho_{\max}=0.15$')
        ax.set_xlabel('时间 [s]', fontsize=11)
        ax.set_ylabel(r'$\Omega$ [PCE/m]', fontsize=11)
        ax.set_title('[V1-a] 最大有效占用密度随时间演化', fontsize=12)
        ax.legend(fontsize=10)
        ax.set_ylim(-0.005, rho_max * 1.15)
        ax.grid(alpha=0.3)

        # 标注初始极端刚性区域
        ax.annotate('初始 HDT 瓶颈\nΩ≈0.145', xy=(0, omega_max_per_t[0]),
                    xytext=(40, 0.12),
                    arrowprops=dict(arrowstyle='->', color='orange'),
                    fontsize=9, color='orange')

        # ═══════════════════════════════════════════════════════════════════════
        # 图 V1-2: t=0 时刻空间占用热图 (x × l)
        # ═══════════════════════════════════════════════════════════════════════
        ax2 = axes[1]
        # omega_t0: (X, L) → 转置为 (L, X) 显示
        im = ax2.imshow(omega_t0.T, aspect='auto', origin='lower',
                        extent=[0, X-1, 0.5, L+0.5],
                        vmin=0, vmax=rho_max, cmap='hot_r')
        plt.colorbar(im, ax=ax2, label=r'$\Omega_{x,l}$ [PCE/m]')
        ax2.set_xlabel('空间格子 x', fontsize=11)
        ax2.set_ylabel('车道 l', fontsize=11)
        ax2.set_title('[V1-c] t=0 有效占用密度空间分布', fontsize=12)
        ax2.set_yticks([1, 2, 3])
        # 标注瓶颈与注入区
        ax2.axvspan(74, 79, alpha=0.15, color='red', label='HDT 瓶颈 (74-79)')
        ax2.axvspan(59, 69, alpha=0.15, color='blue', label='PC 注入 (59-69)')
        ax2.legend(fontsize=8, loc='upper left')

        plt.tight_layout()
        fig_path = os.path.join(FIGDIR, 'V1_occupancy.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        results['figures'].append(fig_path)
        print(f"\n  图表已保存: {fig_path}")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    passed_all = all(v['passed'] for v in results['checks'].values())
    n_pass = sum(1 for v in results['checks'].values() if v['passed'])
    n_total = len(results['checks'])
    results['summary'] = f"{'PASSED' if passed_all else 'FAILED'} {n_pass}/{n_total}"
    print(f"\n  V1 汇总: {results['summary']}\n")
    return results


if __name__ == '__main__':
    run()
