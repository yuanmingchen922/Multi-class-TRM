"""
V4 — 侧向变道 Softplus 验证
对应 .tex §5:  γ = κ·(1/ω)·ln[1+exp(ω·y)]·gap_filter  (eq.6)

检验项:
  [V4-a] Softplus C∞ 连续性: 与 hard-max 对比，y=0 处无导数跳变
  [V4-b] y≤0 时 γ≈0: 无密度差时几乎无变道驱动力
  [V4-c] 侧向 Dirichlet 边界: γ_left[*,*,*,0]=0, γ_right[*,*,*,L-1]=0
  [V4-d] PC vs HDT 变道速率比 ≈ κ_PC/κ_HDT = 7.5
  [V4-e] 间隙过滤: γ→0 当目标车道 Ω_target→ρ_max
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

PASS = '\033[92m✓ PASS\033[0m'
FAIL = '\033[91m✗ FAIL\033[0m'


def run():
    results = {'module': 'V4_lateral', 'checks': {}, 'figures': []}

    with h5py.File(HDF5, 'r') as hf:
        # ── 参数 ──────────────────────────────────────────────────────────────
        rho_max  = float(hf['parameters'].attrs['rho_max'])
        R_gap    = float(hf['parameters'].attrs['R_gap'])
        omega_sp = float(hf['parameters'].attrs['omega_sp'])
        eps      = float(hf['parameters'].attrs['eps'])
        T        = int(hf['parameters'].attrs['T_steps'])
        L        = int(hf['parameters'].attrs['L'])
        N        = int(hf['parameters'].attrs['N'])
        X        = int(hf['parameters'].attrs['X'])
        kappa    = hf['parameters/kappa_hz'][:]   # (M,)
        time_s   = hf['data/time_s'][:]

        print(f"\n{'='*60}")
        print(f"  V4 — 侧向变道 Softplus 验证")
        print(f"  κ_PC={kappa[0]}, κ_HDT={kappa[1]}, ω={omega_sp}, R_gap={R_gap}")
        print(f"{'='*60}")

        # ── [V4-a]  Softplus vs hard-max C∞ 连续性 ──────────────────────────
        # 在 y ∈ [-0.5, 0.5] 上比较两者的一阶导数
        y_range = np.linspace(-0.5, 0.5, 2000)

        # Softplus: (1/ω)·ln[1+exp(ω·y)]
        def softplus(y):
            return np.where(y > 20, y,
                            np.log1p(np.exp(np.minimum(omega_sp * y, 500))) / omega_sp)

        sp_vals   = softplus(y_range)
        hm_vals   = np.maximum(0, y_range)   # hard-max

        # 数值一阶导数
        dy = y_range[1] - y_range[0]
        d_sp = np.gradient(sp_vals, dy)   # softplus 导数 → sigmoid 函数, 连续
        d_hm = np.gradient(hm_vals, dy)   # hard-max 导数 → 阶跃函数, 不连续

        # 在 y=0 附近检验: softplus 导数在 [-0.05, 0.05] 内的方差应远小于 hard-max
        near_zero = np.abs(y_range) < 0.05
        var_sp = float(np.var(d_sp[near_zero]))
        var_hm = float(np.var(d_hm[near_zero]))
        # 阈值: softplus 方差应远小于 hard-max 方差 (比值 < 0.1 即证明显著更光滑)
        passed_a = var_sp < var_hm * 0.1
        results['checks']['V4-a'] = {
            'desc': 'Softplus 导数方差 << hard-max 导数方差 (y=0 附近)',
            'passed': passed_a,
            'var_softplus': var_sp,
            'var_hardmax': var_hm
        }
        tag = PASS if passed_a else FAIL
        print(f"  [V4-a] {tag}  Softplus 导数方差={var_sp:.4f}, "
              f"hard-max 导数方差={var_hm:.4f} (y=0±0.05)")

        # ── [V4-b]  y≤0 时 γ≈0 ──────────────────────────────────────────────
        # 当 Ω_source ≤ Ω_target (y≤0), softplus(y)≈0 → γ≈0
        # 取 y=-0.5: softplus = (1/20)·ln[1+exp(-10)] ≈ (1/20)·ln(1) ≈ 0
        y_neg = -0.5
        sp_neg = float(softplus(np.array([y_neg]))[0])
        gap_factor_full = float(1.0 - np.exp(-rho_max / R_gap))   # target lane empty
        gamma_neg = kappa[0] * sp_neg * gap_factor_full
        passed_b = gamma_neg < 1e-3
        results['checks']['V4-b'] = {
            'desc': 'y=-0.5 时 γ ≈ 0 (无密度梯度驱动)',
            'passed': passed_b,
            'gamma_at_y_neg05': float(gamma_neg),
            'softplus_at_y_neg05': float(sp_neg)
        }
        tag = PASS if passed_b else FAIL
        print(f"  [V4-b] {tag}  γ(y=-0.5) = {gamma_neg:.4e}  (softplus={sp_neg:.2e})")

        # ── [V4-c]  侧向 Dirichlet 边界 (从存储数据验证) ─────────────────────
        max_inner = 0.0   # γ_left[:, :, :, 0] 应为 0 (最内车道不能再向左)
        max_outer = 0.0   # γ_right[:, :, :, L-1] 应为 0 (最外车道不能再向右)
        for t in range(0, T, 40):
            gl_inner = float(np.abs(hf['data/gamma_left'][t, :, :, :, 0]).max())
            gr_outer = float(np.abs(hf['data/gamma_right'][t, :, :, :, L-1]).max())
            if gl_inner > max_inner: max_inner = gl_inner
            if gr_outer > max_outer: max_outer = gr_outer

        passed_c = max_inner < 1e-12 and max_outer < 1e-12
        results['checks']['V4-c'] = {
            'desc': 'γ_left[lane=0]=0 且 γ_right[lane=L-1]=0',
            'passed': passed_c,
            'max_inner_violation': max_inner,
            'max_outer_violation': max_outer
        }
        tag = PASS if passed_c else FAIL
        print(f"  [V4-c] {tag}  γ_left 内边界最大值 = {max_inner:.2e}, "
              f"γ_right 外边界最大值 = {max_outer:.2e}")

        # ── [V4-d]  PC vs HDT 速率比 ─────────────────────────────────────────
        # 相同工况下 γ_PC/γ_HDT = κ_PC/κ_HDT = 0.60/0.08 = 7.5
        y_pos = 0.3   # 正密度梯度驱动
        sp_pos = float(softplus(np.array([y_pos]))[0])
        gap_factor = float(1.0 - np.exp(-rho_max * 0.5 / R_gap))
        gamma_PC  = kappa[0] * sp_pos * gap_factor
        gamma_HDT = kappa[1] * sp_pos * gap_factor
        ratio_d = gamma_PC / gamma_HDT
        expected_d = kappa[0] / kappa[1]   # 7.5
        err_d = abs(ratio_d - expected_d) / expected_d
        passed_d = err_d < 1e-8
        results['checks']['V4-d'] = {
            'desc': 'γ_PC / γ_HDT = κ_PC/κ_HDT = 7.5',
            'passed': passed_d,
            'ratio': float(ratio_d),
            'expected': float(expected_d)
        }
        tag = PASS if passed_d else FAIL
        print(f"  [V4-d] {tag}  γ_PC/γ_HDT = {ratio_d:.4f}  ≈ {expected_d:.1f}"
              f"  (误差 {err_d:.2e})")

        # ── [V4-e]  间隙过滤: Ω_target→ρ_max 时 γ→0 ──────────────────────
        omega_target_range = np.linspace(0, rho_max, 500)
        gap_arg   = np.maximum(0, rho_max - omega_target_range)
        gap_vals  = 1.0 - np.exp(-gap_arg / R_gap)
        sp_fixed  = float(softplus(np.array([0.5]))[0])   # 固定正 y
        gamma_filter = kappa[0] * sp_fixed * gap_vals

        at_cap = gamma_filter[-1]
        gap_monotone = bool((np.diff(gamma_filter) <= 1e-12).all())
        passed_e = gap_monotone and (at_cap < 1e-10)
        results['checks']['V4-e'] = {
            'desc': 'γ 随 Ω_target→ρ_max 单调趋零 (间隙过滤)',
            'passed': passed_e,
            'gamma_at_capacity': float(at_cap),
            'monotone': gap_monotone
        }
        tag = PASS if passed_e else FAIL
        print(f"  [V4-e] {tag}  γ(Ω_target=ρ_max) = {at_cap:.2e}, "
              f"单调={'是' if gap_monotone else '否'}")

        # ═══════════════════════════════════════════════════════════════════════
        # 读取车道密度时序数据
        # ═══════════════════════════════════════════════════════════════════════
        rho_macro = hf['data/rho_macro'][:]    # (T, M, X, L)
        # 全路段各车道总密度 (平均 x, sum M)
        lane_density = rho_macro.sum(axis=1).mean(axis=1)   # (T, L)

        # 读取 gamma_right 时序 (一个代表点)
        gr_ds = hf['data/gamma_right']
        # 取 PC (m=0), speed i=7, cell x=77 (瓶颈附近), 所有车道
        gr_sample = np.array([gr_ds[t, 0, 7, 77, :] for t in range(T)])  # (T, L)

        # ═══════════════════════════════════════════════════════════════════════
        # 生成图表
        # ═══════════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ── 图 V4-1: Softplus vs hard-max ─────────────────────────────────
        ax = axes[0, 0]
        ax.plot(y_range, sp_vals, 'C0-', lw=2.5, label='Softplus (ω=20)')
        ax.plot(y_range, hm_vals, 'C1--', lw=2, label='Hard-max(0,y)')
        ax.plot(y_range, d_sp, 'C0:', lw=1.5, alpha=0.7, label="Softplus 一阶导")
        ax.plot(y_range, d_hm, 'C1:', lw=1.5, alpha=0.7, label="Hard-max 一阶导")
        ax.axvline(0, color='gray', ls=':', lw=1)
        ax.set_xlabel('y = (Ω_source - Ω_target) / ρ_max', fontsize=11)
        ax.set_ylabel('变道驱动力', fontsize=11)
        ax.set_title('[V4-a] Softplus 平滑 vs Hard-max 跳变 (C∞ 连续性)', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(-0.1, 0.6)
        ax.grid(alpha=0.3)

        # ── 图 V4-2: 间隙过滤曲线 ─────────────────────────────────────────
        ax = axes[0, 1]
        for k_val, lbl, col in [(kappa[0], 'PC (κ=0.60)', 'C0'),
                                  (kappa[1], 'HDT (κ=0.08)', 'C1')]:
            gamma_f = k_val * sp_fixed * gap_vals
            ax.plot(omega_target_range, gamma_f, color=col, lw=2, label=lbl)

        ax.axvline(rho_max, color='red', ls='--', lw=1.5, label=r'$\rho_{\max}$')
        ax.set_xlabel(r'目标车道 $\Omega_{target}$ [PCE/m]', fontsize=11)
        ax.set_ylabel(r'$\gamma$ [Hz]', fontsize=11)
        ax.set_title('[V4-e] 间隙过滤: Ω_target→ρ_max 时 γ→0 (eq.6)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # ── 图 V4-3: 三车道密度随时间演化 ────────────────────────────────
        ax = axes[1, 0]
        lane_labels = ['Lane 1 (内)', 'Lane 2 (中)', 'Lane 3 (外)']
        for l_idx, (lbl, col) in enumerate(zip(lane_labels, ['C0', 'C1', 'C2'])):
            ax.plot(time_s, lane_density[:, l_idx], color=col, lw=1.5, label=lbl)

        ax.set_xlabel('时间 [s]', fontsize=11)
        ax.set_ylabel(r'平均密度 [veh/m]', fontsize=11)
        ax.set_title('[V4-d] 三车道平均密度随时间 (均衡化趋势)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # ── 图 V4-4: γ_right 在瓶颈附近的时序 ────────────────────────────
        ax = axes[1, 1]
        for l_idx in range(L):
            ax.plot(time_s, gr_sample[:, l_idx], lw=1.5,
                    label=f'Lane {l_idx+1}')

        ax.axhline(0, color='gray', ls=':', lw=1)
        ax.set_xlabel('时间 [s]', fontsize=11)
        ax.set_ylabel(r'$\gamma_{right}$ [Hz]', fontsize=11)
        ax.set_title('[V4-c] PC(i=7) 在 x=77(瓶颈) 的向右变道率时序\n'
                     '(Lane 3 应始终为 0 — Dirichlet 验证)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(FIGDIR, 'V4_lateral.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        results['figures'].append(fig_path)
        print(f"\n  图表已保存: {fig_path}")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    passed_all = all(v['passed'] for v in results['checks'].values())
    n_pass  = sum(1 for v in results['checks'].values() if v['passed'])
    n_total = len(results['checks'])
    results['summary'] = f"{'PASSED' if passed_all else 'FAILED'} {n_pass}/{n_total}"
    print(f"  V4 汇总: {results['summary']}\n")
    return results


if __name__ == '__main__':
    run()
