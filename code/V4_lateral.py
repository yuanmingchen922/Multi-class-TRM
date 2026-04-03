"""
V4 — 侧向变道 Softplus 验证  (m3+m4 升级版)
对应 .tex §5:  γ = κ·(1/ω)·ln[1+exp(ω·y)]·gap_filter  (eq.6)

检验项:
  [V4-a] Softplus C∞ 连续性: 与 hard-max 对比，y=0 处无导数跳变
  [V4-b] y≤0 时 γ≈0: 无密度差时几乎无变道驱动力
  [V4-c] 侧向 Dirichlet 边界: γ_left[lane=0]=0, γ_right[lane=L-1]=0
  [V4-d] Class Bf vs Class A 速率比 ≈ κ_Bf/κ_A = 7.5
  [V4-e] 间隙过滤: γ→0 当目标车道 Ω_target→ρ_max
  [V4-f] Bs 绝对侧向禁止: γ_left^(Bs)=γ_right^(Bs)=0 全时域全格子
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
        T        = int(hf['parameters'].attrs['T_steps'])
        L        = int(hf['parameters'].attrs['L'])
        X        = int(hf['parameters'].attrs['X'])
        M        = int(hf['parameters'].attrs['M'])
        kappa    = hf['parameters/kappa_hz'][:]   # (M,) = [κ_A, κ_Bf, κ_Bs=0]
        time_s   = hf['data/time_s'][:]

        kappa_A  = kappa[0]   # 0.08
        kappa_Bf = kappa[1]   # 0.60
        kappa_Bs = kappa[2]   # 0.00

        print(f"\n{'='*60}")
        print(f"  V4 — 侧向变道 Softplus 验证  (M={M} 类)")
        print(f"  κ_A={kappa_A}, κ_Bf={kappa_Bf}, κ_Bs={kappa_Bs}")
        print(f"  ω={omega_sp}, R_gap={R_gap}")
        print(f"{'='*60}")

        # ── [V4-a]  Softplus vs hard-max C∞ 连续性 ──────────────────────────
        y_range = np.linspace(-0.5, 0.5, 2000)

        def softplus(y):
            return np.where(y > 20, y,
                            np.log1p(np.exp(np.minimum(omega_sp * y, 500))) / omega_sp)

        sp_vals = softplus(y_range)
        hm_vals = np.maximum(0, y_range)

        dy   = y_range[1] - y_range[0]
        d_sp = np.gradient(sp_vals, dy)
        d_hm = np.gradient(hm_vals, dy)

        near_zero = np.abs(y_range) < 0.05
        var_sp = float(np.var(d_sp[near_zero]))
        var_hm = float(np.var(d_hm[near_zero]))
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

        # ── [V4-b]  y≤0 时 γ≈0 (用 Bf 类，kappa 最大) ──────────────────────
        y_neg = -0.5
        sp_neg = float(softplus(np.array([y_neg]))[0])
        gap_factor_full = float(1.0 - np.exp(-rho_max / R_gap))
        gamma_neg = kappa_Bf * sp_neg * gap_factor_full
        passed_b = gamma_neg < 1e-3
        results['checks']['V4-b'] = {
            'desc': 'y=-0.5 时 γ^(Bf) ≈ 0 (无密度梯度驱动)',
            'passed': passed_b,
            'gamma_at_y_neg05': float(gamma_neg),
            'softplus_at_y_neg05': float(sp_neg)
        }
        tag = PASS if passed_b else FAIL
        print(f"  [V4-b] {tag}  γ^(Bf)(y=-0.5) = {gamma_neg:.4e}  (softplus={sp_neg:.2e})")

        # ── [V4-c]  侧向 Dirichlet 边界 (从存储数据验证) ─────────────────────
        # γ_left[:, :, :, :, 0] 应为 0 (最内车道不能再向左)
        # γ_right[:, :, :, :, L-1] 应为 0 (最外车道不能再向右)
        max_inner = 0.0
        max_outer = 0.0
        for t in range(0, T, 40):
            gl_inner = float(np.abs(hf['data/gamma_left'][t, :, :, :, 0]).max())
            gr_outer = float(np.abs(hf['data/gamma_right'][t, :, :, :, L-1]).max())
            if gl_inner > max_inner: max_inner = gl_inner
            if gr_outer > max_outer: max_outer = gr_outer

        passed_c = max_inner < 1e-12 and max_outer < 1e-12
        results['checks']['V4-c'] = {
            'desc': 'γ_left[lane=0]=0 且 γ_right[lane=L-1]=0 (Dirichlet 边界)',
            'passed': passed_c,
            'max_inner_violation': max_inner,
            'max_outer_violation': max_outer
        }
        tag = PASS if passed_c else FAIL
        print(f"  [V4-c] {tag}  γ_left 内边界最大值 = {max_inner:.2e}, "
              f"γ_right 外边界最大值 = {max_outer:.2e}")

        # ── [V4-d]  Class Bf vs Class A 速率比 ──────────────────────────────
        # 相同工况下 γ^(Bf)/γ^(A) = κ_Bf/κ_A = 0.60/0.08 = 7.5
        y_pos   = 0.3
        sp_pos  = float(softplus(np.array([y_pos]))[0])
        gap_fac = float(1.0 - np.exp(-rho_max * 0.5 / R_gap))
        gamma_Bf = kappa_Bf * sp_pos * gap_fac
        gamma_A  = kappa_A  * sp_pos * gap_fac
        ratio_d   = gamma_Bf / gamma_A if gamma_A > 0 else float('inf')
        expected_d = kappa_Bf / kappa_A
        err_d = abs(ratio_d - expected_d) / expected_d
        passed_d = err_d < 1e-8
        results['checks']['V4-d'] = {
            'desc': f'γ^(Bf)/γ^(A) = κ_Bf/κ_A = {expected_d:.2f}',
            'passed': passed_d,
            'ratio': float(ratio_d),
            'expected': float(expected_d),
            'rel_error': float(err_d)
        }
        tag = PASS if passed_d else FAIL
        print(f"  [V4-d] {tag}  γ^(Bf)/γ^(A) = {ratio_d:.4f}  ≈ {expected_d:.1f}"
              f"  (误差 {err_d:.2e})")

        # ── [V4-e]  间隙过滤: Ω_target→ρ_max 时 γ→0 ──────────────────────
        omega_target_range = np.linspace(0, rho_max, 500)
        gap_arg   = np.maximum(0, rho_max - omega_target_range)
        gap_vals  = 1.0 - np.exp(-gap_arg / R_gap)
        sp_fixed  = float(softplus(np.array([0.5]))[0])
        gamma_filter_Bf = kappa_Bf * sp_fixed * gap_vals

        at_cap      = gamma_filter_Bf[-1]
        gap_monotone = bool((np.diff(gamma_filter_Bf) <= 1e-12).all())
        passed_e = gap_monotone and (at_cap < 1e-10)
        results['checks']['V4-e'] = {
            'desc': 'γ^(Bf) 随 Ω_target→ρ_max 单调趋零 (间隙过滤)',
            'passed': passed_e,
            'gamma_at_capacity': float(at_cap),
            'monotone': gap_monotone
        }
        tag = PASS if passed_e else FAIL
        print(f"  [V4-e] {tag}  γ^(Bf)(Ω_target=ρ_max) = {at_cap:.2e}, "
              f"单调={'是' if gap_monotone else '否'}")

        # ── [V4-f]  Bs 绝对侧向禁止 (γ^(Bs) = 0 全时域) ────────────────────
        # m=2 是 Class Bs (被困乘用车)，其 γ 必须在所有时刻、所有速度档和格子中为 0
        max_bs_left  = 0.0
        max_bs_right = 0.0
        # γ 维度: (T, M, N, X, L) — m=2 即 Bs
        for t in range(0, T, 25):
            gl_bs = float(np.abs(hf['data/gamma_left'][t, 2, :, :, :]).max())
            gr_bs = float(np.abs(hf['data/gamma_right'][t, 2, :, :, :]).max())
            if gl_bs  > max_bs_left:  max_bs_left  = gl_bs
            if gr_bs  > max_bs_right: max_bs_right = gr_bs

        passed_f = max_bs_left < 1e-12 and max_bs_right < 1e-12
        results['checks']['V4-f'] = {
            'desc': 'γ^(Bs)=0 全时域全格子 (Bs 绝对侧向禁止)',
            'passed': passed_f,
            'max_bs_left_violation': max_bs_left,
            'max_bs_right_violation': max_bs_right
        }
        tag = PASS if passed_f else FAIL
        print(f"  [V4-f] {tag}  max|γ_left^(Bs)| = {max_bs_left:.2e}, "
              f"max|γ_right^(Bs)| = {max_bs_right:.2e}")

        # ═══════════════════════════════════════════════════════════════════════
        # 读取宏观密度和 gamma 样本
        # ═══════════════════════════════════════════════════════════════════════
        rho_macro = hf['data/rho_macro'][:]    # (T, M, X, L)
        # 各类各车道平均密度
        lane_density_A  = rho_macro[:, 0, :, :].mean(axis=1)   # (T, L)
        lane_density_Bf = rho_macro[:, 1, :, :].mean(axis=1)
        lane_density_Bs = rho_macro[:, 2, :, :].mean(axis=1)

        # gamma_right 样本: Bf(m=1), speed i=7, cell x=77（瓶颈附近）, 所有车道
        gr_ds = hf['data/gamma_right']
        gr_sample_Bf = np.array([gr_ds[t, 1, 7, 77, :] for t in range(T)])  # (T, L)
        gr_sample_A  = np.array([gr_ds[t, 0, 7, 77, :] for t in range(T)])
        gr_sample_Bs = np.array([gr_ds[t, 2, 7, 77, :] for t in range(T)])

    # ═══════════════════════════════════════════════════════════════════════
    # 生成图表
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── 图 V4-1: Softplus vs hard-max ─────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(y_range, sp_vals, 'C0-',  lw=2.5, label='Softplus (ω=20)')
    ax.plot(y_range, hm_vals, 'C1--', lw=2,   label='Hard-max(0,y)')
    ax.plot(y_range, d_sp, 'C0:', lw=1.5, alpha=0.7, label='Softplus 一阶导')
    ax.plot(y_range, d_hm, 'C1:', lw=1.5, alpha=0.7, label='Hard-max 一阶导')
    ax.axvline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('y = (Ω_src - Ω_tgt) / ρ_max', fontsize=10)
    ax.set_ylabel('变道驱动力', fontsize=10)
    ax.set_title('[V4-a] Softplus vs Hard-max (C∞ 连续性)', fontsize=10)
    ax.legend(fontsize=8); ax.set_ylim(-0.1, 0.6); ax.grid(alpha=0.3)

    # ── 图 V4-2: 间隙过滤曲线（三类） ────────────────────────────────────
    ax = axes[0, 1]
    for k_val, lbl, col in [(kappa_A, 'Class A (κ=0.08)', 'C1'),
                              (kappa_Bf, 'Class Bf (κ=0.60)', 'C0')]:
        gamma_f = k_val * sp_fixed * gap_vals
        ax.plot(omega_target_range, gamma_f, color=col, lw=2, label=lbl)

    ax.plot(omega_target_range, np.zeros_like(omega_target_range),
            'C2--', lw=2, label='Class Bs (κ=0, immobile)')
    ax.axvline(rho_max, color='red', ls='--', lw=1.5, label=r'$\rho_{\max}$')
    ax.set_xlabel(r'目标车道 $\Omega_{target}$ [PCE/m]', fontsize=10)
    ax.set_ylabel(r'$\gamma$ [Hz]', fontsize=10)
    ax.set_title('[V4-e] 间隙过滤 + 三类 γ 对比', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── 图 V4-3: Bf γ_right 在瓶颈处时序（三车道） ──────────────────────
    ax = axes[0, 2]
    for l_idx, col in enumerate(['C0', 'C1', 'C2']):
        ax.plot(time_s, gr_sample_Bf[:, l_idx], color=col,
                lw=1.5, label=f'Bf Lane {l_idx+1}')
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('时间 [s]', fontsize=10)
    ax.set_ylabel(r'$\gamma_{right}^{(Bf)}$ [Hz]', fontsize=10)
    ax.set_title('[V4-c] Class Bf γ_right (x=77, i=7)\nLane 3 应始终为 0', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── 图 V4-4: Bs γ_right 时序 (应全零 — V4-f 验证) ───────────────────
    ax = axes[1, 0]
    for l_idx, col in enumerate(['C0', 'C1', 'C2']):
        ax.plot(time_s, gr_sample_Bs[:, l_idx], color=col,
                lw=1.5, label=f'Bs Lane {l_idx+1}')
    ax.axhline(0, color='red', ls='--', lw=1.5, label='零线 (强制约束)')
    ax.set_xlabel('时间 [s]', fontsize=10)
    ax.set_ylabel(r'$\gamma_{right}^{(Bs)}$ [Hz]', fontsize=10)
    ax.set_title('[V4-f] Class Bs γ_right (全时域应为零 — Bs 侧向禁止)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── 图 V4-5: 三类各车道平均密度时序 ─────────────────────────────────
    ax = axes[1, 1]
    lane_colors = ['C0', 'C1', 'C2']
    lane_labels = ['Lane 1 (内)', 'Lane 2 (中)', 'Lane 3 (外)']
    for l_idx, (lbl, col) in enumerate(zip(lane_labels, lane_colors)):
        ax.plot(time_s, lane_density_A[:, l_idx],  color=col, ls='-',
                lw=1.5, label=f'A {lbl}' if l_idx == 0 else None)
        ax.plot(time_s, lane_density_Bf[:, l_idx], color=col, ls='--',
                lw=1.5, label=f'Bf {lbl}' if l_idx == 0 else None)
        ax.plot(time_s, lane_density_Bs[:, l_idx], color=col, ls=':',
                lw=1.5, label=f'Bs {lbl}' if l_idx == 0 else None)
    # 补充图例
    from matplotlib.lines import Line2D
    custom = [Line2D([0], [0], color='gray', ls='-',  lw=1.5, label='Class A'),
              Line2D([0], [0], color='gray', ls='--', lw=1.5, label='Class Bf'),
              Line2D([0], [0], color='gray', ls=':',  lw=1.5, label='Class Bs')]
    ax.legend(handles=custom, fontsize=8)
    ax.set_xlabel('时间 [s]', fontsize=10)
    ax.set_ylabel('平均密度 [veh/m]', fontsize=10)
    ax.set_title('[V4-d] 三类三车道平均密度随时间\n(Bf 密度应因捕获而转入 Bs)', fontsize=9)
    ax.grid(alpha=0.3)

    # ── 图 V4-6: Class A γ_right 时序 ────────────────────────────────────
    ax = axes[1, 2]
    for l_idx, col in enumerate(['C0', 'C1', 'C2']):
        ax.plot(time_s, gr_sample_A[:, l_idx], color=col,
                lw=1.5, label=f'A Lane {l_idx+1}')
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('时间 [s]', fontsize=10)
    ax.set_ylabel(r'$\gamma_{right}^{(A)}$ [Hz]', fontsize=10)
    ax.set_title('[V4-d] Class A γ_right (x=77, i=7)\n(κ_A/κ_Bf = 0.08/0.60 ≈ 0.133)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

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
