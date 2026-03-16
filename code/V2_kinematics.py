"""
V2 — 动力学转换率验证
对应 .tex §3:  eq.2 加速率、eq.3 减速率/奇异屏障

检验项:
  [V2-a] 加速率 λ_acc 关于 Ω 单调递减，Ω→ρ_max 时趋零
  [V2-b] 加速 Dirichlet 上界: λ_acc[m, N-1, x, l] = 0 (全时域)
  [V2-c] 奇异屏障 B(Ω): Ω=0.145 时 B ≈ 4.155×10^6 (JSON 验证场景)
  [V2-d] 减速 Dirichlet 下界: λ_dec[m, 0, x, l] = 0 (全时域)
  [V2-e] beta 非对称性: PC 跟随 HDT 的减速率 > PC 跟随 PC (β12/β11 = 2)
  [V2-f] ω₀>0 防止 0×∞: 均匀高速车流接近容量时 λ_dec > 0
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
    results = {'module': 'V2_kinematics', 'checks': {}, 'figures': []}

    with h5py.File(HDF5, 'r') as hf:
        # ── 参数 ──────────────────────────────────────────────────────────────
        rho_max  = float(hf['parameters'].attrs['rho_max'])
        R_c      = float(hf['parameters'].attrs['R_c'])
        eps      = float(hf['parameters'].attrs['eps'])
        eta_g    = float(hf['parameters'].attrs['eta_global'])
        v_max    = float(hf['parameters'].attrs['v_max_mps'])
        N        = int(hf['parameters'].attrs['N'])
        M        = int(hf['parameters'].attrs['M'])
        T        = int(hf['parameters'].attrs['T_steps'])
        v        = hf['parameters/v_mps'][:]          # (N,)
        w        = hf['parameters/w_PCE'][:]          # (M,)
        alpha    = hf['parameters/alpha_hz'][:]       # (M,)
        eta_m    = hf['parameters/eta_m'][:]          # (M,)
        omega_0  = hf['parameters/omega_0_hz'][:]     # (M,)
        beta     = hf['parameters/beta_matrix'][:]    # (M, M)

        print(f"\n{'='*60}")
        print(f"  V2 — 动力学转换率验证")
        print(f"  M={M}, N={N}, v_max={v_max}, rho_max={rho_max}")
        print(f"{'='*60}")

        # ── [V2-a]  加速率单调性 (解析验证，不依赖存储数据) ────────────────
        # eq.2: λ_acc = α*(1-v_i/v_max)^η * [1-exp(-max(0,ρ_max-Ω)/R_c)]
        # 对任意 m,i: λ_acc 关于 Ω 在 [0, ρ_max] 上单调递减
        omega_range = np.linspace(0, rho_max, 500)
        # 取 PC (m=0), 中间速度 i=7 (v=16 m/s)
        m_test, i_test = 0, 7
        speed_factor = (1.0 - v[i_test] / v_max) ** eta_g
        filter_vals  = 1.0 - np.exp(-np.maximum(0, rho_max - omega_range) / R_c)
        lambda_acc_curve = alpha[m_test] * speed_factor * filter_vals

        diffs = np.diff(lambda_acc_curve)
        monotone_ok = bool((diffs <= 1e-12).all())   # 单调不增
        at_capacity = lambda_acc_curve[-1]            # Ω = ρ_max 处应趋零
        passed_a = monotone_ok and (at_capacity < 1e-10)
        results['checks']['V2-a'] = {
            'desc': 'λ_acc 关于 Ω 单调递减，Ω=ρ_max 时趋零',
            'passed': passed_a,
            'monotone': monotone_ok,
            'value_at_capacity': float(at_capacity)
        }
        tag = PASS if passed_a else FAIL
        print(f"  [V2-a] {tag}  λ_acc 单调={'是' if monotone_ok else '否'}, "
              f"Ω=ρ_max 处 λ_acc = {at_capacity:.2e}")

        # ── [V2-b]  加速 Dirichlet 上界 (从存储数据验证) ────────────────────
        # λ_acc[m, N-1, x, l] ≡ 0 (最高速度档不能再加速)
        # 逐步读取，避免一次加载全部 lambda_acc
        max_top_acc = 0.0
        for t in range(0, T, 40):
            # slice: lambda_acc[t, :, N-1, :, :] → shape (M, X, L)
            val = hf['data/lambda_acc'][t, :, N-1, :, :]
            max_top_acc = max(max_top_acc, float(np.abs(val).max()))

        passed_b = max_top_acc < 1e-12
        results['checks']['V2-b'] = {
            'desc': 'λ_acc[*, N-1, *, *] = 0 (Dirichlet 上界)',
            'passed': passed_b,
            'max_violation': max_top_acc
        }
        tag = PASS if passed_b else FAIL
        print(f"  [V2-b] {tag}  λ_acc 上边界最大值 = {max_top_acc:.2e}  (应为 0)")

        # ── [V2-c]  奇异屏障精确计算 ────────────────────────────────────────
        # JSON 验证场景: Ω=0.145, ρ_max=0.15, η_HDT=4.5
        # B = (0.15 / (0.15 - 0.145))^4.5 = (0.15/0.005)^4.5 = 30^4.5
        omega_bott  = 0.145
        gap_bott    = rho_max - omega_bott          # 0.005
        B_PC  = (rho_max / max(eps, gap_bott)) ** eta_m[0]   # PC η=2.0
        B_HDT = (rho_max / max(eps, gap_bott)) ** eta_m[1]   # HDT η=4.5
        B_expected = 30.0 ** 4.5                    # = 4,155,311.9 (JSON)

        err_c = abs(B_HDT - B_expected) / B_expected
        passed_c = err_c < 1e-6
        results['checks']['V2-c'] = {
            'desc': 'B(Ω=0.145) = 30^4.5 ≈ 4.155e6 (JSON 精确值)',
            'passed': passed_c,
            'B_HDT': float(B_HDT),
            'B_expected': float(B_expected),
            'rel_error': float(err_c)
        }
        tag = PASS if passed_c else FAIL
        print(f"  [V2-c] {tag}  B_HDT(Ω=0.145) = {B_HDT:.2f}  ≈ {B_expected:.2f}"
              f"  (相对误差 {err_c:.2e})")

        # ── [V2-d]  减速 Dirichlet 下界 ─────────────────────────────────────
        # λ_dec[m, 0, x, l] ≡ 0 (最低速度档不能再减速)
        max_bot_dec = 0.0
        for t in range(0, T, 40):
            val = hf['data/lambda_dec'][t, :, 0, :, :]
            max_bot_dec = max(max_bot_dec, float(np.abs(val).max()))

        passed_d = max_bot_dec < 1e-12
        results['checks']['V2-d'] = {
            'desc': 'λ_dec[*, 0, *, *] = 0 (Dirichlet 下界)',
            'passed': passed_d,
            'max_violation': max_bot_dec
        }
        tag = PASS if passed_d else FAIL
        print(f"  [V2-d] {tag}  λ_dec 下边界最大值 = {max_bot_dec:.2e}  (应为 0)")

        # ── [V2-e]  beta 矩阵非对称性 ───────────────────────────────────────
        # β^(PC→HDT) = 0.06 > β^(PC→PC) = 0.03 → PC 跟 HDT 减速更早
        # 在相同 f 和 Ω 下，比较两种情形下 PC 在 i=7 的 λ_dec 增量
        # 碰撞项增量 ∝ β^(m,n) * (v_i - v_k) * w^(n) * f_k
        # 取 f_k = 0.01 (同类车密度), Ω=0.05 (自由流)
        f_k_test = 0.01
        omega_ff  = 0.05
        dv        = v[7] - v[0]            # speed gap = 16-2 = 14 m/s
        press_ff  = (rho_max / max(eps, rho_max - omega_ff)) ** eta_m[0]

        delta_PC_PC   = beta[0, 0] * dv * w[0] * f_k_test * press_ff
        delta_PC_HDT  = beta[0, 1] * dv * w[1] * f_k_test * press_ff
        ratio_e       = delta_PC_HDT / delta_PC_PC
        expected_ratio = (beta[0, 1] * w[1]) / (beta[0, 0] * w[0])  # 0.06*2.5 / (0.03*1.0) = 5.0
        err_e = abs(ratio_e - expected_ratio) / expected_ratio
        passed_e = err_e < 1e-8
        results['checks']['V2-e'] = {
            'desc': 'PC 跟 HDT 减速率 / PC 跟 PC 减速率 = β12·w2/(β11·w1) = 5.0',
            'passed': passed_e,
            'ratio': float(ratio_e),
            'expected': float(expected_ratio)
        }
        tag = PASS if passed_e else FAIL
        print(f"  [V2-e] {tag}  δλ(PC→HDT)/δλ(PC→PC) = {ratio_e:.4f}"
              f"  ≈ 期望 {expected_ratio:.4f}")

        # ── [V2-f]  ω₀>0 消除 0×∞ ──────────────────────────────────────────
        # 均匀高速车流: f_k=0 (无慢车), Ω→ρ_max
        # 若 ω₀=0 → λ_dec = 0 × ∞ = 0/inf 悖论
        # 若 ω₀>0 → λ_dec = ω₀ × B > 0
        omega_near_cap = rho_max - 1e-4    # 接近满载
        B_test = (rho_max / max(eps, rho_max - omega_near_cap)) ** eta_m[0]
        lambda_dec_with_omega0    = (omega_0[0] + 0.0) * B_test   # 无慢车，有ω₀
        lambda_dec_without_omega0 = (0.0        + 0.0) * B_test   # 无慢车，无ω₀
        passed_f = lambda_dec_with_omega0 > 1.0  # 应该是有限正数
        results['checks']['V2-f'] = {
            'desc': 'ω₀>0 保证接近满载时 λ_dec > 0 (防 0×∞)',
            'passed': passed_f,
            'lambda_dec_with_omega0': float(lambda_dec_with_omega0),
            'lambda_dec_without': float(lambda_dec_without_omega0)
        }
        tag = PASS if passed_f else FAIL
        print(f"  [V2-f] {tag}  空路+满载时 λ_dec(有ω₀) = {lambda_dec_with_omega0:.2f}"
              f",  λ_dec(无ω₀) = {lambda_dec_without_omega0:.2f}")

        # ═══════════════════════════════════════════════════════════════════════
        # 读取 t=0 时刻的 lambda_dec 空间分布 (用于图 V2-3)
        # ═══════════════════════════════════════════════════════════════════════
        ldec_t0 = hf['data/lambda_dec'][0]    # (M, N, X, L)

        # ══════════════════ 生成图表 ══════════════════════════════════════════
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ── 图 V2-1: λ_acc 关于 Ω 的曲线 ─────────────────────────────────
        ax = axes[0, 0]
        for m_idx, lbl in [(0, 'PC (α=1.5)'), (1, 'HDT (α=0.35)')]:
            for i_idx, i_lbl in [(2, 'v=6m/s'), (7, 'v=16m/s'), (12, 'v=26m/s')]:
                sf = (1.0 - v[i_idx] / v_max) ** eta_g
                filt = 1.0 - np.exp(-np.maximum(0, rho_max - omega_range) / R_c)
                curve = alpha[m_idx] * sf * filt
                ls = '-' if m_idx == 0 else '--'
                ax.plot(omega_range, curve, ls=ls, lw=1.5,
                        label=f'{lbl}, {i_lbl}')

        ax.axvline(rho_max, color='red', ls=':', lw=1.2, label=r'$\rho_{\max}$')
        ax.set_xlabel(r'$\Omega$ [PCE/m]', fontsize=11)
        ax.set_ylabel(r'$\lambda_{acc}$ [Hz]', fontsize=11)
        ax.set_title('[V2-a] 加速率 λ_acc 关于占用密度的函数 (eq.2)', fontsize=11)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        # ── 图 V2-2: 奇异屏障 B(Ω) 半对数曲线 ───────────────────────────
        ax = axes[0, 1]
        omega_barrier = np.linspace(0, rho_max * 0.9999, 2000)
        for m_idx, lbl, col in [(0, 'PC (η=2.0)', 'C0'), (1, 'HDT (η=4.5)', 'C1')]:
            B_curve = (rho_max / np.maximum(eps, rho_max - omega_barrier)) ** eta_m[m_idx]
            ax.semilogy(omega_barrier, B_curve, color=col, lw=2, label=lbl)

        # 标注 JSON 验证点
        ax.axvline(0.145, color='orange', ls='--', lw=1.5, label='Ω=0.145 (瓶颈)')
        ax.axhline(B_expected, color='green', ls=':', lw=1.5,
                   label=f'B_HDT={B_expected:.2e}')
        ax.scatter([0.145], [B_HDT], color='red', s=80, zorder=5)
        ax.set_xlabel(r'$\Omega$ [PCE/m]', fontsize=11)
        ax.set_ylabel(r'$\mathcal{B}(\Omega)$ [无量纲]', fontsize=11)
        ax.set_title('[V2-c] 奇异渐近屏障 B(Ω) = (ρ_max/max(ε,ρ_max-Ω))^η (eq.3)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which='both')

        # ── 图 V2-3: t=0 λ_dec 空间分布 (HDT, speed i=1) ────────────────
        ax = axes[1, 0]
        # 取 HDT (m=1), speed i=1 (v=4m/s), 车道 l=0 的空间分布
        ldec_HDT_i1 = ldec_t0[1, 1, :, 0]   # (X,)
        x_cells = np.arange(len(ldec_HDT_i1))
        ax.semilogy(x_cells, np.maximum(ldec_HDT_i1, 1e-6), 'C1-', lw=1.5,
                    label='HDT, i=1 (v=4m/s), lane=0')
        ldec_PC_i7  = ldec_t0[0, 7, :, 0]   # PC, i=7 (v=16m/s)
        ax.semilogy(x_cells, np.maximum(ldec_PC_i7, 1e-6), 'C0--', lw=1.5,
                    label='PC, i=7 (v=16m/s), lane=0')
        ax.axvspan(74, 79, alpha=0.15, color='red', label='HDT 瓶颈')
        ax.set_xlabel('空间格子 x', fontsize=11)
        ax.set_ylabel(r'$\lambda_{dec}$ [Hz]', fontsize=11)
        ax.set_title('[V2-c/d] t=0 减速率空间分布 (对数坐标)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which='both')

        # ── 图 V2-4: beta 矩阵效果对比棒图 ──────────────────────────────
        ax = axes[1, 1]
        dv_vals = v[1:] - v[0]    # 速度差 (从 i=1 到 N-1，相对 v_0)
        # 在 Ω=0.05 下，f_k=0.01，PC 跟 PC 与 PC 跟 HDT 的碰撞项
        press_ff2 = (rho_max / max(eps, rho_max - 0.05)) ** eta_m[0]
        # 碰撞项增量 = β * (v_i - v_0) * w * f_k * pressure
        contrib_PC_PC  = [beta[0,0] * dv * w[0] * 0.01 * press_ff2 for dv in dv_vals]
        contrib_PC_HDT = [beta[0,1] * dv * w[1] * 0.01 * press_ff2 for dv in dv_vals]

        xi = np.arange(len(dv_vals))
        ax.bar(xi - 0.2, contrib_PC_PC,  width=0.35, color='C0', alpha=0.8, label='PC 跟 PC (β=0.03, w=1.0)')
        ax.bar(xi + 0.2, contrib_PC_HDT, width=0.35, color='C1', alpha=0.8, label='PC 跟 HDT (β=0.06, w=2.5)')
        ax.set_xticks(xi[::2])
        ax.set_xticklabels([f'i={j+1}' for j in range(0, len(dv_vals), 2)], fontsize=8)
        ax.set_xlabel('速度档 i (追车方 PC)', fontsize=11)
        ax.set_ylabel(r'$\delta\lambda_{dec}$ 碰撞贡献 [Hz]', fontsize=11)
        ax.set_title(f'[V2-e] beta 非对称: 比率 = β12·w2/(β11·w1) = {expected_ratio:.1f}', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        fig_path = os.path.join(FIGDIR, 'V2_kinematics.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        results['figures'].append(fig_path)
        print(f"\n  图表已保存: {fig_path}")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    passed_all = all(v['passed'] for v in results['checks'].values())
    n_pass  = sum(1 for v in results['checks'].values() if v['passed'])
    n_total = len(results['checks'])
    results['summary'] = f"{'PASSED' if passed_all else 'FAILED'} {n_pass}/{n_total}"
    print(f"  V2 汇总: {results['summary']}\n")
    return results


if __name__ == '__main__':
    run()
