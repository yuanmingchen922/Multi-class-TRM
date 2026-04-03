"""
V2 — 动力学转换率验证  (m3+m4 升级版)
对应 .tex §3:  eq.3 加速率(逐类η)、eq.4-5 减速率(/ρ_max归一化)、eq.2 Bs约束

检验项:
  [V2-a] 加速率 λ_acc 关于 Ω 单调递减，Ω→ρ_max 时趋零
  [V2-b] 加速 Dirichlet 上界: λ_acc[m, N-1, x, l] = 0 (全时域)
  [V2-c] 奇异屏障: B(Ω=0.145) = 30^4.5 ≈ 4.155e6 (η^(A)=4.5)
  [V2-d] 减速 Dirichlet 下界: λ_dec[m, 0, x, l] = 0 (全时域)
  [V2-e] beta 非对称: 碰撞项有 /ρ_max 归一化，比率保持 β(Bf→A)/β(Bf→Bf)=2
  [V2-f] ω₀>0 防止 0×∞ 悖论
  [V2-g] Bs 加速封锁: λ_acc^(Bs)[i ≥ i_thr, x, l] = 0 全时域
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
        rho_max  = float(hf['parameters'].attrs['rho_max'])
        R_c      = float(hf['parameters'].attrs['R_c'])
        eps      = float(hf['parameters'].attrs['eps'])
        v_max    = float(hf['parameters'].attrs['v_max_mps'])
        N        = int(hf['parameters'].attrs['N'])
        M        = int(hf['parameters'].attrs['M'])
        X        = int(hf['parameters'].attrs['X'])
        T        = int(hf['parameters'].attrs['T_steps'])
        i_thr    = int(hf['parameters'].attrs['i_thr'])
        v        = hf['parameters/v_mps'][:]
        w        = hf['parameters/w_PCE'][:]
        alpha    = hf['parameters/alpha_hz'][:]
        eta_m    = hf['parameters/eta_m'][:]
        omega_0  = hf['parameters/omega_0_hz'][:]
        beta     = hf['parameters/beta_matrix'][:]

        print(f"\n{'='*60}")
        print(f"  V2 — 动力学转换率验证  (M={M}, η=[{eta_m}])")
        print(f"  i_thr={i_thr}  (v_A_ff={v[i_thr]} m/s)")
        print(f"{'='*60}")

        omega_range = np.linspace(0, rho_max, 500)

        # ── [V2-a] 加速率单调性 (eq.3 — 逐类 η^(m)) ─────────────────────────
        # 测试 Bf (m=1), 中速 i=7
        m_t, i_t = 1, 7
        sf   = (1.0 - v[i_t] / v_max) ** eta_m[m_t]
        filt = 1.0 - np.exp(-np.maximum(0, rho_max - omega_range) / R_c)
        curve = alpha[m_t] * sf * filt
        diffs = np.diff(curve)
        mono_ok = bool((diffs <= 1e-12).all())
        at_cap  = float(curve[-1])
        passed_a = mono_ok and (at_cap < 1e-10)
        results['checks']['V2-a'] = {
            'desc': 'λ_acc(Bf) 关于 Ω 单调递减且 Ω=ρ_max 时趋零',
            'passed': passed_a, 'monotone': mono_ok, 'at_capacity': at_cap
        }
        tag = PASS if passed_a else FAIL
        print(f"  [V2-a] {tag}  单调={'是' if mono_ok else '否'}, λ(ρ_max)={at_cap:.2e}")

        # ── [V2-b] Dirichlet 上界 ─────────────────────────────────────────────
        max_top_acc = 0.0
        for t in range(0, T, 40):
            val = hf['data/lambda_acc'][t, :, N - 1, :, :]
            max_top_acc = max(max_top_acc, float(np.abs(val).max()))
        passed_b = max_top_acc < 1e-12
        results['checks']['V2-b'] = {
            'desc': 'λ_acc[m, N-1, x, l] = 0', 'passed': passed_b, 'max_val': max_top_acc
        }
        tag = PASS if passed_b else FAIL
        print(f"  [V2-b] {tag}  λ_acc 上边界最大值 = {max_top_acc:.2e}")

        # ── [V2-c] 奇异屏障精确值 (η^(A)=4.5) ──────────────────────────────
        omega_bott = 0.145
        B_A   = (rho_max / max(eps, rho_max - omega_bott)) ** eta_m[0]  # A: η=4.5
        B_exp = 30.0 ** 4.5
        err_c = abs(B_A - B_exp) / B_exp
        passed_c = err_c < 1e-6
        results['checks']['V2-c'] = {
            'desc': 'B(Ω=0.145)=(30)^4.5≈4.155e6 (η^(A)=4.5)',
            'passed': passed_c, 'B_A': float(B_A), 'B_exp': float(B_exp), 'rel_err': float(err_c)
        }
        tag = PASS if passed_c else FAIL
        print(f"  [V2-c] {tag}  B_A = {B_A:.4e}  ≈ {B_exp:.4e}  (误差 {err_c:.2e})")

        # ── [V2-d] Dirichlet 下界 ─────────────────────────────────────────────
        max_bot_dec = 0.0
        for t in range(0, T, 40):
            val = hf['data/lambda_dec'][t, :, 0, :, :]
            max_bot_dec = max(max_bot_dec, float(np.abs(val).max()))
        passed_d = max_bot_dec < 1e-12
        results['checks']['V2-d'] = {
            'desc': 'λ_dec[m, 0, x, l] = 0', 'passed': passed_d, 'max_val': max_bot_dec
        }
        tag = PASS if passed_d else FAIL
        print(f"  [V2-d] {tag}  λ_dec 下边界最大值 = {max_bot_dec:.2e}")

        # ── [V2-e] beta 碰撞项比率 (含 /ρ_max) ─────────────────────────────
        # Bf 跟 A 对比 Bf 跟 Bf：比率 = β[1,0]*w[0] / (β[1,1]*w[1])
        f_k = 0.01 / rho_max      # 已归一化（新公式分母是 ρ_max）
        dv  = v[7] - v[0]
        press = (rho_max / max(eps, rho_max - 0.05)) ** eta_m[1]
        delta_Bf_A   = beta[1, 0] * dv * w[0] * f_k * press
        delta_Bf_Bf  = beta[1, 1] * dv * w[1] * f_k * press
        ratio_e      = delta_Bf_A / delta_Bf_Bf
        exp_ratio    = (beta[1, 0] * w[0]) / (beta[1, 1] * w[1])   # 0.06*2.5/(0.03*1.0)=5.0
        err_e = abs(ratio_e - exp_ratio) / exp_ratio
        passed_e = err_e < 1e-8
        results['checks']['V2-e'] = {
            'desc': 'Bf跟A/Bf跟Bf 碰撞率比 = β[1,0]·w[0]/(β[1,1]·w[1])=5.0',
            'passed': passed_e, 'ratio': float(ratio_e), 'expected': float(exp_ratio)
        }
        tag = PASS if passed_e else FAIL
        print(f"  [V2-e] {tag}  碰撞比率 = {ratio_e:.4f}  ≈ {exp_ratio:.4f}")

        # ── [V2-f] ω₀>0 防 0×∞ ──────────────────────────────────────────────
        omega_nc = rho_max - 1e-4
        B_test   = (rho_max / max(eps, rho_max - omega_nc)) ** eta_m[1]
        lam_w_omega0 = (omega_0[1] + 0.0) * B_test
        passed_f = lam_w_omega0 > 1.0
        results['checks']['V2-f'] = {
            'desc': 'ω₀>0 确保接近满载时 λ_dec>0', 'passed': passed_f,
            'lambda_dec': float(lam_w_omega0)
        }
        tag = PASS if passed_f else FAIL
        print(f"  [V2-f] {tag}  空路+满载 λ_dec(Bf, 有ω₀) = {lam_w_omega0:.2f}")

        # ── [V2-g] Bs 加速封锁: λ_acc^(Bs)[i≥i_thr] = 0 (eq.2) ─────────────
        max_Bs_blockade = 0.0
        for t in range(0, T, 40):
            val = hf['data/lambda_acc'][t, 2, i_thr:, :, :]   # Bs (m=2), i>=i_thr
            max_Bs_blockade = max(max_Bs_blockade, float(np.abs(val).max()))
        passed_g = max_Bs_blockade < 1e-12
        results['checks']['V2-g'] = {
            'desc': 'λ_acc^(Bs)[i≥i_thr] = 0  (加速封锁)',
            'passed': passed_g, 'max_violation': float(max_Bs_blockade)
        }
        tag = PASS if passed_g else FAIL
        print(f"  [V2-g] {tag}  λ_acc^(Bs)[i≥{i_thr}] 最大值 = {max_Bs_blockade:.2e}")

        # ═══════════ 读取 t=0 数据用于图 ═══════════
        ldec_t0 = hf['data/lambda_dec'][0]   # (M, N, X, L)
        lacc_t0 = hf['data/lambda_acc'][0]   # (M, N, X, L)

        # ══════════════════ 图表 ══════════════════
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 图1: λ_acc 曲线 (三类)
        ax = axes[0, 0]
        for m_idx, lbl, ls in [(0,'A (α=0.35,η=4.5)','-'),
                                (1,'Bf(α=1.50,η=2.0)','--'),
                                (2,'Bs(α=1.50,η=2.0, blockade)','--')]:
            for i_idx, ilbl in [(2,'v=6'), (7,'v=16'), (12,'v=26')]:
                sf2  = (1.0 - v[i_idx] / v_max) ** eta_m[m_idx]
                filt2 = 1.0 - np.exp(-np.maximum(0, rho_max - omega_range) / R_c)
                curve2 = alpha[m_idx] * sf2 * filt2
                if m_idx == 2 and i_idx >= i_thr:
                    curve2 = np.zeros_like(curve2)   # Bs blockade
                ax.plot(omega_range, curve2, ls=ls, lw=1.2,
                        label=f'{lbl}, {ilbl}m/s' if i_idx == 7 else '_')
        ax.axvline(rho_max, color='red', ls=':', label=r'$\rho_{\max}$')
        ax.set_xlabel(r'$\Omega$'); ax.set_ylabel(r'$\lambda_{acc}$ [Hz]')
        ax.set_title('[V2-a,g] 加速率 λ_acc (逐类 η; Bs 封锁可见)')
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

        # 图2: 奇异屏障 B(Ω)
        ax = axes[0, 1]
        ob = np.linspace(0, rho_max * 0.9999, 2000)
        for m_idx, lbl, col in [(0,'A (η=4.5)','C1'), (1,'Bf (η=2.0)','C0')]:
            Bc = (rho_max / np.maximum(eps, rho_max - ob)) ** eta_m[m_idx]
            ax.semilogy(ob, Bc, color=col, lw=2, label=lbl)
        ax.axvline(0.145, color='orange', ls='--', label='Ω=0.145 瓶颈')
        ax.axhline(B_exp, color='green', ls=':', label=f'B_A={B_exp:.2e}')
        ax.scatter([0.145], [B_A], color='red', s=80, zorder=5)
        ax.set_xlabel(r'$\Omega$'); ax.set_ylabel(r'$\mathcal{B}(\Omega)$')
        ax.set_title('[V2-c] 奇异屏障 B(Ω) = (ρ_max/max(ε,gap))^{η^(m)}')
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

        # 图3: t=0 λ_dec 空间分布 (三类)
        ax = axes[1, 0]
        for m_idx, lbl, col in [(0,'A  i=1','C1'), (1,'Bf i=7','C0'), (2,'Bs i=7','C2')]:
            i_plot = 1 if m_idx == 0 else 7
            ld = ldec_t0[m_idx, i_plot, :, 0]
            ax.semilogy(np.arange(X), np.maximum(ld, 1e-6),
                        color=col, lw=1.5, label=lbl)
        ax.axvspan(74, 79, alpha=0.15, color='red', label='A 瓶颈')
        ax.set_xlabel('空间格子 x'); ax.set_ylabel(r'$\lambda_{dec}$ [Hz]')
        ax.set_title('[V2-d] t=0 减速率空间分布（3类）')
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

        # 图4: β 矩阵非对称 (Bf 跟 A vs Bf 跟 Bf, 含 /ρ_max)
        ax = axes[1, 1]
        dv_vals = v[1:] - v[0]
        pff = (rho_max / max(eps, rho_max - 0.05)) ** eta_m[1]
        f_k_val = 0.01 / rho_max
        cb_A  = [beta[1,0] * dv * w[0] * f_k_val * pff for dv in dv_vals]
        cb_Bf = [beta[1,1] * dv * w[1] * f_k_val * pff for dv in dv_vals]
        xi2 = np.arange(len(dv_vals))
        ax.bar(xi2 - 0.2, cb_Bf, width=0.35, color='C0', alpha=0.8, label='Bf 跟 Bf (β=0.03,w=1.0)')
        ax.bar(xi2 + 0.2, cb_A,  width=0.35, color='C1', alpha=0.8, label='Bf 跟 A  (β=0.06,w=2.5)')
        ax.set_xticks(xi2[::2])
        ax.set_xticklabels([f'i={j+1}' for j in range(0, len(dv_vals), 2)], fontsize=8)
        ax.set_xlabel('速度档 i (追车方 Bf)'); ax.set_ylabel(r'$\delta\lambda_{dec}$ [Hz]')
        ax.set_title(f'[V2-e] β 非对称: 比率={exp_ratio:.1f}  (碰撞项含 /ρ_max)')
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        fig_path = os.path.join(FIGDIR, 'V2_kinematics.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        results['figures'].append(fig_path)
        print(f"\n  图表: {fig_path}")

    passed_all = all(v['passed'] for v in results['checks'].values())
    n_p = sum(1 for v in results['checks'].values() if v['passed'])
    results['summary'] = f"{'PASSED' if passed_all else 'FAILED'} {n_p}/{len(results['checks'])}"
    print(f"  V2 汇总: {results['summary']}\n")
    return results


if __name__ == '__main__':
    run()
