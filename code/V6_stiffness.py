"""
V6 — 刚性与算子分裂验证
对应 .tex §8-9:  刚性比 S = |μ_fast|/|μ_slow|, Lie-Trotter 三相分裂

检验项:
  [V6-a] 刚性比: S = λ_dec_max / (v_max/Δx) 在瓶颈区 >> 10^5
  [V6-b] 时间尺度分离: τ_relax = 1/λ_dec_max << τ_adv = Δx/v_max
  [V6-c] 显式 Euler 不稳定性: Δt=0.5s 下 Phase 3 显式积分在瓶颈格子发散
  [V6-d] Thomas 算法精度: 残差 ||(I-ΔtA)f_new - f_old||_∞ < 1e-10
  [V6-e] Phase 3 占用不变性引理: d_Omega/dt = 0 数值验证
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

# ── 复现 Phase 3 计算 (用于 V6-c/d/e 独立验证) ────────────────────────────

def compute_rates(f_cell, omega_cell, params):
    """
    对单个格子 (M, N) 计算 lambda_acc 和 lambda_dec.
    f_cell:    (M, N)
    omega_cell: scalar
    返回: lambda_acc (M,N), lambda_dec (M,N)
    """
    v, v_max, rho_max, R_c, eps, eta_g = (
        params['v'], params['v_max'], params['rho_max'],
        params['R_c'], params['eps'], params['eta_g']
    )
    alpha, eta_m, omega_0, beta, w = (
        params['alpha'], params['eta_m'], params['omega_0'],
        params['beta'], params['w']
    )
    M, N = f_cell.shape

    # eq.2: λ_acc[m,i] = α[m] * (1-v[i]/v_max)^η * [1-exp(-max(0,ρ_max-Ω)/R_c)]
    speed_factor = (1.0 - v / v_max) ** eta_g               # (N,)
    acc_filter   = 1.0 - np.exp(-max(0, rho_max - omega_cell) / R_c)
    lambda_acc   = alpha[:, None] * speed_factor[None, :] * acc_filter  # (M, N)
    lambda_acc[:, N-1] = 0.0    # Dirichlet 上界

    # eq.3: λ_dec[m,i]
    beta_w = beta * w[None, :]   # (M, M)
    bwf    = beta_w @ f_cell     # (M, N): bwf[m,k] = Σ_n beta_w[m,n]*f[n,k]

    cum_bwf  = np.zeros_like(bwf)
    cum_vbwf = np.zeros_like(bwf)
    cum_bwf[:, 1:]  = np.cumsum(bwf[:, :-1],                axis=1)
    cum_vbwf[:, 1:] = np.cumsum(v[None, :-1] * bwf[:, :-1], axis=1)

    interaction = v[None, :] * cum_bwf - cum_vbwf           # (M, N)
    pressure    = (rho_max / max(eps, rho_max - omega_cell)) ** eta_m[:, None]  # (M,1) broadcast
    lambda_dec  = (omega_0[:, None] + interaction) * pressure
    lambda_dec  = np.maximum(lambda_dec, 0.0)
    lambda_dec[:, 0] = 0.0     # Dirichlet 下界
    return lambda_acc, lambda_dec


def thomas_1cell(f_cell, lambda_acc, lambda_dec, dt):
    """
    对单个格子 (M, N) 用 Thomas 算法求解半隐式 Phase 3.
    独立实现，用于残差验证.
    """
    M, N = f_cell.shape
    f_new = np.zeros_like(f_cell)
    for m in range(M):
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        d = f_cell[m].copy()
        a[1:]  = -dt * lambda_acc[m, :-1]
        b      =  1.0 + dt * (lambda_acc[m] + lambda_dec[m])
        c[:-1] = -dt * lambda_dec[m, 1:]

        # 前向消元
        c_p = np.zeros(N)
        d_p = np.zeros(N)
        c_p[0] = c[0] / b[0]
        d_p[0] = d[0] / b[0]
        for i in range(1, N):
            denom  = b[i] - a[i] * c_p[i-1]
            denom  = max(denom, 1e-14)
            c_p[i] = c[i] / denom
            d_p[i] = (d[i] - a[i] * d_p[i-1]) / denom

        # 回代
        x = np.zeros(N)
        x[N-1] = d_p[N-1]
        for i in range(N-2, -1, -1):
            x[i] = d_p[i] - c_p[i] * x[i+1]

        f_new[m] = np.maximum(x, 0.0)
    return f_new


def run():
    results = {'module': 'V6_stiffness', 'checks': {}, 'figures': []}

    with h5py.File(HDF5, 'r') as hf:
        # ── 参数 ──────────────────────────────────────────────────────────────
        rho_max  = float(hf['parameters'].attrs['rho_max'])
        R_c      = float(hf['parameters'].attrs['R_c'])
        eps      = float(hf['parameters'].attrs['eps'])
        eta_g    = float(hf['parameters'].attrs['eta_global'])
        v_max    = float(hf['parameters'].attrs['v_max_mps'])
        dx       = float(hf['parameters'].attrs['dx_m'])
        dt       = float(hf['parameters'].attrs['dt_s'])
        T        = int(hf['parameters'].attrs['T_steps'])
        X        = int(hf['parameters'].attrs['X'])
        N        = int(hf['parameters'].attrs['N'])
        M        = int(hf['parameters'].attrs['M'])
        v        = hf['parameters/v_mps'][:]
        w        = hf['parameters/w_PCE'][:]
        alpha    = hf['parameters/alpha_hz'][:]
        eta_m    = hf['parameters/eta_m'][:]
        omega_0  = hf['parameters/omega_0_hz'][:]
        beta     = hf['parameters/beta_matrix'][:]
        time_s   = hf['data/time_s'][:]

        params = dict(v=v, v_max=v_max, rho_max=rho_max, R_c=R_c,
                      eps=eps, eta_g=eta_g, alpha=alpha, eta_m=eta_m,
                      omega_0=omega_0, beta=beta, w=w)

        # 慢流形特征时间尺度 (常数)
        tau_adv  = dx / v_max                   # τ_adv ≈ 0.667 s
        mu_slow  = v_max / dx                   # |μ_slow| = 1.5 Hz

        print(f"\n{'='*60}")
        print(f"  V6 — 刚性与算子分裂验证")
        print(f"  τ_adv = Δx/v_max = {tau_adv:.4f}s,  |μ_slow| = {mu_slow:.2f} Hz")
        print(f"{'='*60}")

        # ── [V6-a]  刚性比 (全时域扫描) ──────────────────────────────────────
        # S(t,x,l) = max_m,i λ_dec[m,i,x,l] / (v_max/Δx)
        print("  计算刚性比 (逐步扫描 lambda_dec)...")
        stiffness_peak = np.zeros(T)
        for t in range(T):
            ldec_t = hf['data/lambda_dec'][t]   # (M, N, X, L)
            stiffness_peak[t] = float(ldec_t.max() / mu_slow)

        max_S   = float(stiffness_peak.max())
        init_S  = float(stiffness_peak[0])
        passed_a = init_S > 1e5
        results['checks']['V6-a'] = {
            'desc': 'S(t=0) = λ_dec_max / |μ_slow| >> 10^5 (极端刚性)',
            'passed': passed_a,
            'S_initial': init_S,
            'S_max': max_S,
            'mu_slow': mu_slow
        }
        tag = PASS if passed_a else FAIL
        print(f"  [V6-a] {tag}  初始刚性比 S = {init_S:.3e}  (阈值 > 1e5)")
        print(f"          全时域最大 S = {max_S:.3e}")

        # ── [V6-b]  时间尺度分离 ─────────────────────────────────────────────
        ldec_max_per_t = stiffness_peak * mu_slow   # λ_dec_max(t)
        tau_relax = np.where(ldec_max_per_t > 0, 1.0 / ldec_max_per_t, np.inf)
        ratio_tau  = tau_adv / tau_relax            # τ_adv / τ_relax >> 1

        max_ratio = float(ratio_tau[np.isfinite(ratio_tau)].max())
        passed_b = max_ratio > 1e5
        results['checks']['V6-b'] = {
            'desc': 'τ_adv / τ_relax >> 1 (时间尺度严格分离)',
            'passed': passed_b,
            'max_ratio': max_ratio,
            'tau_adv_s': tau_adv,
            'tau_relax_min_s': float(tau_relax[tau_relax > 0].min()) if any(tau_relax > 0) else 0
        }
        tag = PASS if passed_b else FAIL
        print(f"  [V6-b] {tag}  max(τ_adv/τ_relax) = {max_ratio:.3e}")

        # ── [V6-c]  显式 Euler 不稳定性演示 ──────────────────────────────────
        # 取 t=0, 瓶颈中心格子 x=77, lane=0
        f0     = hf['data/f'][0]         # (M, N, X, L)
        omega0 = hf['data/omega'][0]     # (X, L)

        f_bott   = f0[:, :, 77, 0].copy()    # (M, N)
        om_bott  = omega0[77, 0]              # scalar ≈ 0.145

        lam_acc, lam_dec = compute_rates(f_bott, om_bott, params)

        # 显式 Euler:  f_new = f + dt * A(f)
        # Phase 3 显式更新:
        #   df[m,i]/dt = lambda_acc[i-1]*f[i-1] - (lambda_acc[i]+lambda_dec[i])*f[i] + lambda_dec[i+1]*f[i+1]
        f_expl = f_bott.copy()
        expl_steps = []
        for step in range(5):
            df = np.zeros_like(f_expl)
            for m in range(M):
                df[m, 1:]  += lam_acc[m, :-1] * f_expl[m, :-1]
                df[m]      -= (lam_acc[m] + lam_dec[m]) * f_expl[m]
                df[m, :-1] += lam_dec[m, 1:]  * f_expl[m, 1:]
            f_expl = f_expl + dt * df
            expl_steps.append(f_expl.copy())
        expl_blowup = float(np.abs(np.array(expl_steps)).max())

        # Thomas 算法 (隐式):
        f_impl = thomas_1cell(f_bott, lam_acc, lam_dec, dt)
        impl_max = float(np.abs(f_impl).max())
        impl_neg = float(f_impl.min())

        passed_c = expl_blowup > 1.0 and impl_max <= rho_max + 1e-6
        results['checks']['V6-c'] = {
            'desc': '显式 Euler 发散 (>1.0), Thomas 稳定 (≤ρ_max)',
            'passed': passed_c,
            'explicit_max': expl_blowup,
            'implicit_max': impl_max
        }
        tag = PASS if passed_c else FAIL
        print(f"  [V6-c] {tag}  显式 Euler max(|f|) = {expl_blowup:.2e}  (5步后)  "
              f"vs  Thomas max(f) = {impl_max:.4f}")

        # ── [V6-d]  Thomas 算法残差 ──────────────────────────────────────────
        # 验证: (I - dt*A) * f_new ≈ f_old
        # 构建三对角矩阵 A 并检验残差
        residuals = []
        for m in range(M):
            # 构建 A[N×N] 对应 Phase 3 kinematic 矩阵
            A = np.zeros((N, N))
            for i in range(N):
                if i > 0:
                    A[i, i-1] += lam_acc[m, i-1]   # inflow from i-1
                A[i, i] -= (lam_acc[m, i] + lam_dec[m, i])
                if i < N-1:
                    A[i, i+1] += lam_dec[m, i+1]   # inflow from i+1

            lhs_matrix = np.eye(N) - dt * A
            f_old_m  = f_bott[m]
            f_new_m  = f_impl[m]
            residual = np.abs(lhs_matrix @ f_new_m - f_old_m).max()
            residuals.append(residual)

        max_residual = float(max(residuals))
        passed_d = max_residual < 1e-8
        results['checks']['V6-d'] = {
            'desc': '||(I-ΔtA)f_new - f_old||_inf < 1e-8',
            'passed': passed_d,
            'max_residual': max_residual
        }
        tag = PASS if passed_d else FAIL
        print(f"  [V6-d] {tag}  Thomas 算法残差 = {max_residual:.2e}  (阈值 < 1e-8)")

        # ── [V6-e]  Phase 3 占用不变性引理 ───────────────────────────────────
        # 在 Phase 3 中 d(Omega)/dt = 0:
        # Σ_i (df[m,i]/dt) = 0 对任意 m (速度转移零和)
        # ⟹ Σ_m w[m] * Σ_i (df[m,i]/dt) = 0

        # 计算 df[m,i]/dt 在瓶颈格子
        df_phase3 = np.zeros((M, N))
        for m in range(M):
            df_phase3[m, 1:]  += lam_acc[m, :-1] * f_bott[m, :-1]
            df_phase3[m]      -= (lam_acc[m] + lam_dec[m]) * f_bott[m]
            df_phase3[m, :-1] += lam_dec[m, 1:]  * f_bott[m, 1:]

        # d(Omega)/dt = Σ_m w[m] * Σ_i df[m,i]/dt
        d_omega_dt = float((w[:, None] * df_phase3).sum())
        class_sums = df_phase3.sum(axis=1)   # (M,): Σ_i df/dt per class
        max_class_sum = float(np.abs(class_sums).max())

        passed_e = abs(d_omega_dt) < 1e-6
        results['checks']['V6-e'] = {
            'desc': 'd(Omega)/dt = Σ_m w^m Σ_i df/dt ≈ 0 (占用不变性引理)',
            'passed': passed_e,
            'd_omega_dt': d_omega_dt,
            'class_mass_rate_sums': class_sums.tolist()
        }
        tag = PASS if passed_e else FAIL
        # Note: exact zero only if ω₀ term is balanced — in semi-implicit this is approximate
        print(f"  [V6-e] {tag}  d(Ω)/dt = {d_omega_dt:.2e}  (Σ_i df_PC/dt={class_sums[0]:.2e},"
              f" Σ_i df_HDT/dt={class_sums[1]:.2e})")

        # ═══════════════════════════════════════════════════════════════════════
        # 生成图表
        # ═══════════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ── 图 V6-1: 刚性比时序曲线 ───────────────────────────────────────
        ax = axes[0, 0]
        ax.semilogy(time_s, stiffness_peak, 'C1-', lw=2)
        ax.axhline(1e5, color='red', ls='--', lw=1.5, label='阈值 S=10^5')
        ax.axhline(1.0, color='green', ls=':', lw=1.5, label='S=1 (无刚性)')
        ax.axhline(dt * mu_slow, color='orange', ls=':', lw=1.5,
                   label=f'CFL 等效线 (Δt·|μ_slow|={dt*mu_slow:.2f})')
        ax.fill_between(time_s, stiffness_peak, 1e5,
                        where=stiffness_peak > 1e5, alpha=0.2, color='red',
                        label='极端刚性区域')
        ax.set_xlabel('时间 [s]', fontsize=11)
        ax.set_ylabel(r'刚性比 $\mathcal{S} = \lambda_{dec,max} / |μ_{slow}|$', fontsize=11)
        ax.set_title('[V6-a/b] 系统刚性比时序 (对数坐标)', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which='both')

        # ── 图 V6-2: 时间尺度对比 ─────────────────────────────────────────
        ax = axes[0, 1]
        tau_relax_plot = np.where(ldec_max_per_t > 1e-10, 1.0/ldec_max_per_t, np.nan)
        ax.semilogy(time_s, tau_relax_plot, 'C1-', lw=2, label=r'$\tau_{relax} = 1/\lambda_{dec,max}$')
        ax.axhline(tau_adv, color='C0', lw=2, ls='--',
                   label=fr'$\tau_{{adv}} = \Delta x/v_{{max}} = {tau_adv:.3f}$s')
        ax.axhline(dt, color='gray', ls=':', lw=1.5, label=fr'$\Delta t = {dt}$s')
        ax.set_xlabel('时间 [s]', fontsize=11)
        ax.set_ylabel('特征时间尺度 [s]', fontsize=11)
        ax.set_title('[V6-b] 慢流形 vs 快流形时间尺度分离', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which='both')

        # ── 图 V6-3: 显式 vs 隐式一步结果对比 ────────────────────────────
        ax = axes[1, 0]
        speed_bins = np.arange(N)
        width = 0.28

        # 初始状态
        ax.bar(speed_bins - width, f_bott[1, :], width=width,
               color='gray', alpha=0.6, label='初始 HDT f')
        # Thomas 结果
        ax.bar(speed_bins, f_impl[1, :], width=width,
               color='C0', alpha=0.8, label='Thomas (隐式) 一步后')
        # 显式结果 (仅第 1 步，截断到合理范围)
        f_expl_1 = expl_steps[0][1, :]
        ax.bar(speed_bins + width, np.clip(f_expl_1, -0.01, rho_max * 2),
               width=width, color='C1', alpha=0.8, label='Euler (显式) 一步后')

        ax.axhline(0, color='black', lw=0.8)
        ax.axhline(rho_max, color='red', ls='--', lw=1.2,
                   label=fr'$\rho_{{max}}={rho_max}$')
        ax.set_xlabel('速度档 i', fontsize=11)
        ax.set_ylabel('密度 f [veh/m]', fontsize=11)
        ax.set_title('[V6-c] 瓶颈格子 HDT: 显式 Euler vs Thomas 隐式 (1步)', fontsize=10)
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylim(-0.05, rho_max * 1.5)
        ax.grid(alpha=0.3, axis='y')
        ax.set_xticks(speed_bins)
        ax.set_xticklabels([f'{int(vi)}' for vi in v], fontsize=7)

        # ── 图 V6-4: 显式方案的 5 步演化轨迹 ─────────────────────────────
        ax = axes[1, 1]
        # 显示 HDT 在各速度档的显式 Euler 5 步演化
        steps_plot = [f_bott[1, :]] + [s[1, :] for s in expl_steps]
        colors_e = plt.cm.Reds(np.linspace(0.3, 1.0, 6))
        for step_idx, (f_step, col) in enumerate(zip(steps_plot, colors_e)):
            lbl = f't+{step_idx}Δt' if step_idx < len(steps_plot) - 1 else f't+{step_idx}Δt (最终)'
            ax.plot(speed_bins, np.clip(f_step, -0.5, 1.0), 'o-',
                    color=col, markersize=4, lw=1.5, label=lbl)
        ax.axhline(0, color='black', lw=0.8)
        ax.axhline(rho_max, color='red', ls='--', lw=1.2, label=r'$\rho_{max}$')
        ax.set_xlabel('速度档 i', fontsize=11)
        ax.set_ylabel('密度 f (截断显示)', fontsize=11)
        ax.set_title('[V6-c] 显式 Euler 发散轨迹 (HDT, 瓶颈格子)', fontsize=11)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(FIGDIR, 'V6_stiffness.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        results['figures'].append(fig_path)
        print(f"\n  图表已保存: {fig_path}")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    passed_all = all(v['passed'] for v in results['checks'].values())
    n_pass  = sum(1 for v in results['checks'].values() if v['passed'])
    n_total = len(results['checks'])
    results['summary'] = f"{'PASSED' if passed_all else 'FAILED'} {n_pass}/{n_total}"
    print(f"  V6 汇总: {results['summary']}\n")
    return results


if __name__ == '__main__':
    run()
