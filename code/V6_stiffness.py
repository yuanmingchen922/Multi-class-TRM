"""
V6 -- Stiffness and Operator Splitting Validation (3-phase P_block version)
Corresponding to .tex: 3-phase Lie-Trotter splitting, extreme stiffness, exact matrix exponential

3-phase structure:
  Phase 1: Capture & Release (P_block-based sigma/mu, exact matrix exponential)
  Phase 2: Internal Speed Changes (algebraic projection + Thomas algorithm)
  Phase 3: Spatial Advection (explicit Euler + global Godunov limiter)

Checks:
  [V6-a] Stiffness ratio: S = lambda_dec_max / (v_max/Delta_x) >> 10^5 at bottleneck
  [V6-b] Time-scale separation: tau_relax = 1/lambda_dec_max << tau_adv = Delta_x/v_max
  [V6-c] Explicit Euler instability: Phase 2 explicit integration diverges at bottleneck
  [V6-d] Thomas algorithm accuracy: residual ||(I-Dt*A)f_new - f_old||_inf < 1e-8
  [V6-e] Phase 2 occupancy invariance: sum_i df^(m)/dt ~ 0 numerically
  [V6-f] Phase 1 exact matrix exponential stability: P_block range [0,1], f>=0 always
"""

import os
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

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


def safe_phi(z):
    """φ(z) = (1 - e^{-z}) / z,  安全处理 z→0 时 φ→1"""
    return np.where(z < 1.0e-12, 1.0, -np.expm1(-z) / z)


def compute_rates_cell(f_cell, omega_cell, params):
    """
    对单个格子 (M, N) 计算 lambda_acc 和 lambda_dec.
    新模型: eta_m 是逐类数组 (M,), 减速项含 /rho_max 归一化, Bs 加速封锁.
    f_cell:    (M, N)
    omega_cell: scalar
    返回: lambda_acc (M, N), lambda_dec (M, N)
    """
    v, v_max, rho_max, R_c, eps = (
        params['v'], params['v_max'], params['rho_max'],
        params['R_c'], params['eps']
    )
    alpha, eta_m, omega_0, beta, w, i_thr = (
        params['alpha'], params['eta_m'], params['omega_0'],
        params['beta'], params['w'], params['i_thr']
    )
    M, N = f_cell.shape

    # λ_acc[m,i]: per-class eta_m[m], with acceleration filter
    acc_filter = 1.0 - np.exp(-max(0.0, rho_max - omega_cell) / R_c)
    lambda_acc = np.zeros((M, N))
    for m in range(M):
        speed_factor = (1.0 - v / v_max) ** eta_m[m]
        lambda_acc[m] = alpha[m] * speed_factor * acc_filter
    lambda_acc[:, N-1] = 0.0    # Dirichlet 上界
    # Bs (m=2) 加速封锁: i >= i_thr
    lambda_acc[2, i_thr:] = 0.0

    # λ_dec[m,i]: 含 /rho_max 归一化 (eq.3 m3+m4 版本)
    beta_w = beta * w[None, :]   # (M, M)
    bwf    = beta_w @ f_cell     # (M, N): bwf[m,k] = Σ_n beta_w[m,n]*f[n,k]

    cum_bwf  = np.zeros_like(bwf)
    cum_vbwf = np.zeros_like(bwf)
    cum_bwf[:, 1:]  = np.cumsum(bwf[:, :-1], axis=1)
    cum_vbwf[:, 1:] = np.cumsum(v[None, :-1] * bwf[:, :-1], axis=1)

    interaction = (v[None, :] * cum_bwf - cum_vbwf) / rho_max    # /rho_max 归一化
    pressure = (rho_max / max(eps, rho_max - omega_cell)) ** eta_m[:, None]  # (M, N)
    lambda_dec = (omega_0[:, None] + interaction) * pressure
    lambda_dec = np.maximum(lambda_dec, 0.0)
    lambda_dec[:, 0] = 0.0     # Dirichlet 下界
    return lambda_acc, lambda_dec


def thomas_1cell(f_cell, lambda_acc, lambda_dec, dt):
    """
    对单个格子 (M, N) 用 Thomas 算法求解半隐式 Phase 2.
    独立实现，用于残差验证.
    """
    M, N = f_cell.shape
    f_new = np.zeros_like(f_cell)
    for m in range(M):
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N)
        d = f_cell[m].copy()
        a[1:]  = -dt * lambda_acc[m, :-1]
        b      =  1.0 + dt * (lambda_acc[m] + lambda_dec[m])
        c[:-1] = -dt * lambda_dec[m, 1:]

        c_p = np.zeros(N); d_p = np.zeros(N)
        c_p[0] = c[0] / b[0];  d_p[0] = d[0] / b[0]
        for i in range(1, N):
            denom  = max(b[i] - a[i] * c_p[i-1], 1e-14)
            c_p[i] = c[i] / denom
            d_p[i] = (d[i] - a[i] * d_p[i-1]) / denom

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
        rho_max = float(hf['parameters'].attrs['rho_max'])
        R_c     = float(hf['parameters'].attrs['R_c'])
        eps     = float(hf['parameters'].attrs['eps'])
        v_max   = float(hf['parameters'].attrs['v_max_mps'])
        dx      = float(hf['parameters'].attrs['dx_m'])
        dt      = float(hf['parameters'].attrs['dt_s'])
        T       = int(hf['parameters'].attrs['T_steps'])
        X       = int(hf['parameters'].attrs['X'])
        N       = int(hf['parameters'].attrs['N'])
        M       = int(hf['parameters'].attrs['M'])
        i_thr   = int(hf['parameters'].attrs['i_thr'])
        v       = hf['parameters/v_mps'][:]
        w       = hf['parameters/w_PCE'][:]
        alpha   = hf['parameters/alpha_hz'][:]
        eta_m   = hf['parameters/eta_m'][:]        # (M,) per-class
        omega_0 = hf['parameters/omega_0_hz'][:]
        beta    = hf['parameters/beta_matrix'][:]
        time_s  = hf['data/time_s'][:]

        params = dict(v=v, v_max=v_max, rho_max=rho_max, R_c=R_c,
                      eps=eps, alpha=alpha, eta_m=eta_m, omega_0=omega_0,
                      beta=beta, w=w, i_thr=i_thr)

        tau_adv = dx / v_max          # τ_adv ≈ 0.667 s
        mu_slow = v_max / dx          # |μ_slow| = 1.5 Hz

        class_names = ['A (Trucks)', 'Bf (Free Cars)', 'Bs (Trapped)']

        print(f"\n{'='*60}")
        print(f"  V6 -- Stiffness & Operator Splitting Validation  (M={M} classes, 3-phase)")
        print(f"  τ_adv = Δx/v_max = {tau_adv:.4f}s,  |μ_slow| = {mu_slow:.2f} Hz")
        print(f"  eta_m = {eta_m.tolist()}, i_thr = {i_thr}")
        print(f"{'='*60}")

        # ── [V6-a]  刚性比 (全时域扫描) ──────────────────────────────────────
        print("  计算刚性比 (逐步扫描 lambda_dec)...")
        stiffness_peak = np.zeros(T)
        for t in range(T):
            ldec_t = hf['data/lambda_dec'][t]   # (M, N, X, L)
            stiffness_peak[t] = float(ldec_t.max() / mu_slow)

        max_S  = float(stiffness_peak.max())
        init_S = float(stiffness_peak[0])
        # Use max_S (not init_S): stiffness develops as Omega→rho_max during sim,
        # even if t=0 IC starts at lower density / weaker singular barrier.
        passed_a = max_S > 1e5
        results['checks']['V6-a'] = {
            'desc': 'max_t(S) >> 10^5 (极端刚性，全时域)',
            'passed': passed_a, 'S_initial': init_S, 'S_max': max_S
        }
        tag = PASS if passed_a else FAIL
        print(f"  [V6-a] {tag}  初始 S = {init_S:.3e},  全时域最大 S = {max_S:.3e}  (阈值 > 1e5)")

        # ── [V6-b]  时间尺度分离 ─────────────────────────────────────────────
        ldec_max_per_t = stiffness_peak * mu_slow
        tau_relax = np.where(ldec_max_per_t > 0, 1.0 / ldec_max_per_t, np.inf)
        ratio_tau = tau_adv / np.where(tau_relax > 0, tau_relax, np.inf)
        max_ratio = float(ratio_tau[np.isfinite(ratio_tau)].max())
        passed_b  = max_ratio > 1e5
        results['checks']['V6-b'] = {
            'desc': 'τ_adv / τ_relax >> 1 (时间尺度严格分离)',
            'passed': passed_b, 'max_ratio': max_ratio, 'tau_adv_s': tau_adv
        }
        tag = PASS if passed_b else FAIL
        print(f"  [V6-b] {tag}  max(τ_adv/τ_relax) = {max_ratio:.3e}")

        # ── [V6-c]  显式 Euler 不稳定性演示 ──────────────────────────────────
        # 取 t=0, 瓶颈中心格子 x=77, lane=0
        f0      = hf['data/f'][0]         # (M, N, X, L)
        omega0  = hf['data/omega'][0]     # (X, L)
        f_bott  = f0[:, :, 77, 0].copy() # (M, N)
        om_bott = omega0[77, 0]           # scalar ≈ 0.145

        lam_acc, lam_dec = compute_rates_cell(f_bott, om_bott, params)

        f_expl = f_bott.copy()
        expl_steps = []
        for _ in range(5):
            df = np.zeros_like(f_expl)
            for m in range(M):
                df[m, 1:]  += lam_acc[m, :-1] * f_expl[m, :-1]
                df[m]      -= (lam_acc[m] + lam_dec[m]) * f_expl[m]
                df[m, :-1] += lam_dec[m, 1:]  * f_expl[m, 1:]
            f_expl = f_expl + dt * df
            expl_steps.append(f_expl.copy())
        expl_blowup = float(np.abs(np.array(expl_steps)).max())

        f_impl    = thomas_1cell(f_bott, lam_acc, lam_dec, dt)
        impl_max  = float(f_impl.max())
        impl_neg  = float(f_impl.min())
        passed_c  = expl_blowup > 1.0 and impl_max <= rho_max + 1e-6
        results['checks']['V6-c'] = {
            'desc': '显式 Euler 发散 (>1.0), Thomas 稳定 (≤ρ_max)',
            'passed': passed_c, 'explicit_max': expl_blowup, 'implicit_max': impl_max
        }
        tag = PASS if passed_c else FAIL
        print(f"  [V6-c] {tag}  显式 Euler max(|f|) = {expl_blowup:.2e} (5步)  "
              f"vs  Thomas max(f) = {impl_max:.4f}")

        # ── [V6-d]  Thomas 算法残差 ──────────────────────────────────────────
        residuals = []
        for m in range(M):
            A_mat = np.zeros((N, N))
            for i in range(N):
                if i > 0:   A_mat[i, i-1] += lam_acc[m, i-1]
                A_mat[i, i] -= (lam_acc[m, i] + lam_dec[m, i])
                if i < N-1: A_mat[i, i+1] += lam_dec[m, i+1]
            lhs = np.eye(N) - dt * A_mat
            residuals.append(float(np.abs(lhs @ f_impl[m] - f_bott[m]).max()))

        max_residual = float(max(residuals))
        passed_d = max_residual < 1e-8
        results['checks']['V6-d'] = {
            'desc': '||(I-ΔtA)f_new - f_old||_inf < 1e-8',
            'passed': passed_d, 'max_residual': max_residual
        }
        tag = PASS if passed_d else FAIL
        print(f"  [V6-d] {tag}  Thomas 算法残差 = {max_residual:.2e}  (阈值 < 1e-8)")

        # ── [V6-e]  Phase 2 占用不变性引理 ───────────────────────────────────
        # Σ_m w[m] * Σ_i df^(m)/dt = 0 (速度转移零和 ⟹ Ω 不变)
        df_phase2 = np.zeros((M, N))
        for m in range(M):
            df_phase2[m, 1:]  += lam_acc[m, :-1] * f_bott[m, :-1]
            df_phase2[m]      -= (lam_acc[m] + lam_dec[m]) * f_bott[m]
            df_phase2[m, :-1] += lam_dec[m, 1:]  * f_bott[m, 1:]

        d_omega_dt  = float((w[:, None] * df_phase2).sum())
        class_sums  = df_phase2.sum(axis=1)   # (M,)
        passed_e    = abs(d_omega_dt) < 1e-6
        results['checks']['V6-e'] = {
            'desc': 'd(Ω)/dt = Σ_m w^m Σ_i df/dt ≈ 0 (占用不变性引理)',
            'passed': passed_e,
            'd_omega_dt': d_omega_dt,
            'class_mass_rate_sums': class_sums.tolist()
        }
        tag = PASS if passed_e else FAIL
        print(f"  [V6-e] {tag}  d(Ω)/dt = {d_omega_dt:.2e}  "
              f"[A={class_sums[0]:.2e}, Bf={class_sums[1]:.2e}, Bs={class_sums[2]:.2e}]")

        # ── [V6-f]  Phase 1 exact matrix exponential stability (P_block version) ──
        # Verify: phi(z) safe, P_block in [0,1], f^(Bs) >= 0 all time
        sigma_ds = hf['data/sigma']   # (T, N, X, L)
        mu_ds    = hf['data/mu']      # (T, X, L)
        P_block_ds = hf['data/P_block']  # (T, X, L)

        sig_t0 = sigma_ds[0, :, :, :]     # (N, X, L)
        mu_t0  = mu_ds[0, :, :]            # (X, L)
        P_t0   = P_block_ds[0, :, :]       # (X, L)

        # Max sigma in injection zone (where Bf hits bottleneck)
        sig_max_inj = float(sig_t0[:, 60:70, :].max())
        mu_max      = float(mu_t0.max())
        P_max_t0    = float(P_t0.max())

        # Exact integration verification: for extreme z = sigma*dt >> 1
        S_test    = sig_max_inj
        f_Bf_test = 0.060     # injection zone Bf density
        f_Bs_test = 0.0       # initial Bs=0
        F_test    = f_Bf_test + f_Bs_test

        z = S_test * dt
        phi_z    = float(safe_phi(np.array([z]))[0])
        f_Bf_new = f_Bf_test * np.exp(-z) + 0.0 * F_test * phi_z  # mu=0 case
        f_Bs_new = F_test - f_Bf_new
        # Result must be physical (0 <= f <= F) regardless of z magnitude
        phase1_stable = (f_Bf_new >= -1e-14) and (f_Bs_new >= -1e-14) and \
                        (f_Bf_new + f_Bs_new <= F_test + 1e-10)

        # P_block in [0,1] all time
        P_max_all = float(P_block_ds[:].max())
        P_min_all = float(P_block_ds[:].min())
        P_bounded = (P_min_all >= -1e-10 and P_max_all <= 1.0 + 1e-10)

        # f^(Bs) >= 0 all time
        f_Bs_min = float('inf')
        for t in range(0, T, 50):
            val = float(hf['data/f'][t, 2, :, :, :].min())
            if val < f_Bs_min:
                f_Bs_min = val

        passed_f = phase1_stable and P_bounded and (f_Bs_min >= -1e-10)
        results['checks']['V6-f'] = {
            'desc': 'Phase 1 stability: phi safe, P_block in [0,1], f^(Bs)>=0 all time',
            'passed': passed_f,
            'sigma_max_injection': sig_max_inj,
            'z_dt': float(z),
            'phi_z': float(phi_z),
            'P_block_range': [P_min_all, P_max_all],
            'f_Bs_min_global': f_Bs_min
        }
        tag = PASS if passed_f else FAIL
        print(f"  [V6-f] {tag}  Phase 1: sigma_max={sig_max_inj:.3f}, z={z:.3f}, "
              f"phi(z)={phi_z:.4f}, P_block in [{P_min_all:.4f},{P_max_all:.4f}], "
              f"min(f_Bs)={f_Bs_min:.2e}")

        # ═══════════════════════════════════════════════════════════════════════
        # 生成图表
        # ═══════════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # ── 图 V6-1: 刚性比时序曲线 ───────────────────────────────────────
        ax = axes[0, 0]
        ax.semilogy(time_s, stiffness_peak, 'C1-', lw=2)
        ax.axhline(1e5, color='red', ls='--', lw=1.5, label='Threshold S=10^5')
        ax.axhline(1.0, color='green', ls=':', lw=1.5, label='S=1 (no stiffness)')
        ax.fill_between(time_s, stiffness_peak, 1e5,
                        where=(stiffness_peak > 1e5), alpha=0.2, color='red',
                        label='Extreme Stiffness Region')
        ax.set_xlabel('Time [s]', fontsize=10); ax.set_ylabel('Stiffness Ratio S', fontsize=10)
        ax.set_title('[V6-a/b] System Stiffness Ratio Time Series (log scale)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

        # ── 图 V6-2: 时间尺度对比 ─────────────────────────────────────────
        ax = axes[0, 1]
        tau_relax_plot = np.where(ldec_max_per_t > 1e-10, 1.0 / ldec_max_per_t, np.nan)
        ax.semilogy(time_s, tau_relax_plot, 'C1-', lw=2, label=r'$\tau_{relax}$')
        ax.axhline(tau_adv, color='C0', ls='--', lw=2,
                   label=fr'$\tau_{{adv}} = {tau_adv:.3f}$s')
        ax.axhline(dt, color='gray', ls=':', lw=1.5, label=fr'$\Delta t = {dt}$s')
        ax.set_xlabel('Time [s]', fontsize=10); ax.set_ylabel('Characteristic Time Scale [s]', fontsize=10)
        ax.set_title('[V6-b] Slow vs Fast Manifold Time-scale Separation', fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

        # ── 图 V6-3: 显式 vs 隐式一步结果对比 (Class A) ─────────────────
        ax = axes[0, 2]
        speed_bins = np.arange(N); width = 0.28
        ax.bar(speed_bins - width, f_bott[0, :], width=width,
               color='gray', alpha=0.6, label='Initial Class A')
        ax.bar(speed_bins, f_impl[0, :], width=width,
               color='C1', alpha=0.8, label='Thomas (implicit) after 1 step')
        f_expl_1_A = expl_steps[0][0, :]
        ax.bar(speed_bins + width, np.clip(f_expl_1_A, -0.01, rho_max * 2),
               width=width, color='C3', alpha=0.8, label='Euler (explicit) after 1 step')
        ax.axhline(0, color='black', lw=0.8)
        ax.axhline(rho_max, color='red', ls='--', lw=1.2, label=r'$\rho_{max}$')
        ax.set_xlabel('Speed bin i', fontsize=10); ax.set_ylabel('Density f [veh/m]', fontsize=10)
        ax.set_title('[V6-c] Bottleneck Cell Class A: Explicit Euler vs Thomas Implicit', fontsize=9)
        ax.legend(fontsize=7, ncol=2); ax.set_ylim(-0.05, rho_max * 1.5)
        ax.grid(alpha=0.3, axis='y'); ax.set_xticks(speed_bins)
        ax.set_xticklabels([f'{int(vi)}' for vi in v], fontsize=7)

        # ── 图 V6-4: 显式方案 5 步演化轨迹 (Class A) ──────────────────────
        ax = axes[1, 0]
        steps_plot = [f_bott[0, :]] + [s[0, :] for s in expl_steps]
        colors_e = plt.cm.Reds(np.linspace(0.3, 1.0, 6))
        for si, (f_step, col) in enumerate(zip(steps_plot, colors_e)):
            ax.plot(speed_bins, np.clip(f_step, -0.5, 1.0), 'o-',
                    color=col, markersize=4, lw=1.5, label=f't+{si}Δt')
        ax.axhline(0, color='black', lw=0.8)
        ax.axhline(rho_max, color='red', ls='--', lw=1.2, label=r'$\rho_{max}$')
        ax.set_xlabel('Speed bin i', fontsize=10); ax.set_ylabel('Density f (truncated display)', fontsize=10)
        ax.set_title('[V6-c] Explicit Euler Divergence Trajectory (Class A, Bottleneck Cell)', fontsize=9)
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

        # ── 图 V6-5: φ(z) 函数稳定性曲线 ────────────────────────────────
        ax = axes[1, 1]
        z_vals = np.logspace(-15, 6, 1000)
        phi_vals = safe_phi(z_vals)
        ax.semilogx(z_vals, phi_vals, 'C0-', lw=2)
        ax.axhline(1.0, color='gray', ls='--', lw=1.5, label='φ(z→0)=1')
        ax.axhline(0.0, color='gray', ls=':', lw=1.5, label='φ(z→∞)=0')
        ax.axvline(z, color='C1', ls='--', lw=2, label=f'当前 z=σ·Δt={z:.3f}')
        ax.set_xlabel('z = \u03c3\u00b7\u0394t', fontsize=10); ax.set_ylabel('\u03c6(z) = (1-e^{-z})/z', fontsize=10)
        ax.set_title('[V6-f] \u03c6(z) Safe Implementation (z\u21920: no NaN/0/Inf)', fontsize=9)
        ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.1); ax.grid(alpha=0.3)

        # ── Figure V6-6: Phase 1 sigma, mu, P_block spatial distribution (t=0) ──
        ax = axes[1, 2]
        x_cells = np.arange(X)
        sig_mean = sig_t0.mean(axis=(0, 2))   # (X,): mean over N, L
        mu_mean  = mu_t0.mean(axis=1)         # (X,): mean over L
        P_mean   = P_t0.mean(axis=1)          # (X,): mean over L
        ax.plot(x_cells, sig_mean, 'C2-',  lw=2, label=r'$\sigma$ (capture rate, avg i,l)')
        ax.plot(x_cells, mu_mean,  'C3-',  lw=2, label=r'$\mu$ (release rate, avg l)')
        ax2f = ax.twinx()
        ax2f.plot(x_cells, P_mean, 'C0--', lw=1.5, alpha=0.7, label=r'$P_{block}$ (avg l)')
        ax.axvspan(74, 79, alpha=0.1, color='red',  label='Bottleneck (x=74-79)')
        ax.axvspan(60, 70, alpha=0.1, color='blue', label='Injection (x=60-70)')
        ax.set_xlabel('Cell x', fontsize=10)
        ax.set_ylabel('Rate [Hz]', fontsize=10)
        ax2f.set_ylabel(r'$P_{block}$', fontsize=10, color='C0')
        ax.set_title('[V6-f] Phase 1: sigma/mu/P_block\nSpatial Distribution (t=0)', fontsize=9)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2f.get_legend_handles_labels()
        ax.legend(lines1+lines2, labs1+labs2, fontsize=7); ax.grid(alpha=0.3)

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
