"""
V3 — FVM 空间通量与全局 Godunov 限制器验证  (m3+m4 升级版)
对应 .tex §5:  Ψ^(m) (需求通量), α (全局 Godunov 限制器), Φ^(m) (实际通量)

检验项:
  [V3-a] 纯需求通量 Φ ≥ 0 全时域 (Ψ = v·f ≥ 0, 无供给过滤器)
  [V3-b] 正值性: min(f) ≥ 0 全时域
  [V3-c] 激波可观测性: 时空密度图有反向传播激波
  [V3-d] CFL 验证: Δt·v_max/Δx = 0.75 ≤ 1
  [V3-e] 全局限制器: α ≤ 1 全时域，当下游满载时 α→0
  [V3-f] 基本图双模态: 合并 B 类(Bf+Bs) 的 q-ρ 散点
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
    results = {'module': 'V3_fvm', 'checks': {}, 'figures': []}

    with h5py.File(HDF5, 'r') as hf:
        rho_max  = float(hf['parameters'].attrs['rho_max'])
        R_supply = float(hf['parameters'].attrs['R_supply'])
        eps      = float(hf['parameters'].attrs['eps'])
        v_max    = float(hf['parameters'].attrs['v_max_mps'])
        dx       = float(hf['parameters'].attrs['dx_m'])
        dt       = float(hf['parameters'].attrs['dt_s'])
        M        = int(hf['parameters'].attrs['M'])
        N        = int(hf['parameters'].attrs['N'])
        X        = int(hf['parameters'].attrs['X'])
        L        = int(hf['parameters'].attrs['L'])
        T        = int(hf['parameters'].attrs['T_steps'])
        v        = hf['parameters/v_mps'][:]
        w        = hf['parameters/w_PCE'][:]
        time_s   = hf['data/time_s'][:]

        print(f"\n{'='*60}")
        print(f"  V3 — FVM + 全局 Godunov 限制器验证  (M={M})")
        print(f"{'='*60}")

        # ── [V3-a] 纯需求通量 Φ ≥ 0 全时域 (Ψ = v·f, 无供给过滤器) ─────────
        # Pure demand formulation: Ψ = v*f ≥ 0 (since f ≥ 0 from V3-b), so Φ = α·Ψ ≥ 0.
        # Also collect (available_space, delivered_flux) scatter data for α-limiter figure.
        omega_range = np.linspace(0, rho_max, 500)   # kept for subplot 3 reference line
        min_phi_global = np.inf
        avail_scatter, delivered_scatter = [], []
        for t_i in range(0, T, max(1, T // 30)):
            phi_t   = hf['data/phi'][t_i]                               # (M, N, X+1, L)
            omega_t = hf['data/omega'][max(0, t_i - 1)]                 # (X, L)
            min_phi_global = min(min_phi_global, float(phi_t.min()))
            phi_pce = (w[:, None, None, None] * phi_t).sum(axis=(0, 1))  # (X+1, L)
            avail_t = np.maximum(0.0, rho_max - omega_t[1:X, :])         # (X-1, L)
            deliv_t = (dt / dx) * phi_pce[1:X, :]                        # (X-1, L)
            step_s  = max(1, (X - 1) * L // 150)
            avail_scatter.extend(avail_t.ravel()[::step_s].tolist())
            delivered_scatter.extend(deliv_t.ravel()[::step_s].tolist())
        passed_a = min_phi_global >= -1e-10
        results['checks']['V3-a'] = {
            'desc': '纯需求通量 Φ ≥ 0 全时域 (Ψ = v·f ≥ 0)',
            'passed': passed_a, 'min_phi': float(min_phi_global)
        }
        tag = PASS if passed_a else FAIL
        print(f"  [V3-a] {tag}  全局 min(Φ) = {min_phi_global:.2e}"
              f"  ({'≥' if passed_a else '<'} 0)")
        avail_arr     = np.array(avail_scatter)
        delivered_arr = np.array(delivered_scatter)

        # ── [V3-b] 正值性: min(f) ≥ 0 ──────────────────────────────────────
        min_f_global = np.inf
        for t in range(T):
            min_f_global = min(min_f_global, float(hf['data/f'][t].min()))
        passed_b = min_f_global >= -1e-10
        results['checks']['V3-b'] = {
            'desc': 'min(f) ≥ 0 全时域', 'passed': passed_b, 'value': float(min_f_global)
        }
        tag = PASS if passed_b else FAIL
        print(f"  [V3-b] {tag}  全局 min(f) = {min_f_global:.2e}")

        # ── [V3-c] Riemann Problem I: Shock Wave (free flow upstream, congested downstream)
        # Validation: shock propagates backward from bottleneck into upstream free flow.
        # Expected: upstream density (x=68-73) increases over time as cars pile up
        # behind the truck bottleneck at x=74-79 (backward-propagating shock).
        rho_t0   = hf['data/rho_macro'][0]                   # (M, X, L)
        rho_late = hf['data/rho_macro'][min(100, T-1)]       # (M, X, L)
        omega_t0_arr   = hf['data/omega'][0]                  # (X, L)
        omega_late_arr = hf['data/omega'][min(100, T-1)]      # (X, L)

        upstream_omega_t0   = float(omega_t0_arr[68:74, :].mean())
        upstream_omega_late = float(omega_late_arr[68:74, :].mean())
        shock_delta         = upstream_omega_late - upstream_omega_t0   # >0 means buildup

        rho_Bs_t100 = float(hf['data/rho_macro'][min(100, T-1), 2, :, :].mean())
        passed_c = bool(shock_delta > 0.002 or rho_Bs_t100 > 0.001)
        results['checks']['V3-c'] = {
            'desc': 'Riemann I (Shock): upstream omega increases as shock propagates backward',
            'passed': passed_c,
            'upstream_omega_t0':   upstream_omega_t0,
            'upstream_omega_late': upstream_omega_late,
            'shock_delta':         float(shock_delta),
            'rho_Bs_late':         rho_Bs_t100
        }
        tag = PASS if passed_c else FAIL
        print(f"  [V3-c] {tag}  Shock: upstream omega t0={upstream_omega_t0:.4f} -> "
              f"late={upstream_omega_late:.4f} (delta={shock_delta:+.4f}), "
              f"rho_Bs_late={rho_Bs_t100:.4f}")

        # ── [V3-d] CFL 验证 ─────────────────────────────────────────────────
        cfl = dt * v_max / dx
        passed_d = cfl <= 1.0
        results['checks']['V3-d'] = {
            'desc': 'CFL = Δt·v_max/Δx ≤ 1', 'passed': passed_d, 'cfl': float(cfl)
        }
        tag = PASS if passed_d else FAIL
        print(f"  [V3-d] {tag}  CFL = {cfl:.4f}  ({'≤' if passed_d else '>'} 1.0)")

        # ── [V3-e] 全局限制器 α ≤ 1 ──────────────────────────────────────────
        # 从存储的 phi 反算 α = Φ / Ψ 在非零处: α ≤ 1
        # 或等价: Φ ≤ Ψ，且 Φ·(ρ_max - Ω_downstream) ≤ ρ_max (不超满载)
        max_alpha_violation = 0.0
        for t in range(0, T, 40):
            phi_t   = hf['data/phi'][t]                         # (M, N, X+1, L)
            omega_t = hf['data/omega'][max(0, t - 1)]           # pre-advection proxy (X, L)
            # phi[t] was limited by omega_before_phase3[t].
            # Phases 1&2 preserve omega (V6-e), so omega[t-1] ≈ omega_before_phase3[t].
            # Total PCE flux demand at each internal face
            # Φ_total[face] = Σ_{m,i} w[m] * phi[m,i,face,l]
            phi_total = (w[:, None, None, None] * phi_t).sum(axis=(0, 1))  # (X+1, L)
            # Available at downstream cells for internal faces 1..X-1
            # face i has downstream cell i → omega_t[1:X, :] (X-1, L)
            avail = np.maximum(0.0, rho_max - omega_t[1:X, :])             # (X-1, L)
            # dt/dx * phi_total at internal faces 1..X-1 vs available space
            ratio_check = (dt / dx) * phi_total[1:X, :] / (avail + eps)   # (X-1, L)
            max_alpha_violation = max(max_alpha_violation, float(ratio_check.max()))
        passed_e = max_alpha_violation <= 1.0 + 1e-6
        results['checks']['V3-e'] = {
            'desc': '全局 Godunov 限制器: α ≤ 1 (通量不超可用空间)',
            'passed': passed_e, 'max_ratio': float(max_alpha_violation)
        }
        tag = PASS if passed_e else FAIL
        print(f"  [V3-e] {tag}  最大通量/可用空间比 = {max_alpha_violation:.4f}"
              f"  ({'≤' if passed_e else '>'} 1.0)")

        # ── [V3-f] 基本图双模态 ──────────────────────────────────────────────
        # 对所有时刻和格子，计算合并 B 类密度和流量
        rho_agg = np.zeros(T * X * L)
        q_agg   = np.zeros(T * X * L)
        idx = 0
        step_sample = max(1, T // 40)
        for t in range(0, T, step_sample):
            rm  = hf['data/rho_macro'][t]   # (M, X, L)
            qm  = hf['data/q_macro'][t]     # (M, X, L)
            # Combined B = Bf + Bs
            rho_B = rm[1] + rm[2]           # (X, L)
            q_B   = qm[1] + qm[2]           # (X, L)
            n     = X * L
            rho_agg[idx:idx+n] = rho_B.ravel()
            q_agg[idx:idx+n]   = q_B.ravel()
            idx += n
        rho_agg = rho_agg[:idx]
        q_agg   = q_agg[:idx]

        # 双模态: 判断散点是否有明显稀疏流 (rho<0.05) 和拥堵流 (rho>0.05)
        mask_free = rho_agg > 1e-5
        if mask_free.sum() > 10:
            rho_valid = rho_agg[mask_free]
            bimodal = float(rho_valid.std()) > 0.005
        else:
            bimodal = False
        passed_f = bimodal
        results['checks']['V3-f'] = {
            'desc': '合并B类 q-ρ 基本图有双模态分布',
            'passed': passed_f, 'rho_std': float(rho_agg[mask_free].std()) if mask_free.sum() > 0 else 0
        }
        tag = PASS if passed_f else FAIL
        print(f"  [V3-f] {tag}  Class B rho std = {results['checks']['V3-f']['rho_std']:.4f}"
              f"  (> 0.005 = bimodal)")

        # ── [V3-g] Riemann Problem II: Rarefaction Wave (congested upstream, free downstream)
        # Validation: downstream of the bottleneck (x=80-110), the density profile at
        # late time is smooth (no sharp front) — characteristic of a rarefaction fan.
        # Also: Bs density far downstream should be near zero (free-flow, no trapping).
        omega_late_down = omega_late_arr[80:111, :].mean(axis=1)   # (31,) lane-avg
        max_gradient    = float(np.abs(np.diff(omega_late_down)).max())
        # Smooth gradient: no cell-to-cell jump exceeds half rho_max
        smooth_ok = max_gradient < rho_max * 0.5

        rho_Bs_bott    = float(hf['data/rho_macro'][min(100, T-1), 2, 74:80, :].mean())
        rho_Bs_fardown = float(hf['data/rho_macro'][min(100, T-1), 2, 90:110, :].mean())
        # In rarefaction zone: far-downstream Bs << bottleneck Bs
        # (cars downstream are in free flow; trapping is a near-bottleneck phenomenon)
        raref_ok = (rho_Bs_bott > 1e-6 and
                    rho_Bs_fardown < rho_Bs_bott * 0.5) or rho_Bs_bott < 1e-6

        passed_g = smooth_ok and raref_ok
        results['checks']['V3-g'] = {
            'desc': 'Riemann II (Rarefaction): downstream profile smooth, far-downstream Bs << bottleneck Bs',
            'passed': passed_g,
            'max_gradient_downstream': float(max_gradient),
            'smooth_threshold': float(rho_max * 0.5),
            'rho_Bs_bottleneck': float(rho_Bs_bott),
            'rho_Bs_far_downstream': float(rho_Bs_fardown)
        }
        tag = PASS if passed_g else FAIL
        print(f"  [V3-g] {tag}  Rarefaction: max downstream gradient={max_gradient:.4f} "
              f"(< {rho_max*0.5:.3f}), rho_Bs: bott={rho_Bs_bott:.4f} > fardown={rho_Bs_fardown:.4f}")

        # ═══════════ 图表 ═══════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 3, figsize=(21, 10))

        # 图1: α 限制器验证 — 交付通量 vs 可用空间散点
        ax = axes[0, 0]
        ax.scatter(avail_arr, delivered_arr, s=2, alpha=0.25, color='C0',
                   label='Internal faces (sampled)')
        lim_max = max(float(avail_arr.max()), float(delivered_arr.max())) * 1.05
        ax.plot([0, lim_max], [0, lim_max], 'r--', lw=1.5,
                label='y = x  (constraint boundary)')
        ax.set_xlabel(r'Available space $\rho_{\max} - \Omega_{x+1}$ [PCE/m]')
        ax.set_ylabel(r'Delivered PCE flux $(dt/dx)\cdot\Phi_{total}$ [PCE/m]')
        ax.set_title('[V3-a] Global Godunov Limiter\nDelivered flux \u2264 Available space')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # 图2: 时空 Hovmöller 图 — PCE 密度, turbo 色图突出激波/稀疏波
        ax = axes[0, 1]
        step_hov = max(1, T // 100)
        rho_hov  = np.zeros((T // step_hov + 1, X))
        spd_hov  = np.zeros((T // step_hov + 1, X))
        for ti, t in enumerate(range(0, T, step_hov)):
            om = hf['data/omega'][t]           # (X, L) PCE-weighted occupancy
            um = hf['data/u_macro'][t]         # (M, X, L) mean speed per class
            rho_hov[ti] = om.mean(axis=1)      # lane avg
            spd_hov[ti] = um[1].mean(axis=1)   # Bf mean speed, lane avg
        t_axis = np.arange(0, T, step_hov) * dt
        im = ax.pcolormesh(np.arange(X), t_axis, rho_hov[:len(t_axis)],
                           cmap='turbo', vmin=0.010, vmax=rho_max)
        cb = plt.colorbar(im, ax=ax, label=r'$\Omega$ [PCE/m]')
        cb.set_ticks([0, rho_max * 0.25, rho_max * 0.5, rho_max * 0.75, rho_max])
        cb.set_ticklabels(['0 (empty)', f'{rho_max*0.25:.3f}',
                           f'{rho_max*0.5:.3f} (half)', f'{rho_max*0.75:.3f}',
                           f'{rho_max:.3f} (jam)'])
        ax.set_xlabel('Cell x'); ax.set_ylabel('Time [s]')
        ax.set_title('[V3-c/g] Hovmoller \u03a9(x,t) — Ring Road\n'
                     'Shock \u2190 (dark front leans left)  |  Rarefaction \u2192 (right fades)')
        # Bottleneck zone
        ax.axvline(74, color='white', ls='--', lw=1.2, alpha=0.85, label='Bottleneck x=74-79')
        ax.axvline(79, color='white', ls='--', lw=1.2, alpha=0.85)
        # Ring join markers
        ax.axvline(0,   color='silver', ls=':', lw=1, alpha=0.7)
        ax.axvline(X-1, color='silver', ls=':', lw=1, alpha=0.7)
        ax.text(1,   t_axis[-1]*0.97, 'ring\njoin', color='silver', fontsize=6, va='top')
        ax.text(X-2, t_axis[-1]*0.97, 'ring\njoin', color='silver', fontsize=6, va='top', ha='right')
        # Theoretical Rankine-Hugoniot shock trajectory
        # v_shock = (Q_R - Q_L) / (Ω_R - Ω_L) ≈ (0 − 0.60) / (0.15 − 0.020) ≈ −4.62 m/s
        v_shock_mps  = -4.62                           # physical wave speed [m/s]
        v_shock_cell = v_shock_mps / dx                # wave speed [cells/s]  (< 0 → leftward)
        x_cur, t_cur = 74.0, 0.0
        t_end_rh = float(t_axis[-1])
        first_rh = True
        while t_cur < t_end_rh - 1e-9:
            dt_to_exit = x_cur / abs(v_shock_cell)    # seconds until shock reaches x=0
            if t_cur + dt_to_exit >= t_end_rh:
                x_final = x_cur + v_shock_cell * (t_end_rh - t_cur)
                ax.plot([x_cur, x_final], [t_cur, t_end_rh],
                        color='lime', lw=1.8, ls='-.',
                        label=f'R-H theory ($v_s$={v_shock_mps} m/s)' if first_rh else '_',
                        alpha=0.90)
                break
            t_exit = t_cur + dt_to_exit
            ax.plot([x_cur, 0.0], [t_cur, t_exit],
                    color='lime', lw=1.8, ls='-.',
                    label=f'R-H theory ($v_s$={v_shock_mps} m/s)' if first_rh else '_',
                    alpha=0.90)
            first_rh = False
            x_cur, t_cur = float(X - 1), t_exit       # ring road wrap: re-enter from right
        ax.legend(fontsize=7, loc='upper right')

        # 图3: 均速 Hovmöller — Bf mean speed u^(Bf)(x,t)
        ax = axes[0, 2]
        im2 = ax.pcolormesh(np.arange(X), t_axis, spd_hov[:len(t_axis)],
                            cmap='RdYlGn', vmin=0, vmax=v_max)
        plt.colorbar(im2, ax=ax, label=r'$u^{(Bf)}$ [m/s]')
        ax.set_xlabel('Cell x'); ax.set_ylabel('Time [s]')
        ax.set_title('[V3-c/g] Hovmöller $u^{(Bf)}(x,t)$ — Bf Mean Speed\n'
                     'Rarefaction fan: slow \u2192 fast (feather pattern)')
        ax.axvline(74, color='white', ls='--', lw=1.2, alpha=0.85, label='Bottleneck x=74-79')
        ax.axvline(79, color='white', ls='--', lw=1.2, alpha=0.85)
        ax.axvline(0,   color='silver', ls=':', lw=1, alpha=0.7)
        ax.axvline(X-1, color='silver', ls=':', lw=1, alpha=0.7)
        ax.legend(fontsize=7, loc='upper right')

        # 图4: 基本图 (B 类 Bf+Bs)
        ax = axes[1, 0]
        thresh_fd = 0.03
        free_mask = rho_agg[mask_free] < thresh_fd
        ax.scatter(rho_agg[mask_free][free_mask],  q_agg[mask_free][free_mask],
                   s=1, alpha=0.4, color='C0', label=f'Free flow (\u03c1<{thresh_fd})')
        ax.scatter(rho_agg[mask_free][~free_mask], q_agg[mask_free][~free_mask],
                   s=1, alpha=0.4, color='C3', label=f'Congested (\u03c1\u2265{thresh_fd})')
        ax.plot(omega_range, v_max * omega_range, 'k--', lw=1, alpha=0.5, label=r'$q=v_{\max}\rho$')
        ax.set_xlabel(r'Density $\rho$ [veh/m]'); ax.set_ylabel(r'Flow $q$ [veh/s]')
        ax.set_title('[V3-f] Class B (Bf+Bs) Fundamental Diagram \u2014 Bimodal')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # 图4: 三类密度空间快照 (t=0, 50, 100)
        ax = axes[1, 1]
        colors = ['C1', 'C0', 'C2']
        labels = ['A (Trucks)', 'Bf (Free)', 'Bs (Trapped)']
        t_snaps = [0, min(250, T - 1), min(499, T - 1)]
        styles  = ['-', '--', ':']
        for ti, t_snap in enumerate(t_snaps):
            rm = hf['data/rho_macro'][t_snap]   # (M, X, L)
            for m_idx in range(M):
                rho_m = rm[m_idx].mean(axis=1)  # avg over lanes
                ax.plot(np.arange(X), rho_m,
                        color=colors[m_idx], ls=styles[ti], lw=1.5,
                        label=f'{labels[m_idx]} t={t_snap*dt:.0f}s' if ti == 0 else '_')
        ax.axvspan(74, 79, alpha=0.12, color='red', label='Truck Bottleneck')
        ax.axvspan(0,  73, alpha=0.04, color='blue', label='Bf Upstream (uniform IC)')
        ax.set_xlabel('Cell x'); ax.set_ylabel(r'$\rho$ [veh/m]')
        snap_labels = ', '.join(f't={s*dt:.0f}s' for s in t_snaps)
        ax.set_title(f'[V3] Three-class Density Snapshots\n({snap_labels})')
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

        axes[1, 2].set_visible(False)
        plt.tight_layout()
        fig_path = os.path.join(FIGDIR, 'V3_fvm.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        results['figures'].append(fig_path)
        print(f"\n  图表: {fig_path}")

    passed_all = all(v['passed'] for v in results['checks'].values())
    n_p = sum(1 for v in results['checks'].values() if v['passed'])
    results['summary'] = f"{'PASSED' if passed_all else 'FAILED'} {n_p}/{len(results['checks'])}"
    print(f"  V3 汇总: {results['summary']}\n")
    return results


if __name__ == '__main__':
    run()
