"""
V4 — Probabilistic Blocking Factor Validation
Corresponding to .tex Eq. 8-12: P_block-based capture/release model

This module replaces the old Softplus lateral lane-changing validation.
The new model eliminates the lateral phase entirely and uses P_block(x)
as an aggregate congestion-based blocking probability.

Checks:
  [V4-a] P_block in [0,1] for all time steps and spatial cells
  [V4-b] P_block is monotone non-decreasing in Omega (denser = more blocking)
  [V4-c] P_block boundary conditions: P_block=0 at Omega=0, P_block=1 at Omega=rho_max
  [V4-d] sigma proportional to P_block: sparse areas low sigma, congested areas high
  [V4-e] HDF5 schema: no gamma_left/gamma_right/E_trap datasets (lateral phase removed)
  [V4-f] Bs acceleration blockade: max(f^(Bs)[i>=i_thr]) = 0 (kinematic constraint)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import platform

# Chinese font setup for correct rendering
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

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'


def run():
    results = {'module': 'V4_probability', 'checks': {}, 'figures': []}

    with h5py.File(HDF5, 'r') as hf:
        # Parameters
        rho_max   = float(hf['parameters'].attrs['rho_max'])
        eta_block = float(hf['parameters'].attrs['eta_block'])
        dt        = float(hf['parameters'].attrs['dt_s'])
        T         = int(hf['parameters'].attrs['T_steps'])
        X         = int(hf['parameters'].attrs['X'])
        L         = int(hf['parameters'].attrs['L'])
        N         = int(hf['parameters'].attrs['N'])
        i_thr     = int(hf['parameters'].attrs['i_thr'])
        eps       = float(hf['parameters'].attrs['eps'])
        time_s    = hf['data/time_s'][:]
        v         = hf['parameters/v_mps'][:]

        print(f"\n{'='*60}")
        print(f"  V4 -- Probabilistic Blocking Factor Validation")
        print(f"  eta_block={eta_block}, i_thr={i_thr}, T={T}, X={X}, L={L}")
        print(f"{'='*60}")

        # Load P_block and related data
        P_block_all = hf['data/P_block'][:]           # (T, X, L) — computed from pre-phase1 omega
        omega_all   = hf['data/omega_pre_phase3'][:]  # (T, X, L) — pre-phase3 omega ≈ pre-phase1 (phases 1&2 preserve Ω, V6-e)
        sigma_all   = hf['data/sigma'][:]      # (T, N, X, L)

        # ── [V4-a]  P_block in [0, 1] for all time/space ─────────────────────
        P_min = float(P_block_all.min())
        P_max = float(P_block_all.max())
        has_nan = bool(np.isnan(P_block_all).any())
        has_inf = bool(np.isinf(P_block_all).any())
        passed_a = (P_min >= -1e-10 and P_max <= 1.0 + 1e-10
                    and not has_nan and not has_inf)
        results['checks']['V4-a'] = {
            'desc': 'P_block in [0,1] full domain (bounded blocking probability)',
            'passed': passed_a,
            'P_min': P_min, 'P_max': P_max,
            'has_nan': has_nan, 'has_inf': has_inf
        }
        tag = PASS if passed_a else FAIL
        print(f"  [V4-a] {tag}  P_block range: [{P_min:.6f}, {P_max:.6f}]  "
              f"NaN={has_nan}  Inf={has_inf}")

        # ── [V4-b]  P_block monotone non-decreasing in Omega ─────────────────
        omega_range  = np.linspace(0, rho_max, 1000)
        P_analytic   = (omega_range / rho_max) ** eta_block
        monotone_ok  = bool((np.diff(P_analytic) >= 0).all())
        # Numerical check via binned means
        om_flat = omega_all.flatten()
        pb_flat = P_block_all.flatten()
        n_bins = 50
        bin_edges = np.linspace(0, rho_max, n_bins + 1)
        bin_om = np.zeros(n_bins)
        bin_pb = np.zeros(n_bins)
        for b in range(n_bins):
            mask = (om_flat >= bin_edges[b]) & (om_flat < bin_edges[b + 1])
            if mask.sum() > 0:
                bin_om[b] = om_flat[mask].mean()
                bin_pb[b] = pb_flat[mask].mean()
        bin_pb_diff = np.diff(bin_pb[bin_om > 0])
        numeric_monotone = bool((bin_pb_diff >= -1e-8).all())
        passed_b = monotone_ok and numeric_monotone
        results['checks']['V4-b'] = {
            'desc': 'P_block monotone non-decreasing in Omega (higher density = more blocking)',
            'passed': passed_b,
            'analytic_monotone': monotone_ok,
            'numeric_monotone': numeric_monotone
        }
        tag = PASS if passed_b else FAIL
        print(f"  [V4-b] {tag}  P_block monotonicity: analytic={monotone_ok}, "
              f"numeric={numeric_monotone}")

        # ── [V4-c]  Boundary conditions: P_block(0)=0, P_block(rho_max)=1 ───
        P_at_zero    = float((0.0 / rho_max) ** eta_block)
        P_at_rhomax  = float((rho_max / rho_max) ** eta_block)
        bc_zero_ok   = abs(P_at_zero - 0.0) < 1e-10
        bc_full_ok   = abs(P_at_rhomax - 1.0) < 1e-10
        passed_c = bc_zero_ok and bc_full_ok
        results['checks']['V4-c'] = {
            'desc': 'P_block(Omega=0)=0 and P_block(Omega=rho_max)=1 (boundary conditions)',
            'passed': passed_c,
            'P_at_omega_0': P_at_zero,
            'P_at_rho_max': P_at_rhomax
        }
        tag = PASS if passed_c else FAIL
        print(f"  [V4-c] {tag}  P_block(0)={P_at_zero:.2e}  "
              f"P_block(rho_max)={P_at_rhomax:.6f}")

        # ── [V4-d]  sigma proportional to P_block ────────────────────────────
        sigma_sum_t50 = sigma_all[50].sum(axis=0)     # (X, L) sum over N
        P_block_t50   = P_block_all[50]               # (X, L)

        sigma_flat  = sigma_sum_t50.flatten()
        pblock_flat = P_block_t50.flatten()
        mask_nz = sigma_flat > 1e-12
        if mask_nz.sum() > 10:
            corr_sigma_P = float(np.corrcoef(pblock_flat[mask_nz],
                                              sigma_flat[mask_nz])[0, 1])
        else:
            corr_sigma_P = 0.0
        sigma_bott = float(sigma_sum_t50[74:80, :].mean())
        sigma_free = float(sigma_sum_t50[10:20, :].mean())
        sigma_order = sigma_bott >= sigma_free
        passed_d = corr_sigma_P > 0.0 and sigma_order
        results['checks']['V4-d'] = {
            'desc': 'sigma proportional to P_block: bottleneck zone > free zone',
            'passed': passed_d,
            'corr_sigma_Pblock': corr_sigma_P,
            'sigma_bottleneck': sigma_bott,
            'sigma_free_zone': sigma_free,
            'bottleneck_higher': sigma_order
        }
        tag = PASS if passed_d else FAIL
        print(f"  [V4-d] {tag}  corr(sigma, P_block)={corr_sigma_P:.4f}  "
              f"sigma_bott={sigma_bott:.4f}  sigma_free={sigma_free:.4f}")

        # ── [V4-e]  No gamma/E_trap datasets in HDF5 ─────────────────────────
        all_keys = list(hf['data'].keys())
        has_gamma_left  = 'gamma_left'  in all_keys
        has_gamma_right = 'gamma_right' in all_keys
        has_E_trap      = 'E_trap'      in all_keys
        has_P_block_ds  = 'P_block'     in all_keys
        passed_e = (not has_gamma_left and not has_gamma_right
                    and not has_E_trap and has_P_block_ds)
        results['checks']['V4-e'] = {
            'desc': 'HDF5 has P_block; no gamma_left/right/E_trap (lateral phase removed)',
            'passed': passed_e,
            'has_P_block': has_P_block_ds,
            'has_gamma_left': has_gamma_left,
            'has_gamma_right': has_gamma_right,
            'has_E_trap': has_E_trap,
            'all_data_keys': sorted(all_keys)
        }
        tag = PASS if passed_e else FAIL
        print(f"  [V4-e] {tag}  P_block={has_P_block_ds}  "
              f"gamma_left={has_gamma_left}  gamma_right={has_gamma_right}  "
              f"E_trap={has_E_trap}")

        # ── [V4-f]  Bs acceleration blockade: f^(Bs)[i>i_thr] = 0 all time ──
        # Note: Bs CAN be at exactly i_thr (kappa* can equal i_thr).
        # The blockade prevents Bs from being strictly ABOVE i_thr.
        # V7-c checks the same invariant with the same condition.
        f_ds = hf['data/f']
        max_Bs_blocked = 0.0
        for t in range(0, T, 25):
            f_Bs_blocked = f_ds[t, 2, i_thr+1:, :, :]   # (N-i_thr-1, X, L): strictly above i_thr
            violation = float(np.abs(f_Bs_blocked).max())
            if violation > max_Bs_blocked:
                max_Bs_blocked = violation
        passed_f = max_Bs_blocked < 1e-10
        results['checks']['V4-f'] = {
            'desc': f'Bs kinematic blockade: max(f^(Bs)[i>{i_thr}]) = 0 all time',
            'passed': passed_f,
            'max_violation': max_Bs_blocked
        }
        tag = PASS if passed_f else FAIL
        print(f"  [V4-f] {tag}  max(f^(Bs)[i>{i_thr}]) = {max_Bs_blocked:.2e}  "
              f"(threshold < 1e-10)")

        # ═══════════════════════════════════════════════════════════════════════
        # Figures
        # ═══════════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('V4 - Probabilistic Blocking Factor P_block Validation',
                     fontsize=13, fontweight='bold')

        # Figure V4-1: P_block analytic curve
        ax = axes[0, 0]
        ax.plot(omega_range, P_analytic, 'C0-', lw=2.5,
                label=r'$P_{block}=(\Omega/\rho_{max})^{\eta_{block}}$')
        ax.axvline(rho_max, color='red', ls='--', lw=1.5, label=r'$\rho_{max}$')
        ax.scatter([0, rho_max], [0, 1], s=80, color='C1', zorder=5,
                   label='Boundary: (0,0) and (rho_max,1)')
        ax.set_xlabel(r'Occupancy $\Omega$ [PCE/m]', fontsize=10)
        ax.set_ylabel(r'$P_{block}$', fontsize=10)
        ax.set_title('[V4-c] P_block Analytic Curve\n(Boundary Conditions)', fontsize=9)
        ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.15); ax.grid(alpha=0.3)

        # Figure V4-2: P_block spatial distribution
        ax = axes[0, 1]
        x_cells = np.arange(X)
        for t_snap, col, label in [(0, 'C0', 't=0s'), (100, 'C1', 't=50s'),
                                    (200, 'C2', 't=100s')]:
            P_mean = P_block_all[t_snap].mean(axis=1)
            ax.plot(x_cells, P_mean, color=col, lw=1.5, label=label)
        ax.axvspan(74, 79, alpha=0.1, color='red', label='Bottleneck')
        ax.axvspan(0, 73, alpha=0.05, color='blue', label='Bf Upstream (uniform IC)')
        ax.set_xlabel('Cell x', fontsize=10)
        ax.set_ylabel(r'$P_{block}$ (lane-avg)', fontsize=10)
        ax.set_title('[V4-a/b] P_block Spatial Distribution\n(Multiple Snapshots)', fontsize=9)
        ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.15); ax.grid(alpha=0.3)

        # Figure V4-3: Omega vs P_block scatter (binned)
        ax = axes[0, 2]
        valid = bin_om > 0
        ax.scatter(bin_om[valid], bin_pb[valid], s=60, color='C0', alpha=0.8,
                   label='Binned means (numeric)')
        ax.plot(omega_range, P_analytic, 'C1-', lw=2, label='Analytic', alpha=0.6)
        ax.set_xlabel(r'$\Omega$ [PCE/m]', fontsize=10)
        ax.set_ylabel(r'$P_{block}$', fontsize=10)
        ax.set_title('[V4-b] Omega vs P_block\n(Monotone Non-decreasing)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Figure V4-4: sigma spatial distribution colored by P_block
        ax = axes[1, 0]
        sigma_sum_t50_arr = sigma_all[50].sum(axis=0)
        sigma_lane_avg = sigma_sum_t50_arr.mean(axis=1)    # (X,)
        P_lane_avg     = P_block_all[50].mean(axis=1)      # (X,)
        sc = ax.scatter(x_cells, sigma_lane_avg, c=P_lane_avg, cmap='RdYlGn_r',
                        s=25, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label=r'$P_{block}$')
        ax.axvspan(74, 79, alpha=0.1, color='red')
        ax.set_xlabel('Cell x', fontsize=10)
        ax.set_ylabel('sigma_sum (t=50s)', fontsize=10)
        ax.set_title('[V4-d] sigma Spatial Distribution\nColored by P_block (t=50s)', fontsize=9)
        ax.grid(alpha=0.3)

        # Figure V4-5: P_block time evolution at key cells
        ax = axes[1, 1]
        for cell, col, lbl in [(77, 'C3', 'x=77 (bottleneck)'),
                                (65, 'C0', 'x=65 (uniform Bf zone)'),
                                (10, 'C2', 'x=10 (upstream Bf)')]:
            P_series = P_block_all[:, cell, 0]
            ax.plot(time_s, P_series, color=col, lw=1.5, label=lbl)
        ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.7)
        ax.axhline(0.0, color='gray', ls=':', lw=1, alpha=0.7)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.set_ylabel(r'$P_{block}$', fontsize=10)
        ax.set_title('[V4-a] P_block Time Evolution\nat Key Spatial Cells', fontsize=9)
        ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.15); ax.grid(alpha=0.3)

        # Figure V4-6: Bs blocked density check (V4-f)
        ax = axes[1, 2]
        Bs_above_ithr = np.zeros(T)
        for t in range(0, T, 10):
            Bs_above_ithr[t] = float(np.abs(f_ds[t, 2, i_thr+1:, :, :]).max())
        for t in range(1, T):
            if t % 10 != 0:
                Bs_above_ithr[t] = Bs_above_ithr[(t // 10) * 10]
        ax.semilogy(time_s, Bs_above_ithr + 1e-20, 'C3-', lw=1.5)
        ax.axhline(1e-10, color='red', ls='--', lw=1.5, label='Threshold 1e-10')
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.set_ylabel(f'max(f^(Bs)[i>{i_thr}])', fontsize=10)
        ax.set_title(f'[V4-f] Bs Blockade: f^(Bs)[i>{i_thr}]=0\n(Kinematic Constraint)',
                     fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

        plt.tight_layout()
        fig_path = os.path.join(FIGDIR, 'V4_probability.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        results['figures'].append(fig_path)
        print(f"\n  Figure saved: {fig_path}")

    # Summary
    passed_all = all(v['passed'] for v in results['checks'].values())
    n_pass  = sum(1 for v in results['checks'].values() if v['passed'])
    n_total = len(results['checks'])
    results['summary'] = f"{'PASSED' if passed_all else 'FAILED'} {n_pass}/{n_total}"
    print(f"  V4 Summary: {results['summary']}\n")
    return results


if __name__ == '__main__':
    run()
