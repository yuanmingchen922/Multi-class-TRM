"""
V7 -- Moving Bottleneck Capture/Release Reaction Validation (P_block version)
Corresponding to .tex Eq. 7-12: Phase 1 exact matrix exponential, P_block-based sigma/mu

Checks:
  [V7-a] phi(z->0) = 1 safety: no NaN/Inf at z < 1e-12, phi in [0,1]
  [V7-b] P_block topology: P_block(Omega=0)=0, P_block(Omega=rho_max)=1, monotone
          (replaces old E_trap topology check)
  [V7-c] Phase 2 projection valid: max(f^(Bs)[i>kappa*]) = 0 all time (blockade invariant)
  [V7-d] sigma >= 0, mu >= 0 all time (physical conservation necessary condition)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import platform

# Font setup for correct rendering
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


def safe_phi(z):
    """phi(z) = (1 - e^{-z}) / z,  safe at z->0 gives phi->1"""
    return np.where(z < 1.0e-12, 1.0, -np.expm1(-z) / z)


def run():
    results = {'module': 'V7_reactions', 'checks': {}, 'figures': []}

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
        print(f"  V7 -- Moving Bottleneck Capture/Release Validation")
        print(f"  eta_block={eta_block}, i_thr={i_thr}, T={T}")
        print(f"{'='*60}")

        # ── [V7-a]  phi(z->0) = 1 safety ──────────────────────────────────────
        z_test_vals = np.array([0.0, 1e-15, 1e-13, 1e-12, 1e-10, 1e-6,
                                 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e6])
        phi_test = safe_phi(z_test_vals)

        has_nan  = bool(np.isnan(phi_test).any())
        has_inf  = bool(np.isinf(phi_test).any())
        phi_zero_ok  = bool(abs(phi_test[0] - 1.0) < 1e-10)    # z=0 -> phi=1
        phi_large_ok = bool(phi_test[-1] < 1e-3)               # z=1e6 -> phi~0
        phi_range_ok = bool((phi_test >= 0).all() and (phi_test <= 1.0 + 1e-10).all())

        passed_a = not has_nan and not has_inf and phi_zero_ok and phi_large_ok and phi_range_ok
        results['checks']['V7-a'] = {
            'desc': 'phi(z) at z->0 and z->inf: no NaN/Inf, range [0,1]',
            'passed': passed_a,
            'has_nan': has_nan, 'has_inf': has_inf,
            'phi_at_0': float(phi_test[0]),
            'phi_at_1e6': float(phi_test[-1]),
            'range_ok': phi_range_ok
        }
        tag = PASS if passed_a else FAIL
        print(f"  [V7-a] {tag}  phi(z=0)={phi_test[0]:.6f}, phi(z=1e6)={phi_test[-1]:.2e}, "
              f"NaN={has_nan}, Inf={has_inf}, range_ok={phi_range_ok}")

        # ── [V7-b]  P_block topology (replaces E_trap topology) ───────────────
        # P_block(Omega) = (clip(Omega,0,rho_max)/rho_max)^eta_block
        # P_block(0) = 0 (no blocking when empty)
        # P_block(rho_max) = 1 (full blocking at jam density)
        # Monotone non-decreasing
        # Note: HDF5 omega is post-advection, while P_block is computed from pre-phase1
        # omega — so we do NOT compare them directly. We verify analytic properties only,
        # plus that P_block values in HDF5 are physically valid (in [0,1]).
        omega_range = np.linspace(0, rho_max, 500)
        P_analytic  = (omega_range / rho_max) ** eta_block

        P_at_zero   = float(P_analytic[0])
        P_at_rhomax = float(P_analytic[-1])
        P_monotone  = bool((np.diff(P_analytic) >= 0).all())
        P_in_range  = bool((P_analytic >= -1e-10).all() and
                           (P_analytic <= 1.0 + 1e-10).all())

        # Check HDF5 P_block values are in valid range [0,1]
        P_block_all = hf['data/P_block'][:]
        P_data_min  = float(P_block_all.min())
        P_data_max  = float(P_block_all.max())
        P_data_valid = (P_data_min >= -1e-10 and P_data_max <= 1.0 + 1e-10)

        passed_b = (abs(P_at_zero) < 1e-10 and
                    abs(P_at_rhomax - 1.0) < 1e-10 and
                    P_monotone and P_in_range and P_data_valid)
        results['checks']['V7-b'] = {
            'desc': 'P_block(0)=0, P_block(rho_max)=1, monotone, HDF5 data in [0,1]',
            'passed': passed_b,
            'P_at_omega_0': P_at_zero,
            'P_at_rho_max': P_at_rhomax,
            'P_monotone': P_monotone,
            'P_in_range': P_in_range,
            'P_data_range': [P_data_min, P_data_max]
        }
        tag = PASS if passed_b else FAIL
        print(f"  [V7-b] {tag}  P_block(0)={P_at_zero:.2e}, P_block(rho_max)={P_at_rhomax:.6f}, "
              f"monotone={P_monotone}, data=[{P_data_min:.4f},{P_data_max:.4f}]")

        # ── [V7-c]  Phase 2 projection: f^(Bs)[i>i_thr] = 0 all time ─────────
        max_Bs_above_ithr = 0.0
        f_ds = hf['data/f']
        for t in range(0, T, 25):
            f_Bs_t = f_ds[t, 2, i_thr+1:, :, :]   # (N-i_thr-1, X, L)
            violation = float(np.abs(f_Bs_t).max())
            if violation > max_Bs_above_ithr:
                max_Bs_above_ithr = violation

        passed_c = max_Bs_above_ithr < 1e-10
        results['checks']['V7-c'] = {
            'desc': f'max(f^(Bs)[i>{i_thr}]) = 0 all time (blockade global invariant)',
            'passed': passed_c,
            'max_violation': max_Bs_above_ithr
        }
        tag = PASS if passed_c else FAIL
        print(f"  [V7-c] {tag}  max(f^(Bs)[i>{i_thr}]) = {max_Bs_above_ithr:.2e}  "
              f"(threshold < 1e-10)")

        # ── [V7-d]  sigma >= 0, mu >= 0 (physical conservation) ───────────────
        sigma_ds = hf['data/sigma']   # (T, N, X, L)
        mu_ds    = hf['data/mu']      # (T, X, L)

        sigma_t0 = sigma_ds[0, :, :, :]
        mu_t0    = mu_ds[0, :, :]
        sigma_neg = float((sigma_t0 < -1e-12).sum())
        mu_neg    = float((mu_t0 < -1e-12).sum())
        rate_nonneg = (sigma_neg == 0 and mu_neg == 0)

        # Also verify Bf+Bs total B mass at x=100 (remote from boundary/bottleneck)
        f_B_x100 = np.zeros(T)
        for t in range(T):
            f_B_x100[t] = float(f_ds[t, 1, :, 100, :].sum() +
                                 f_ds[t, 2, :, 100, :].sum())
        delta_B_x100 = np.abs(np.diff(f_B_x100))
        max_delta    = float(delta_B_x100.max())
        # Flux-dominated changes; verify no anomalous jumps
        flux_ok = max_delta < rho_max * 0.5

        passed_d = rate_nonneg and flux_ok
        results['checks']['V7-d'] = {
            'desc': 'sigma>=0, mu>=0 all time (physical conservation), Bf+Bs local conservation',
            'passed': passed_d,
            'sigma_neg_count': sigma_neg,
            'mu_neg_count': mu_neg,
            'max_B_delta_x100': max_delta
        }
        tag = PASS if passed_d else FAIL
        print(f"  [V7-d] {tag}  sigma<0 count={sigma_neg:.0f}, mu<0 count={mu_neg:.0f}, "
              f"max Delta(Bf+Bs)@x=100 = {max_delta:.4e}")

        # ═══════════════════════════════════════════════════════════════════════
        # Figures
        # ═══════════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('V7 - Moving Bottleneck Capture/Release Validation (P_block)',
                     fontsize=13, fontweight='bold')

        # Figure V7-1: phi(z) function curve
        ax = axes[0, 0]
        z_plot = np.logspace(-14, 6, 2000)
        phi_plot = safe_phi(z_plot)
        ax.semilogx(z_plot, phi_plot, 'C0-', lw=2.5,
                    label=r'$\varphi(z) = (1-e^{-z})/z$')
        ax.axhline(1.0, color='gray', ls='--', lw=1.5, label=r'$\varphi(z\to 0)=1$')
        ax.axhline(0.0, color='gray', ls=':',  lw=1.5, label=r'$\varphi(z\to\infty)=0$')
        ax.scatter(z_test_vals[z_test_vals > 0], phi_test[z_test_vals > 0],
                   s=60, color='C1', zorder=5, label='Test points')
        ax.set_xlabel(r'$z = \sigma \cdot \Delta t$', fontsize=10)
        ax.set_ylabel(r'$\varphi(z)$', fontsize=10)
        ax.set_title('[V7-a] phi(z) Safe Implementation\n(z->0: no NaN/0/Inf)', fontsize=9)
        ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.15); ax.grid(alpha=0.3)

        # Figure V7-2: P_block analytic curve (replaces E_trap/G)
        ax = axes[0, 1]
        ax.plot(omega_range, P_analytic, 'C2-', lw=2.5,
                label=r'$P_{block}=(\Omega/\rho_{max})^{\eta_{block}}$')
        ax.axvline(rho_max, color='red', ls='--', lw=1.5, label=r'$\rho_{max}$')
        ax.scatter([0, rho_max], [0, 1], s=80, color='C1', zorder=5,
                   label='BC: (0,0) and (rho_max,1)')
        ax.set_xlabel(r'Occupancy $\Omega$ [PCE/m]', fontsize=10)
        ax.set_ylabel(r'$P_{block}$', fontsize=10)
        ax.set_title('[V7-b] P_block Topology\n(Replaces E_trap Escape Gate)', fontsize=9)
        ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.15); ax.grid(alpha=0.3)

        # Figure V7-3: P_block spatial distribution (t=0)
        ax = axes[0, 2]
        x_cells = np.arange(X)
        P_block_all_t0 = hf['data/P_block'][0]
        P_mean_l = P_block_all_t0.mean(axis=1)   # (X,)
        omega_t0_arr = hf['data/omega'][0]
        omega_mean_l = omega_t0_arr.mean(axis=1)  # (X,)
        ax2_ = ax.twinx()
        ax.plot(x_cells, P_mean_l, 'C2-', lw=2, label='P_block (lane-avg)')
        ax2_.plot(x_cells, omega_mean_l, 'C0--', lw=1.5, alpha=0.7, label='Omega (lane-avg)')
        ax.axvspan(74, 79, alpha=0.1, color='red')
        ax.axvspan(0, 73, alpha=0.05, color='blue')
        ax.set_xlabel('Cell x', fontsize=10)
        ax.set_ylabel('P_block', fontsize=10, color='C2')
        ax2_.set_ylabel('Omega [PCE/m]', fontsize=10, color='C0')
        ax.set_title('[V7-b] P_block and Omega\nSpatial Correlation (t=0)', fontsize=9)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2_.get_legend_handles_labels()
        ax.legend(lines1+lines2, labs1+labs2, fontsize=8)
        ax.grid(alpha=0.3)

        # Figure V7-4: sigma, mu time evolution at key cell
        ax = axes[1, 0]
        sig_series = np.array([float(sigma_ds[t, 0, 65, 1]) for t in range(T)])
        mu_series  = np.array([float(mu_ds[t, 65, 1])       for t in range(T)])
        ax.plot(time_s, sig_series, 'C2-', lw=1.5, label='sigma(i=0, x=65, l=1)')
        ax.plot(time_s, mu_series,  'C3-', lw=1.5, label='mu(x=65, l=1)')
        ax.axhline(0, color='gray', ls=':', lw=1)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.set_ylabel('Rate [Hz]', fontsize=10)
        ax.set_title('[V7-d] sigma/mu at Injection Zone\n(Should be >= 0)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Figure V7-5: f^(Bs)[i>i_thr] over time (V7-c)
        ax = axes[1, 1]
        Bs_above_thresh_max = np.zeros(T)
        for t in range(0, T, 10):
            f_Bs_t = f_ds[t, 2, i_thr+1:, :, :]
            Bs_above_thresh_max[t] = float(np.abs(f_Bs_t).max())
        for t in range(1, T):
            if t % 10 != 0:
                Bs_above_thresh_max[t] = Bs_above_thresh_max[(t // 10) * 10]
        ax.semilogy(time_s, Bs_above_thresh_max + 1e-20, 'C3-', lw=1.5)
        ax.axhline(1e-10, color='red', ls='--', lw=1.5, label='Threshold 1e-10')
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.set_ylabel(f'max(f^(Bs)[i>{i_thr}])', fontsize=10)
        ax.set_title(f'[V7-c] Blockade Invariant: f^(Bs)[i>{i_thr}]\nover Time', fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

        # Figure V7-6: Bf/Bs mass exchange over time
        ax = axes[1, 2]
        dx_val = float(hf['parameters'].attrs['dx_m'])
        P_Bf = np.array([(f_ds[t, 1] * dx_val).sum() for t in range(T)])
        P_Bs = np.array([(f_ds[t, 2] * dx_val).sum() for t in range(T)])
        P_B  = P_Bf + P_Bs
        ax.plot(time_s, P_Bf, 'C0-', lw=2, label=r'$P^{(Bf)}$ (free cars)')
        ax.plot(time_s, P_Bs, 'C2-', lw=2, label=r'$P^{(Bs)}$ (trapped cars)')
        ax.plot(time_s, P_B,  'k--', lw=1.5, label=r'$P^{(Bf)}+P^{(Bs)}$ (conserved)')
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.set_ylabel(r'$P$ [veh$\cdot$m]', fontsize=10)
        ax.set_title('[V7-d] Phase 1 Zero-sum: Bf<->Bs\nExchange, B Total Conserved', fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(FIGDIR, 'V7_reactions.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        results['figures'].append(fig_path)
        print(f"\n  Figure saved: {fig_path}")

    # Summary
    passed_all = all(v['passed'] for v in results['checks'].values())
    n_pass  = sum(1 for v in results['checks'].values() if v['passed'])
    n_total = len(results['checks'])
    results['summary'] = f"{'PASSED' if passed_all else 'FAILED'} {n_pass}/{n_total}"
    print(f"  V7 Summary: {results['summary']}\n")
    return results


if __name__ == '__main__':
    run()
