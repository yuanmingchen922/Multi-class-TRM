# Multiclass TRM — Version m7: Probabilistic Blocking Model

> **Branch**: main  
> **Validation**: 41/41 checks PASSED across 7 modules  
> **Dataset**: `multiclass_trm_benchmark_500mb.h5` (358 MB, 250 s, 500 steps)

---

## What Changed from m3+m4

The lateral lane-changing phase (Softplus γ, escape gate G, entrapment factor E_trap) is **completely removed**.
Each lane is now an independent 1D system. The three-class model (A trucks, Bf free cars, Bs trapped cars) is governed by **3-phase Lie-Trotter splitting** instead of 4-phase.

| Item | m3+m4 (old) | m7 / P_block (new) |
|------|-------------|---------------------|
| Operator splitting | 4-phase (capture → kinematics → **lateral** → advection) | 3-phase (capture → kinematics → advection) |
| Blocking mechanism | Softplus γ, escape gate G(Ω), E_trap, ξ cross-lane | `P_block(x)` — local density only |
| Lane coupling | Inter-lane (ξ, R_gap, ω_sp) | **None** — independent 1D per lane |
| Removed parameters | — | `η_lat`, `ξ`, `R_gap`, `ω_sp`, `κ_hz` |
| New parameter | — | `η_block = 2.0` |
| HDF5 datasets removed | — | `gamma_left`, `gamma_right`, `E_trap` |
| HDF5 datasets added | — | `P_block` shape `(T, X, L)` |

---

## Core Equations

### Occupancy (Eq. 1)

$$\Omega_{x,l} = \sum_{m \in \{A, B_f, B_s\}} w^{(m)} \sum_i f_{i,x,l}^{(m)}$$

PCE weights: $w^{(A)} = 2.5$, $w^{(B_f)} = w^{(B_s)} = 1.0$.

### Singular Barrier (Eq. 2)

$$\mathcal{B}(\Omega_x) = \left(\frac{\rho_{\max}}{\max(\varepsilon,\, \rho_{\max} - \Omega_x)}\right)^{\eta^{(m)}}$$

Prevents overcapacity: $\mathcal{B} \to \infty$ as $\Omega \to \rho_{\max}$.  
Per-class exponents: $\eta^{(A)} = 4.5$, $\eta^{(B_f)} = \eta^{(B_s)} = 2.0$.

### Probabilistic Blocking Factor (Eq. 8) ← *new in m7*

$$P_{\text{block}}(x) = \left(\frac{\operatorname{clip}(\Omega_x,\, 0,\, \rho_{\max})}{\rho_{\max}}\right)^{\eta_{\text{block}}}$$

Properties: $P_{\text{block}}(0) = 0$, $P_{\text{block}}(\rho_{\max}) = 1$, monotone non-decreasing.

### Kinematic Exposure (Eq. 7)

$$A_x^{(i)} = \sum_{k \leq i_{\text{thr}}} \mathbf{1}[v_i \geq v_k]\,\bigl(\omega_0^{(B,A)} + \beta^{(B,A)}(v_i - v_k)\bigr)\,\frac{w^{(A)} f_{k,x}^{(A)}}{\rho_{\max}}$$

### Capture Rate (Eq. 9)

$$\sigma_x^{(i)} = \sigma_0^{(B)}\, P_{\text{block}}(x)\, A_x^{(i)}\, \mathcal{B}(\Omega_x)$$

### Truck Footprint (Eq. 10)

$$\theta_x^{(A)} = \sum_{k \leq i_{\text{thr}}} \frac{w^{(A)} f_{k,x}^{(A)}}{\rho_{\max}}$$

### Effective Pressure (Eq. 11)

$$\tilde{S}_x = P_{\text{block}}(x)\, \theta_x^{(A)}$$

### Release Rate (Eq. 12)

$$\mu_x = \mu_0^{(B)}\, \exp\!\left(-\frac{\tilde{S}_x}{R_A}\right)$$

### Exact Matrix Exponential — Phase 1

For each cell and speed bin, the Bf ↔ Bs exchange is solved exactly:

$$f_{B_f}^{*} = f_{B_f}\,e^{-\sigma \Delta t} + \mu F\,\Delta t\,\varphi(\sigma \Delta t), \qquad f_{B_s}^{*} = F - f_{B_f}^{*}$$

where $F = f_{B_f} + f_{B_s}$ (total B mass) and $\varphi(z) = (1 - e^{-z})/z$ (safe at $z \to 0$: $\varphi \to 1$).

### Bs Kinematic Blockade — Phase 2

$$\lambda_{\text{acc}}^{(B_s)}[i \geq i_{\text{thr}}] = 0 \quad \text{(acceleration strictly forbidden above threshold speed bin)}$$

Replaces the old lateral immobilization with a pure kinematic constraint.

### Global Godunov Flux Limiter — Phase 3

$$\alpha = \min\!\left(1,\; \frac{\rho_{\max} - \Omega_{\text{downstream}}}{\sum_{m,i} w^{(m)} \Psi^{(m,i)} \cdot \Delta t / \Delta x}\right)$$

$$\Phi^{(m,i)} = \alpha\, \Psi^{(m,i)}, \qquad \Psi^{(m,i)} = v_i\, f^{(m,i)}\,\bigl(1 - e^{-(\rho_{\max} - \Omega)/R_{\text{supply}}}\bigr)$$

---

## Riemann Problem Validation

Two canonical Riemann problems are encoded in the benchmark and validated by explicit quantitative checks.

### Shock Wave — V3-c (free flow upstream, congested downstream)

**Setup**: High-speed Bf platoon (x = 59–69) collides with stationary truck bottleneck (x = 74–79, $\Omega \approx \rho_{\max}$).

**Mechanism**: $\Omega \to \rho_{\max}$ triggers $\mathcal{B} \to \infty$ and $P_{\text{block}} \to 1$; Bf is captured as Bs at maximum rate; the density jump propagates **backward** (Rankine-Hugoniot).

**Check V3-c**: upstream $\omega(x = 68\text{–}73)$ increases from $t=0$ to $t=50\,\text{s}$ (positive delta confirms backward shock); complementary: $\rho_{B_s} > 0.001$ confirms capture at the shock front.

### Rarefaction Wave — V3-g (congested upstream, free flow downstream)

**Setup**: Downstream of the bottleneck (x = 80–110), traffic escapes into sparse space.

**Mechanism**: $\Omega \ll \rho_{\max}$ downstream → $P_{\text{block}} \to 0$ → $\mu$ increases; Bs is slowly released to Bf; density fan expands **forward** smoothly (no sharp front).

**Check V3-g**: max cell-to-cell gradient $|\omega(x+1) - \omega(x)|$ for $x \in [80, 110]$ is $< 0.5\,\rho_{\max}$ (smooth fan); far-downstream $\rho_{B_s} \ll$ bottleneck $\rho_{B_s}$ (free flow confirmed, no trapping).

---

## 3-Phase Operator Splitting

```
Each time step Δt:

Phase 1 — Capture & Release  (exact matrix exponential, per cell)
    P_block = (clip(Ω, 0, ρ_max) / ρ_max) ^ η_block
    σ       = σ_0 · P_block · A^(i) · B(Ω)
    θ_A     = Σ_{k≤i_thr} w^A · f^A_k / ρ_max
    S̃       = P_block · θ_A
    μ       = μ_0 · exp(-S̃ / R_A)
    f_Bf*   = f_Bf · exp(-σΔt) + μ·F·Δt·φ(σΔt)
    f_Bs*   = F − f_Bf*

Phase 2 — Kinematics  (algebraic projection + Thomas implicit solver)
    Project: f^(Bs)[i > κ*] → f^(Bs)[κ*]       (speed-bin ceiling)
    Blockade: λ_acc^(Bs)[i ≥ i_thr] = 0
    Solve:    (I − Δt·A) f_new = f_old           (Thomas algorithm)

Phase 3 — Spatial Advection  (explicit Euler + global Godunov limiter)
    Ψ     = v_i · f · (1 − exp(−(ρ_max−Ω)/R_supply))
    D     = (Δt/Δx) · Σ_{m,i} w^m · Ψ
    α     = min(1, available / D)
    Φ     = α · Ψ
    f    += (Δt/Δx) · (Φ_in − Φ_out)
```

---

## Validation Modules

| Module | Checks | Key Assertions |
|--------|--------|----------------|
| **V1** Occupancy | 5/5 | $\max(\Omega) \leq \rho_{\max}$; PCE formula exact; $w^{(B_f)} = w^{(B_s)}$ |
| **V2** Kinematics | 7/7 | $\lambda_{\text{acc}}$ monotone; singular barrier exact; Bs blockade holds; $\lambda_{\text{dec}}(0) = 0$ |
| **V3** FVM + Riemann | 7/7 | Supply filter $\Psi \to 0$; $\min(f) \geq 0$; **shock V3-c**; CFL $\leq 1$; $\alpha \leq 1$; bimodal FD; **rarefaction V3-g** |
| **V4** P_block | 6/6 | $P_{\text{block}} \in [0,1]$; monotone; BCs; $\sigma \propto P_{\text{block}}$; no old datasets; Bs blockade |
| **V5** Mass Conservation | 6/6 | Theorem 1 holds; Phase 1 zero-sum ($\Delta P_{B_f} + \Delta P_{B_s} \approx$ boundary flux); lane fractions symmetric |
| **V6** Stiffness | 6/6 | $S = \lambda_{\text{dec,max}} / (v_{\max}/\Delta x) > 10^5$; explicit Euler diverges; Thomas residual $< 10^{-8}$; $\varphi(z)$ safe |
| **V7** Reactions | 4/4 | $\varphi(z \to 0) = 1$; $P_{\text{block}}$ topology; $f^{(B_s)}[i > i_{\text{thr}}] = 0$; $\sigma, \mu \geq 0$ |
| **Total** | **41/41** | ALL PASSED |

---

## Code Changes

### `Benchmark Dataset.json`
- Removed: `R_gap`, `omega_sp`, `eta_lat`, `xi`, `kappa_hz` (all lateral parameters)
- Added: `"eta_block": 2.0`
- Operator splitting updated: 4-phase → 3-phase (Phase 3 lateral removed)

### `generate_dataset.py`
- `phase1_capture_release()` rewritten: E_trap/G/ξ replaced by P_block
- `phase3_lateral()` function **deleted**
- `phase4_advection()` renamed to `phase3_advection()`
- HDF5 schema: removed `gamma_left`, `gamma_right`, `E_trap`; added `P_block (T, X, L)`
- Main loop: 4 phases → 3 phases

### `V4_lateral.py` → complete rewrite
Old purpose: Softplus γ continuity validation  
New purpose: P_block probabilistic blocking validation (V4-a through V4-f)

### `V3_fvm.py`
- V3-c strengthened: explicit **shock wave** Riemann check (upstream omega increase)
- V3-g added: explicit **rarefaction wave** Riemann check (smooth downstream gradient)
- All figure labels translated to English

### `V7_reactions.py`
- V7-b rewritten: E_trap topology → P_block topology check
- All E_trap references removed

### `V5_mass.py`
- V5-e threshold tightened: `0.05 → 1e-6` (symmetric independent lanes)
- All figure labels translated to English

### `V6_stiffness.py`
- V6-f updated: reads `P_block` instead of `E_trap`
- All figure labels translated to English

### `V1_occupancy.py`, `V2_kinematics.py`
- All figure labels translated to English

### `run_all.py`
- Added `_NumpyEncoder` to handle `numpy.bool_` JSON serialization
- Research propositions updated to reflect P_block / 3-phase model
- Total check count updated: 40 → 41

### `results_chinese.md`, `results_english.md`
- Complete rewrite: all figure explanations updated for new module outputs
- Riemann problem section expanded with two-type analysis (shock + rarefaction)
- Check count: 40/40 → 41/41

---

## Stiffness Numbers

| Quantity | Value |
|----------|-------|
| $\tau_{\text{adv}} = \Delta x / v_{\max}$ | 0.667 s |
| $\tau_{\text{relax}} = 1/\lambda_{\text{dec,max}}$ | down to $10^{-15}$ s |
| Time-scale separation $\tau_{\text{adv}}/\tau_{\text{relax}}$ | up to $10^{15}$ |
| Initial stiffness ratio $S(t=0)$ | $1.57 \times 10^7$ |
| Peak stiffness ratio $S_{\max}$ | $1.97 \times 10^{15}$ |
| Thomas algorithm residual | $6.94 \times 10^{-18}$ |
