# Multiclass TRM — Parameter Reference Sheet

**Model version**: m7 (P_block probabilistic blocking)  
**Last updated**: 2026-04-13  
**Purpose**: Parameter overview for supervisor review and calibration discussion

---

## Hovmöller Space-Time Density Diagram (V3)

![V3 Hovmöller](figures/V3_fvm.png)

> Color: PCE-weighted occupancy Ω (turbo colormap). x-axis = road cell (0–149, ring road). y-axis = time (0–250 s).  
> White dashed lines = truck bottleneck zone (x = 74–79).  
> The dark left-leaning boundary is the backward-propagating shock wave (V3-c).  
> The smooth right side is the forward rarefaction wave (V3-g).

---

## 1. Grid / Discretization

| Parameter | Symbol | Value | Unit | Notes |
|-----------|--------|-------|------|-------|
| Spatial cells | X | 150 | — | Total road cells |
| Cell length | Δx | 20.0 | m | Road length = 3,000 m |
| Lanes | L | 3 | — | Each lane is an independent 1D system |
| Speed categories | N | 15 | — | Discrete velocity bins |
| Vehicle classes | M | 3 | — | A (trucks), Bf (free cars), Bs (trapped cars) |
| Time step | Δt | 0.5 | s | |
| Total steps | T | 500 | — | Simulation duration = 250 s |
| CFL number | — | **0.75** | — | = Δt · v_max / Δx ≤ 1.0 ✓ |

---

## 2. Speed Spectrum

| Index i | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---------|---|---|---|---|---|---|---|---|---|---|----|----|----|----|-----|
| v_i (m/s) | 2 | 4 | 6 | 8 | 10 | 12 | **14** | 16 | 18 | 20 | 22 | 24 | 26 | 28 | **30** |
| v_i (km/h) | 7.2 | 14.4 | 21.6 | 28.8 | 36.0 | 43.2 | **50.4** | 57.6 | 64.8 | 72.0 | 79.2 | 86.4 | 93.6 | 100.8 | **108.0** |

- i = 6 (14 m/s) = truck free-flow speed cap (`i_thr`)
- i = 14 (30 m/s) = maximum speed (`v_max`)

---

## 3. Macroscopic Capacity & Filter Parameters

| Parameter | Symbol | Value | Unit | Role |
|-----------|--------|-------|------|------|
| Jam density | ρ_max | 0.15 | PCE/m | Hard capacity ceiling; Ω must never exceed this |
| Supply filter | R_supply | 0.035 | PCE/m | Exponential advection filter: Ψ = v·f·(1 − e^{−avail/R_supply}) |
| Acceleration relaxation | R_c | 0.05 | PCE/m | Capacity threshold for acceleration rate |
| Regularizer | ε | 1×10⁻⁸ | — | Prevents division by zero in singular barriers |

---

## 4. PCE Weights

| Class | Symbol | PCE Weight w | Physical Meaning |
|-------|--------|-------------|-----------------|
| Trucks | A | **2.5** | One truck occupies 2.5× the road space of a car |
| Free cars | Bf | **1.0** | Standard passenger car |
| Trapped cars | Bs | **1.0** | Same physical size as Bf; behaviorally different |

Effective occupancy (Eq. 1):

$$\Omega_{x,l} = \sum_{m \in \{A, B_f, B_s\}} \sum_i w^{(m)} f_{i,x,l}^{(m)}$$

---

## 5. Per-Class Kinematic Parameters

| Parameter | Symbol | Class A (Trucks) | Class Bf (Free Cars) | Class Bs (Trapped Cars) | Unit |
|-----------|--------|------------------|-----------------------|--------------------------|------|
| Base acceleration rate | α | 0.35 | 1.50 | 1.50 | Hz |
| Singular barrier exponent | η_m | **4.5** | 2.0 | 2.0 | — |
| Spontaneous anticipation rate | ω₀ | 0.05 | 0.01 | 0.01 | Hz |
| Truck speed cap | v_A_ff | **14.0 m/s** | — | — | m/s |
| Speed cap index | i_thr | **6** | — | — | — |

> **Note on η_m**: Trucks have η_m = 4.5 (much stiffer barrier) vs. cars at 2.0. This creates an extremely hard deceleration wall near capacity — singular barrier B(Ω) = (ρ_max / (ρ_max − Ω))^η can reach ~4×10⁶ near the bottleneck.

> **Note on Bs acceleration blockade**: λ_acc^(Bs) = 0 for all i ≥ i_thr. Trapped cars are kinematically prevented from accelerating above 14 m/s while behind a truck.

---

## 6. Kinetic Collision Kernel β (3×3 Matrix)

$$\beta^{(m,n)} \text{ [m}^{-1}\text{]}: \quad \text{row = trailing class } m, \quad \text{col = blocking class } n$$

| Trailing \ Blocking | A (trucks) | Bf (free cars) | Bs (trapped cars) |
|--------------------|-----------|---------------|------------------|
| **A (trucks)** | 0.12 | 0.08 | 0.08 |
| **Bf (free cars)** | 0.06 | 0.03 | 0.03 |
| **Bs (trapped cars)** | 0.06 | 0.03 | 0.03 |

> Trucks interact most strongly when following other trucks (0.12). Cars following trucks (0.06) decelerate at half the rate of truck-truck interactions.

---

## 7. Moving Bottleneck / Capture-Release Parameters (Phase 1)

| Parameter | Symbol | Value | Unit | Role in Formula |
|-----------|--------|-------|------|----------------|
| Blocking exponent | η_block | 2.0 | — | P_block = (Ω/ρ_max)^η_block  (Eq. 8) |
| Spontaneous exposure rate | ω₀_BA | 0.05 | Hz | Kinetic exposure A^(i) (Eq. 7) |
| Kinetic exposure rate | β_BA | 0.06 | m⁻¹ | Kinetic exposure A^(i) (Eq. 7) |
| Base capture rate | **σ₀** | **0.8** | Hz | σ = σ₀ · P_block · A^(i) · B(Ω)  (Eq. 9) |
| Base release rate | μ₀ | 0.3 | Hz | μ = μ₀ · exp(−S̃/R_A)  (Eq. 12) |
| Truck dispersal scale | R_A | 0.05 | PCE/m | Controls how fast μ drops with truck presence |

**Key formulas (Phase 1):**

$$P_{\text{block}}(x) = \left(\frac{\Omega_x}{\rho_{\max}}\right)^2 \qquad \text{(Eq. 8)}$$

$$\sigma^{(i)}_x = \sigma_0 \cdot P_{\text{block}} \cdot A^{(i)}_x \cdot B(\Omega_x) \qquad \text{(Eq. 9)}$$

$$\mu_x = \mu_0 \cdot \exp\!\left(-\frac{\tilde{S}_x}{R_A}\right) \qquad \text{(Eq. 12)}$$

---

## 8. Initial Conditions (Benchmark Scenario)

| Zone | Class | Cells | Speed Index | Speed | Density | Ω (PCE/m) |
|------|-------|-------|-------------|-------|---------|-----------|
| Truck bottleneck | A | x = 74–79 | i = 0 | 2.0 m/s (7.2 km/h) | 0.058 veh/m | **0.145** (97% of ρ_max) |
| Uniform upstream | **Bf** | x = 0–73 | i = 14 | 30.0 m/s (108 km/h) | **0.035** veh/m | 0.035 |
| Downstream | — | x = 80–149 | — | — | ~0 (background 1×10⁻⁵) | ~0 |
| Trapped cars | Bs | all | — | — | 0.0 | 0 |

**Boundary condition**: Ring road (periodic). Vehicles exiting at x = 149 re-enter at x = 0. No external inflow. Net boundary flux = 0 → mass conservation is exact.

---

## 9. Parameter Change Log

| Parameter | Previous Value | **Current Value** | Changed On | Reason |
|-----------|---------------|-------------------|-----------|--------|
| σ₀ (base capture rate) | 0.5 Hz | **0.8 Hz** | 2026-04-13 | Increase Bs accumulation; stronger shock signal |
| ρ_Bf (upstream Bf density) | 0.020 veh/m | **0.035 veh/m** | 2026-04-13 | Stronger upstream car flow; clearer shock in Hovmöller |
| Bf initial zone | x = 59–69 (localized pulse) | **x = 0–73 (uniform)** | earlier | Classic Riemann IC; sustained shock vs. transient pulse |
| Boundary condition | Open (Bf injection at x=0) | **Ring road (periodic)** | earlier | Exact mass conservation; cleaner wave dynamics |

**Effect of 2026-04-13 change**:

| Metric | Before (σ₀=0.5, ρ=0.020) | After (σ₀=0.8, ρ=0.035) |
|--------|--------------------------|--------------------------|
| f_Bs_total peak | ~0.003 | **~2.47** |
| Shock delta Δ | +0.104 | +0.087 |
| V5 mass error | ≤ 5×10⁻¹⁶ | ≤ 5×10⁻¹⁶ (machine precision) |
| Validation | 41/41 PASS | **41/41 PASS** |

---

## 10. Validation Status (41/41 PASS)

| Module | Checks | Status | Key Result |
|--------|--------|--------|-----------|
| V1 Occupancy | 5/5 | ✓ | max(Ω) = 0.1499 ≤ ρ_max = 0.15 |
| V2 Kinematics | 7/7 | ✓ | Singular barrier B_A = 4.44×10⁶ at bottleneck |
| V3 FVM / Shockwave | 7/7 | ✓ | Shock Δ = +0.087; rarefaction gradient = 0.019 |
| V4 P_block | 6/6 | ✓ | P_block bounded, monotone; corr(σ, P_block) = 0.30 |
| V5 Mass conservation | 6/6 | ✓ | Error ≤ 5.44×10⁻¹⁶ (machine precision) |
| V6 Stiffness | 6/6 | ✓ | Stiffness ratio S = 1.57×10⁷; Thomas stable |
| V7 Reactions | 4/4 | ✓ | Phase 1 zero-sum; Bs blockade invariant |
