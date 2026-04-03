# Multiclass TRM m3+m4
> **Author**: Mingchen and Bingjie
> **Validation Status**: **ALL PASSED — 41/41 checks, 7 core claims verified [Reliable]**  
> **Simulation Scale**: 150 cells × 3 lanes × 15 speed bins × 3 vehicle classes × 500 timesteps = 512 MB HDF5

---

## Table of Contents

1. [Model Overview](#1-model-overview)
2. [Code Architecture](#2-code-architecture)
3. [Dataset Generation Logic](#3-dataset-generation-logic)
4. [V1 — Effective Occupancy Constraint](#4-v1--effective-occupancy-constraint-v1_occupancypng)
5. [V2 — Kinematic Transition Rates](#5-v2--kinematic-transition-rates-v2_kinematicspng)
6. [V3 — Finite Volume Method & Flux Limiter](#6-v3--finite-volume-method--flux-limiter-v3_fvmpng)
7. [V4 — Lateral Lane-Changing Softplus](#7-v4--lateral-lane-changing-softplus-v4_lateralpng)
8. [V5 — Mass Conservation Theorem](#8-v5--mass-conservation-theorem-v5_masspng)
9. [V6 — Stiffness & Operator Splitting](#9-v6--stiffness--operator-splitting-v6_stiffnesspng)
10. [V7 — Capture/Release Reactions](#10-v7--capturerelease-reactions-v7_reactionspng)
11. [Core Research Claims Summary](#11-core-research-claims-summary)

---

## 1. Model Overview

### Three Vehicle Classes

The model divides highway traffic into three groups — think of it as categorizing every vehicle on a highway into one of three buckets:

| Class | Name | Description | PCE Weight |
|-------|------|-------------|------------|
| **Class A** | Trucks | Heavy vehicles with a free-flow speed cap of 14 m/s | 2.5 |
| **Class Bf** | Free Cars | Passenger cars that can move freely | 1.0 |
| **Class Bs** | Trapped Cars | Passenger cars stuck behind a slow truck, unable to escape | 1.0 |

> **Intuitive picture**: Imagine a slow truck on a highway. Cars just behind the truck (Bs) are boxed in by adjacent traffic and can't overtake. Cars further back (Bf) still have room to change lanes and go around. When a lane gap opens up, Bs cars "escape" and become Bf again.

### Core Physical Variables

- **f(m, i, x, l)**: density of class-m vehicles in speed bin i, cell x, lane l (vehicles/meter)
- **Ω(x, l)**: effective PCE occupancy = Σ w(m) × Σ f(m,i,x,l) — how "full" the road is
- **ρ_max = 0.15 PCE/m**: absolute capacity ceiling — never violated under any circumstances

### 4-Phase Operator Splitting

Each timestep executes four physical processes in strict order:

```
Phase 1: Capture/Release  →  Bf cars get "trapped" (Bf→Bs) or escape (Bs→Bf)
Phase 2: Speed Dynamics   →  Acceleration and deceleration (Thomas algorithm for stiffness)
Phase 3: Lateral Shifts   →  Lane changes between adjacent lanes (Softplus smoothing)
Phase 4: Spatial Advection →  Vehicles move forward along the road (FVM + Global Godunov)
```

---

## 2. Code Architecture

```
code/
├── generate_dataset.py    # Numerical simulation engine → 512 MB HDF5 output
├── Benchmark Dataset.json # Physical parameter specification
├── V1_occupancy.py        # Validates: occupancy never exceeds capacity
├── V2_kinematics.py       # Validates: acceleration/deceleration rate physics
├── V3_fvm.py              # Validates: finite volume fluxes and shock formation
├── V4_lateral.py          # Validates: lane-change rates and Bs immobilization
├── V5_mass.py             # Validates: total mass conservation per class
├── V6_stiffness.py        # Validates: stiffness ratio and implicit solver stability
├── V7_reactions.py        # Validates: capture/release reaction correctness (NEW)
├── run_all.py             # Master runner: executes V1–V7 and summarizes results
└── figures/               # 7 auto-generated validation figures
```

---

## 3. Dataset Generation Logic

### Initial Conditions (Pathological Stress-Test Scenario)

```
Truck Bottleneck Zone (cells 74–79):
  Class A truck density = 0.058 veh/m at minimum speed (2 m/s)
  → Effective PCE occupancy Ω = 2.5 × 0.058 = 0.145 ≈ ρ_max (nearly jammed)

High-Speed Injection Zone (cells 60–70):
  Class Bf car density = 0.060 veh/m at maximum speed (30 m/s)
  → Fast car platoon heading straight into the bottleneck → triggers mass capture events
```

This is equivalent to setting up the worst-case scenario: a near-capacity truck jam ahead, and a dense fast-moving car platoon approaching it from behind. This stresses every component of the model simultaneously.

### Per-Step Computation (Δt = 0.5 s)

**Phase 1 — Exact Matrix Exponential (Capture/Release)**

This is the model's most unique feature. When a Bf car gets boxed in by a truck:

```python
# Escape gate: adjacent lane is full → hard to escape
G = (Ω_adj / ρ_max)^η_lat      # η_lat = 2.0

# Entrapment factor: blocked on both sides = fully trapped
E = G_left × G_right

# Capture rate: more trucks, slower trucks, more boxed-in → higher σ
σ = σ_0 × (1 + ξ·E) × S_tilde  # S_tilde = truck density weighted sum

# Exact analytical solution (no numerical approximation needed)
φ(z) = (1 - e^{-z}) / z         # safely converges to 1 as z→0
f_Bf_new = f_Bf × e^{-S·Δt} + μ × (f_Bf+f_Bs) × φ(S·Δt)
```

**Phase 2 — Semi-Implicit Thomas Algorithm (Speed Dynamics)**

With a stiffness ratio of S ~ 10^7 at the bottleneck, any explicit method diverges instantly. The Thomas algorithm solves the tridiagonal system exactly in O(N) operations:

```python
# Solve: (I - Δt·A) × f_new = f_old
# A is the tridiagonal speed-transition matrix
# Forward sweep + back-substitution, fully stable at any stiffness
```

**Phase 3 — Softplus Lane-Changing (Lateral Dynamics)**

```python
# Softplus turns "switch lanes if density difference is positive" into a
# smooth, differentiable sigmoid-like function
γ = κ × (1/ω) × ln[1 + exp(ω·y)] × gap_filter
# gap_filter: if target lane is full, γ → 0
# Bs class: γ = 0 always (absolute immobilization constraint)
```

**Phase 4 — Global Godunov Flux Limiter (Spatial Advection)**

```python
# Aggregate demand flux across all vehicle classes
D = (Δt/Δx) × Σ_m w(m) × Ψ(m)  # total PCE demand

# Downstream available capacity
available = max(0, ρ_max - Ω_downstream)

# Single global scaling factor (all classes scale together)
α = min(1, available / D)

# Apply to all classes proportionally — capacity is never breached
Φ(m) = α × Ψ(m)
```

---

## 4. V1 — Effective Occupancy Constraint (`V1_occupancy.png`)

### Figure Explanation

**Left panel: Global maximum occupancy over time**

This plot tracks the highest PCE density anywhere on the road at each moment.

- **Blue line**: maximum Ω across all cells and lanes at each timestep
- **Red dashed line**: capacity ceiling ρ_max = 0.15 PCE/m
- **Orange annotation**: initial bottleneck density Ω ≈ 0.145 at t = 0

The key reading: the blue line stays below the red line at all times — the road never exceeds its physical capacity. The density peaks at t=0 (initial bottleneck) and gradually decreases as traffic disperses.

**Right panel: Per-class density contribution at t=0 (middle lane)**

A stacked area chart showing what fraction of the road's "fullness" comes from each vehicle class at simulation start:

- **Orange area (Class A Trucks)**: concentrated at x=74–79, forming the initial bottleneck
- **Blue area (Class Bf Free Cars)**: concentrated at x=60–70, the injection platoon
- **Green area (Class Bs Trapped Cars)**: essentially zero at t=0 (no one is trapped yet)
- **Red shading**: truck bottleneck zone
- **Blue shading**: car injection zone

### Validation Results

| Check | Physical Meaning | Result |
|-------|-----------------|--------|
| V1-a | Ω ≤ ρ_max everywhere, always | ✓ max(Ω)=0.1499 < 0.15 |
| V1-b | Ω ≥ 0 (no negative density) | ✓ min(Ω)=2.09e-4 ≥ 0 |
| V1-c | Initial bottleneck density correct | ✓ Ω≈0.1443 ≈ expected 0.145 |
| V1-d | Stored Ω matches formula recomputation | ✓ error = 0 (machine precision) |
| V1-e | Bf and Bs share equal PCE weight | ✓ w(Bf)=w(Bs)=1.0 |

---

## 5. V2 — Kinematic Transition Rates (`V2_kinematics.png`)

### Figure Explanation

**Top-left: Acceleration rate λ_acc vs. occupancy (three classes)**

This plot shows how eager each vehicle type is to accelerate as road congestion varies.

- **Horizontal axis**: road occupancy Ω (0 = empty, 0.15 = jammed)
- **Vertical axis**: acceleration rate (higher = stronger desire to speed up)
- **Solid lines (Class A)**: steeper decline due to η=4.5 (trucks are more sensitive to congestion)
- **Dashed lines (Bf/Bs cars)**: gentler slope due to η=2.0
- **Bs lines cut to zero at i ≥ i_thr**: this is the "acceleration blockade" — trapped cars cannot accelerate beyond the truck's speed cap

> **Plain English**: The more congested the road, the less room to accelerate. Trucks are more sensitive to congestion (larger η exponent). Trapped cars (Bs) hit a hard speed ceiling and cannot go faster than the leading truck.

**Top-right: Singular barrier B(Ω) — log scale**

The model's built-in "capacity wall" visualized:

- **Vertical axis (log scale)**: B(Ω) = (ρ_max / (ρ_max - Ω))^η — the deceleration amplifier
- **As Ω → ρ_max, B → ∞**: deceleration becomes infinite near capacity, creating an impenetrable wall
- **Orange vertical line**: bottleneck at Ω=0.145, B_A = 30^4.5 ≈ 4.4×10⁶ (enormous!)
- **Green horizontal line**: analytical expected value — matches exactly

> **Plain English**: As the road approaches full capacity, it's like hitting an invisible wall. Vehicles are forced to decelerate dramatically, making it mathematically impossible to "push through" ρ_max.

**Bottom-left: λ_dec spatial distribution at t=0 (log scale)**

Deceleration rate intensity along the road at simulation start:

- **Red shaded region (x=74–79)**: truck bottleneck — deceleration rate peaks here
- **Three classes**: Class A has the highest λ_dec (trucks block more due to higher β), Bf/Bs next

**Bottom-right: β matrix asymmetry verification**

Proves "being stuck behind a truck causes more deceleration than behind a car":

- **Blue bars (Bf following Bf)** vs **Orange bars (Bf following A)**: orange is 5× taller
- Ratio = β[1,0]×w[0] / (β[1,1]×w[1]) = 0.06×2.5 / (0.03×1.0) = **5.0**
- Physically realistic: trucks are harder to pass than passenger cars

---

## 6. V3 — Finite Volume Method & Flux Limiter (`V3_fvm.png`)

### Figure Explanation

**Top-left: Space-time Hovmöller diagram (heat map)**

The most visually intuitive plot in the entire validation suite. The x-axis is road position, y-axis is time, and color intensity shows traffic density.

- **Initial dark region (x=60–80)**: the bottleneck and injection zone at t=0
- **Color fades over time**: density spreads and equilibrates as the simulation progresses
- **Diagonal spreading pattern**: shock waves propagating upstream at finite speed

> **Plain English**: Like dropping a stone in water and watching ripples spread outward, traffic density waves propagate backward from the bottleneck.

**Top-right: Per-class density snapshots (t=0 vs t=125s)**

Side-by-side comparison of how each class evolves:

- **t=0 (dashed)**: trucks piled up at x=74–79, Bf injected at x=60–70, Bs=0
- **t=125s (solid)**: Bs density grows near the injection zone (Bf cars get captured), trucks spread out

**Bottom-left: Fundamental diagram (flow–density relationship)**

The most classic plot in traffic engineering, showing the q–ρ relationship:

- **Low-density points (blue)**: free-flow regime — cars can move fast
- **High-density points (orange)**: congested regime — cars slow down
- **Bimodal distribution**: two clearly separated clusters confirm the model captures both traffic states

**Bottom-right: Global Godunov limiter α over time**

- **α = 1.0**: no limiting needed, full flux passes through
- **α < 1.0**: downstream is getting full, all vehicle fluxes are proportionally reduced
- The generally low α confirms sustained capacity-limiting at the bottleneck

---

## 7. V4 — Lateral Lane-Changing Softplus (`V4_lateral.png`)

### Figure Explanation

**Top-left: Softplus vs Hard-max continuity comparison**

The foundational plot for the lane-changing model's mathematical correctness:

- **Horizontal axis y**: density difference between source and target lane (y>0 = "my lane is more congested, I want to switch")
- **Blue solid (Softplus)**: smooth, continuously differentiable S-shaped driving force
- **Orange dashed (Hard-max)**: classical formulation — abrupt jump in derivative at y=0
- **Dotted lines (1st derivatives)**: Softplus derivative is smooth; Hard-max derivative jumps discontinuously

> **Plain English**: Softplus makes lane-changing decisions more "human" — gradually increasing desire to switch as the density gap widens, rather than a binary on/off switch. Mathematically, this prevents numerical artifacts at the transition point.

**Top-right: Gap filter curves (three vehicle classes)**

How lane-change rate γ decays as the target lane fills up:

- **Bf (blue)**: κ=0.6, strongest desire to change lanes, but γ→0 when target is full
- **Class A trucks (orange)**: κ=0.08, much lower lane-change agility (realistic for heavy vehicles)
- **Bs (green dashed)**: always zero — trapped cars cannot change lanes (hard constraint)
- **Red vertical line**: ρ_max — all curves reach zero here

**Top-right: Class Bf γ_right time series (near bottleneck, x=77)**

- Three lines for three lanes
- **Lane 3 (outer) γ_right = 0**: validates the Dirichlet boundary condition (can't change further right from the rightmost lane)
- Fluctuations reflect the dynamic density evolution near the bottleneck

**Bottom-left: Class Bs γ_right time series (should be zero)**

- **All three lines exactly at zero**: validates the Bs absolute lateral immobilization constraint
- No matter what happens in the simulation, trapped cars' lane-change rates remain precisely zero

**Bottom-center: Per-class, per-lane average density over time**

- **Solid (A), dashed (Bf), dotted (Bs)** in three colors for three lanes
- Bs density rises then falls: corresponding to Bf cars being captured (rise) then released (fall)

**Bottom-right: Class A γ_right time series**

- Truck lane-change rate is 7.5× smaller than cars (κ_A/κ_Bf = 0.08/0.60)
- Realistic representation of heavy-vehicle dynamics

---

## 8. V5 — Mass Conservation Theorem (`V5_mass.png`)

### Figure Explanation

**Top-left: Total mass of each class over time**

- **P(m)(t) = Σ f(m,i,x,l) × Δx**: total "vehicle-meters" for each class on the entire road
- **Class A (orange)**: slowly decreasing as trucks flow out downstream
- **Bf free cars (blue)**: dips then recovers — captured into Bs, then released back
- **Bs trapped cars (green)**: rises then falls — accumulates as Bf is captured, dissipates as released
- **Bf+Bs combined (black dashed)**: nearly horizontal — Phase 1 is purely an internal exchange

> **Key insight**: Individual Bf and Bs masses fluctuate dramatically, but their sum only changes via boundary fluxes. This proves Phase 1 capture/release is a zero-sum internal reaction.

**Top-right: Mass conservation relative error (log scale)**

- **B(Bf+Bs) combined error (blue)**: ~3.5×10⁻³ (well below 1% threshold)
- **Class A error (orange)**: ~8×10⁻⁶ (extremely small — trucks are cleanly conserved)
- **Red dashed line**: pass/fail threshold at 1%

**Top-center: Phase 1 zero-sum visualization**

The most illuminating plot for proving conservation:

- Bf (blue) decreasing ↔ Bs (green) increasing; their sum (black dashed) stays almost flat
- This is a direct visual proof of the mathematical conservation law

**Bottom-left: Bottleneck cell zero-sum lemma (Class A)**

- **Blue solid**: total truck density at cell x=77 over time
- **Black dashed**: theoretically reconstructed from boundary fluxes
- Near-perfect overlap proves that internal speed transitions are zero-sum — only fluxes entering/leaving the cell change total density

**Bottom-right: Lane mass fraction over time**

Verifying bounded lateral redistribution:

- Three lanes' mass fractions change very little over time (std < 5%)
- Lane changes effectively redistribute vehicles without runaway concentration in any single lane

**Bottom-right: Generator diagnostic error over time**

Self-reported mass error from the simulation engine, staying well below 0.3% (far better than the 1% threshold).

---

## 9. V6 — Stiffness & Operator Splitting (`V6_stiffness.png`)

### Figure Explanation

"Stiffness" is a numerical analysis concept: when some processes in an equation are orders of magnitude faster than others, standard numerical methods become unstable. This section validates the model's extreme stiffness and the solutions deployed.

**Top-left: Stiffness ratio over time (log scale)**

- **Stiffness ratio S** = max deceleration rate / advection characteristic frequency
- **Initial S ≈ 1.57×10⁷** (100× larger than the threshold 10⁵!)
- **Peak S ≈ 2.2×10¹⁵** (extreme — the fastest process is a trillion times faster than the slowest)

> **Plain English**: The braking reaction is so fast compared to forward vehicle movement that a standard time-stepping method would need Δt < 10⁻¹⁵ seconds to remain stable — completely impractical. This is why the Thomas implicit algorithm is essential.

**Top-right: Timescale separation**

- **Orange line (τ_relax)**: deceleration reaction timescale — extremely short (nanosecond range)
- **Blue horizontal (τ_adv)**: advection timescale = Δx/v_max = 0.667 s
- **Gray horizontal (Δt)**: actual timestep = 0.5 s
- All three differ by up to 15 orders of magnitude

**Top-right: Explicit vs. implicit one-step result (Class A)**

The most dramatic comparison plot:

- **Gray bars (initial)**: trucks concentrated at low speed bins
- **Blue bars (Thomas implicit)**: physically sensible distribution after one step
- **Orange bars (explicit Euler)**: negative values and overshooting appear immediately (divergence beginning)

**Bottom-left: Explicit Euler 5-step divergence trajectory**

Step-by-step illustration of how the explicit method fails:

- Light red (t+0Δt) → dark red (t+5Δt): curves become increasingly distorted
- After 5 steps: max|f| ≈ 5.5×10²³ — physically meaningless

> This visually demonstrates why the Thomas implicit algorithm is non-negotiable: it stays stable for any stiffness ratio, while the explicit method fails within 5 steps.

**Bottom-center: φ(z) function stability curve**

Numerical safety verification for Phase 1's exact integral:

- **Horizontal axis (log scale)**: integration parameter z = σ·Δt (from 10⁻¹⁵ to 10⁶)
- **Curve**: φ(z) = (1-e⁻ᶻ)/z — monotonically decreasing from 1 to 0
- No NaN or Inf anywhere across 21 orders of magnitude
- **Orange vertical line**: current simulation's z value

**Bottom-right: Phase 1 σ/μ spatial distribution**

- **Green (capture rate σ)**: peaks at the truck bottleneck zone (red shading)
- **Orange (release rate μ)**: elevated at the Bf injection zone (blue shading)

---

## 10. V7 — Capture/Release Reactions (`V7_reactions.png`)

### Figure Explanation

This is the **brand-new** validation module in m3+m4, dedicated to verifying the Bf↔Bs moving bottleneck reaction mechanism.

**Top-left: φ(z) function — safe implementation**

Numerical stability of the exact matrix exponential across 20 orders of magnitude:

- Curve spans z = 10⁻¹⁴ to z = 10⁶
- **At z=0: φ=1** (Taylor expansion substituted to avoid 0/0)
- **At z→∞: φ→0** (physically: infinitely fast capture converts all Bf to Bs)
- Orange dots: all test points including extreme values — zero numerical anomalies

> **Implementation detail**: Direct computation of (1-e⁻ᶻ)/z fails at z=0 because both numerator and denominator are zero. The `safe_phi` function uses a conditional: φ=1 when z<10⁻¹², otherwise uses `-expm1(-z)/z` for precision.

**Top-center: Escape gate G and entrapment factor E (analytical curves)**

- **G(Ω) = (Ω/ρ_max)^η_lat**: as the adjacent lane fills up, the escape gate "closes"
- **E = G_left × G_right**: blocked on both sides = maximally trapped
- Empty lane (Ω=0) → G=0, E=0: freely escapable
- Jammed lane (Ω=ρ_max) → G=1, E=1: fully trapped

**Top-right: E_trap and Ω spatial correlation at t=0**

- **Orange solid (left axis)**: entrapment factor E along the road
- **Blue dashed (right axis)**: occupancy density Ω along the road
- **Near-identical shapes**: validates E indeed grows monotonically with Ω (correlation r=0.90)

**Bottom-left: σ/μ time series at injection zone (x=65)**

- **Green (capture rate σ)**: highest at t≈0 when the fast Bf platoon hits the bottleneck, decays as Bs accumulates
- **Orange (release rate μ)**: rises as Bs builds up and lane gaps open
- Both non-negative throughout → necessary condition for physical conservation verified

**Bottom-center: max f^(Bs)[i>i_thr] over time**

Verification of the acceleration blockade as a global invariant:

- **Trapped cars can never exceed speed v[i_thr] = 14 m/s**
- All values precisely zero (machine precision ~10⁻²⁰) at all times
- Confirms Phase 2 algebraic projection enforces the constraint at every single step

**Bottom-right: Bf/Bs mass exchange (whole road)**

Final proof of Phase 1 zero-sum conservation:

- **Bf (blue)** + **Bs (green)** = **B total (black dashed, nearly constant)**
- Bf decreases ↔ Bs increases in lockstep; their sum barely changes
- Combined error 3.5×10⁻³ vs. Bf-alone error 35% — ratio of 1:100 confirms precise internal zero-sum exchange

---

## 11. Core Research Claims Summary

| # | Claim | Key Checks | Verdict |
|---|-------|-----------|---------|
| **Claim 1** | Softplus lane-changing is smoother than Hard-max (C∞ continuity) | V4-a/b/e | **[Reliable]** |
| **Claim 2** | Singular barrier B(Ω) forms an impenetrable capacity ceiling | V2-c/f, V1-a | **[Reliable]** |
| **Claim 3** | 4-phase Lie-Trotter + Thomas algorithm resolves extreme stiffness | V6-a/c/d/f | **[Reliable]** |
| **Claim 4** | Global mass conservation theorem (Theorem 1) | V5-a/b/c/d | **[Reliable]** |
| **Claim 5** | FVM positivity and shock formation (Global Godunov limiter) | V3-a/b/d/e | **[Reliable]** |
| **Claim 6** | Moving bottleneck capture/release conservation (Phase 1 zero-sum) | V7-a/b/c/d | **[Reliable]** |
| **Claim 7** | Bs constraint system integrity (acceleration blockade + lateral immobilization) | V2-g, V4-f, V7-c | **[Reliable]** |

### Final Numerical Summary

```
Total checks:          41/41  ALL PASSED
Core research claims:   7/7   ALL RELIABLE
Simulation duration:    250 s (500 steps × 0.5 s/step)
Data volume:            512 MB (3 classes × 15 speed bins × 150 cells × 3 lanes)
Validation runtime:     3.3 s
Peak stiffness ratio:   S_max = 2.2×10¹⁵  (demonstrates necessity of implicit solver)
Mass conservation:      max relative error < 0.35% (Bf+Bs combined)
Phase 1 precision:      Combined Bf+Bs error / Bf-alone error = 1:100  (zero-sum is exact)
```

---

## Appendix: Key Parameter Reference

| Parameter | Symbol | Value | Physical Meaning |
|-----------|--------|-------|-----------------|
| Road capacity | ρ_max | 0.15 PCE/m | Maximum 0.15 PCE units per meter |
| Cell length | Δx | 20 m | Each spatial cell is 20 meters |
| Timestep | Δt | 0.5 s | Each step advances 0.5 seconds |
| Maximum speed | v_max | 30 m/s | ~108 km/h |
| Truck speed cap | v_A_ff | 14 m/s | ~50 km/h |
| Speed bins | N | 15 | 2, 4, ..., 30 m/s (even spacing) |
| CFL number | — | 0.75 | Must be ≤ 1 for numerical stability |
| Truck η exponent | η^(A) | 4.5 | Barrier steepness (larger = harder wall) |
| Base capture rate | σ_0 | 0.5 Hz | Baseline Bf→Bs trapping rate |
| Base release rate | μ_0 | 0.3 Hz | Baseline Bs→Bf escape rate |
| Entrapment exponent | η_lat | 2.0 | Escape gate function power |
| PCE weights | w | [2.5, 1.0, 1.0] | Trucks count 2.5× more than cars |
| Truck β coefficient | β[0,0] | 0.12 m⁻¹ | Truck-truck collision kernel |
| Car-truck β | β[1,0] | 0.06 m⁻¹ | Car following truck collision kernel |
