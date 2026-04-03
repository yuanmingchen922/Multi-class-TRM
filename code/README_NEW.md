# Role & Context
You are an expert in Computational Fluid Dynamics (CFD), Traffic Flow Theory, and high-performance Python numerical simulation (NumPy/HDF5). 
I have an existing Python codebase that numerically simulates a macroscopic traffic flow model based on finite volume methods (FVM) and operator splitting. The current codebase represents the "m3" baseline model (2 vehicle classes, 3-phase operator splitting).
I have recently upgraded the mathematical theory to a unified "m3+m4" model, which introduces an "Isomorphic Moving Bottleneck Subclass Extension". 

Your task is to upgrade the numerical simulation engine (`generate_dataset.py`), the configuration file (`Benchmark Dataset.json`), and the validation scripts (`V1` to `V6`) to strictly reflect the new mathematical equations and the new 4-Phase calculation sequence.

# 1. Key Theoretical Upgrades (The "What")
The unified model transitions from 2 classes to 3 operational classes ($m \in \{A, Bf, Bs\}$):
- **Class A (Trucks)**: Heavy vehicles (m=0).
- **Class Bf (B-fast)**: Freely flowing passenger cars (m=1).
- **Class Bs (B-slow)**: Passenger cars trapped behind a moving truck bottleneck (m=2). Note: Both Bf and Bs share the same PCE weight $w^{(Bf)}=w^{(Bs)}$.

New Reactions (occur at the exact same speed index $i$):
- **Capture ($\sigma$)**: $Bf \to Bs$. Triggered when a car is blocked by trucks ahead AND adjacent lanes are blocked.
- **Release ($\mu$)**: $Bs \to Bf$. Triggered when trucks disperse OR an adjacent lane opens.

New Extended Operator Splitting (4 Phases instead of 3):
- **Phase 1**: Capture & Release (Exact Analytical Matrix Exponential)
- **Phase 2**: Internal Speed Changes (Dynamic Algebraic Projection + Thomas Algorithm)
- **Phase 3**: Lateral Lane-Changing (Explicit Euler)
- **Phase 4**: Spatial Advection (Explicit Euler + Godunov Limiter)

# 2. File-by-File Modification Instructions (The "How")

## 2.1 Update `Benchmark Dataset.json`
Add the following new parameters to the JSON configuration. You need to assign reasonable default values for testing:
- `v_A_ff`: Truck free-flow speed limit (determines $i_{thr}$).
- `eta_lat`: Lateral agility sensitivity for the escape gate $G$ (e.g., 2.0).
- `sigma_0_B`: Base capture rate.
- `mu_0_B`: Base release rate.
- `xi`: Lateral-awareness weight for adjacent truck pressure (e.g., 0.5).
- `R_A`: Truck dispersal scale for release rate.
- Update `M=3` (Classes A, Bf, Bs). Set PCE weights `w = [2.5, 1.0, 1.0]`.

## 2.2 Rewrite `generate_dataset.py` (The Core Engine)

**Initialization:**
- Change the shape of `f` to `(M=3, N, X, L)`.
- Initialize `f[1]` (Bf) with passenger car data. Initialize `f[2]` (Bs) as zeros.

**New Phase 1: Exact Capture and Release (Implement before Kinematics)**
- Calculate Kinetic Exposure $\mathcal{A}_{x,l}^{(i)}$ (Eq 8). Use NumPy broadcasting and `np.where(v_i >= v_k)` to implement the indicator function $\mathbf{1}_{v_i \ge v_k}$.
- Calculate Lateral Escape Gate $G_{x, l \to l'}$ (Eq 9). Ensure boundary conditions $G_{1 \to 0} = 1$ and $G_{L \to L+1} = 1$.
- Calculate Entrapment Factor $E_{x,l} = G_{left} \times G_{right}$.
- Calculate Total Capture Rate $\sigma_{x,l}^{(i)}$ (Eq 10).
- Calculate Effective Truck Presence $\tilde{\mathcal{S}}_{x,l}$ (Eq 11) and Release Rate $\mu_{x,l}$ (Eq 12).
- **Exact Integration (Eq 18, 19)**: 
  - Calculate $S^{(i)} = \sigma + \mu$. 
  - **CRITICAL**: To prevent $0/0$ division when $S^{(i)} \to 0$, implement the entire function $\phi(z) = (1-e^{-z})/z$. Use `scipy.special.expm1` or `np.expm1` (i.e., `phi = -np.expm1(-S * dt) / S`, handle `S < 1e-12` safely by returning `1.0`).
  - Update `Bf` and `Bs` simultaneously.

**New Phase 2: Dynamic Algebraic Projection & Kinematics**
- **Algebraic Projection (Eq 20)**: Before calculating $\lambda$, find the dynamic bottleneck speed $\kappa_{x,l}^* \le i_{thr}$ (highest speed index where $f^{(A)} > 0$). Instantly sum all $f^{(Bs)}$ for $j > \kappa^*$ and add it to $f^{(Bs)}$ at $\kappa^*$. Set $f^{(Bs)}$ at $j > \kappa^*$ to 0.
- **Constraints on Bs**: For class `Bs` (m=2), force $\lambda_{i \to i+1}^{(Bs)} = 0$ for $i \ge i_{thr}$ (Acceleration Blockade).
- Run the existing Semi-implicit Thomas algorithm.

**Update Phase 3: Lateral Lane-Changing**
- Absolute Lateral Immobilization: For class `Bs` (m=2), mathematically freeze all lane-changing: `gamma_left[2] = 0` and `gamma_right[2] = 0`.
- Classes A and Bf calculate $\gamma$ using the existing Softplus formulation.

**Update Phase 4: Spatial Advection**
- All classes (A, Bf, Bs) advect using their actual instantaneous speed $v_i$.
- Keep the Godunov flux limiter implementation intact.

## 2.3 Update Validation Scripts (`V1` to `V6`)
The dimension of $M$ is now 3. You must modify the slicing and summation logic:
- `V1_occupancy.py`: Ensure $\Omega$ correctly sums over `m=0, 1, 2` with updated weights `w`.
- `V5_mass.py`: The mass of "Passenger Cars" is now the sum of `Bf` and `Bs`. Prove that `sum(Bf + Bs)` is conserved and equals the boundary fluxes. Verify that Phase 1 (Capture/Release) is a perfect zero-sum exchange: $\sum (\mu f^{(Bs)} - \sigma f^{(Bf)}) \approx 0$.
- `V6_stiffness.py`: Ensure Phase 1 and Phase 2 are included in the 1-step explicit vs implicit stability comparison.

## 2.4 Create a New Validation Script: `V7_reactions.py`
Write a new script `V7_reactions.py` dedicated to the new Extension:
1. **[V7-a] Exact Analytical Stability**: Prove that when $S \to 0$ (empty road), the $\phi(z)$ function safely evaluates to 1 and no `NaN` or `Inf` occurs.
2. **[V7-b] Escape Gate Topology**: Prove $E_{x,l} = 0$ if at least one adjacent lane is empty, preventing capture.
3. **[V7-c] Algebraic Projection Effectiveness**: Prove that after Phase 2 projection, `max(f^{(Bs)}[i > i_{thr}]) == 0` (no high-speed trapped cars exist before advection).

# 3. Coding Constraints & Standards
- **Strict Vectorization**: Do NOT use Python `for` loops over spatial grids (`X`) or lanes (`L`). Use NumPy broadcasting (e.g., `np.einsum`, `[:, np.newaxis]`) exclusively.
- **Data Types**: Enforce `np.float64` strictly.
- **HDF5 Persistence**: Ensure the new shape `(T, 3, N, X, L)` is correctly chunked and saved without bloating the memory footprint unnecessarily. Update the HDF5 metadata attributes to reflect the 4-phase sequence.