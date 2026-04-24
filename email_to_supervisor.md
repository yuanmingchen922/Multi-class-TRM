**Subject:** Two IC design questions — main benchmark & no-bottleneck baseline

Dear Professor [Name],

I'd like to check with you on two initial-condition (IC) changes I made this week. Both are in the asymmetric-kinematics (vA) model. Details and my concerns below.

---

## 1. Main benchmark — truck density reduced

**Change.** In `generate_dataset.py`, the truck initial density at cells 74–79:

| | Before | After |
|---|---|---|
| `f[0, 0, 74:80, :]` | 0.058 | **0.040** |
| Initial Ω (= 2.5 × ρ_A) | 0.145 | **0.100** |
| Fraction of ρ_max (=0.15) | 97% | **67%** |

**Reason.** With Ω ≈ 0.145, the available downstream space was only 0.005 PCE/m (3% of ρ_max). The bottleneck was effectively a sealed wall: Godunov flux was nearly zero, almost no cars could pass through, and the rarefaction fan downstream of the bottleneck was never visible in the Hovmöller.

With Ω = 0.100, the bottleneck is still a strong obstacle (67% of jam density) but leaves 33% free space downstream, so meaningful flux can pass and the rarefaction structure becomes observable in both density and speed Hovmöller plots.

**Question.** Is Ω = 0.100 acceptable as the benchmark truck density, or would you prefer a different value? All 41/41 validation checks still pass.

---

## 2. No-bottleneck baseline (V_nobot) — IC and simulation time

**Change.** In `generate_dataset_nobot.py`:

| | Before | After |
|---|---|---|
| Sparse side (cells 0–74) | ρ = 0.010, v = 30 m/s | ρ = **0.005**, v = 30 m/s |
| Jam side (cells 75–149) | ρ = 0.070, v = 2 m/s | ρ = **0.120**, v = 2 m/s |
| Simulation time | 250 s (500 steps) | **100 s (200 steps)** |

**Reason for the IC change.** The old "jam" at ρ = 0.070 was only 47% of ρ_max. The downstream acceleration filter allowed ~80% acceleration there, so cars immediately accelerated out and the jam dissolved too fast to be meaningful. With ρ = 0.120 (80% of ρ_max) the jam is sustained and we get a true rarefaction fan.

**Reason for the shorter simulation.** On our 150-cell ring road with v_max = 30 m/s, fast cars wrap around the ring in ~100 s. Running 250 s produced overlapping waves (shock + wrap-around) that were visually chaotic. 100 s gives a clean, single-pass Riemann problem.

**My concern with the current V_nobot figure.** The jam region (75 cells wide) stays predominantly red in the Bf speed Hovmöller. This is physically correct — the rarefaction propagates backward from the leading edge only as fast as cars ahead move away — but visually it suggests "nobody can accelerate", which misrepresents the model.

**Three options I'm considering:**

- **(a)** Shrink the jam to 25 cells (cells 125–149) so the rarefaction fully dissipates it within 100 s. This becomes an "isolated jam dissipation" test rather than a classic Riemann problem, but the acceleration dynamics become clearly visible.

- **(b)** Keep the current 75-cell jam but extend the simulation to ~200 s. The jam will partially dissipate, but ring-wrap effects will re-enter.

- **(c)** Switch the V_nobot boundaries from ring-road to open outflow. This gives a clean Riemann problem with no wrap-around, at the cost of inconsistency with the main benchmark (which is ring road).

**Question.** Which of (a), (b), (c) do you prefer, or do you have another suggestion?

---

Happy to run whichever variant you recommend and share the figures. All code is on GitHub at `yuanmingchen922/Multi-class-TRM`, main branch.

Best regards,  
Mingchen
