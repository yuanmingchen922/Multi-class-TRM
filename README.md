# Autonomous Multiclass TRM

A kinetic-theory-based multiclass traffic flow model with strict mass conservation, spatial FVM discretization, and an operator-splitting scheme for resolving extreme ODE stiffness.

## Overview

This repository contains the theoretical formulation, benchmark dataset, and numerical validation framework for the Autonomous Multiclass Traffic flow model (TRM). The model tracks the joint distribution of vehicle density across speed categories, spatial cells, and lanes for multiple vehicle classes simultaneously.

The state variable $f_{i,x,l}^{(m)}$ represents the density of vehicle class $m$ traveling at speed $v_i$ in cell $x$, lane $l$. The model is governed by three coupled physical processes:

1. Longitudinal spatial advection via Finite Volume Method (FVM)
2. Lateral lane-changing dynamics with a Softplus smoothing operator
3. Internal kinematic speed transitions with a singular asymptotic capacity barrier

## Repository Structure

```
code/
    Multi-class_TRM.tex          Model formulation (theory)
    Benchmark Dataset.json       Grid parameters and initial conditions
    generate_dataset.py          Simulation and dataset generator
    multiclass_trm_benchmark_500mb.h5   Benchmark dataset (514.7 MB, 800 timesteps)
    V1_occupancy.py              Validation: effective occupancy constraints (Section 2)
    V2_kinematics.py             Validation: transition rates and singular barrier (Section 3)
    V3_fvm.py                    Validation: FVM flux and shockwave formation (Section 4)
    V4_lateral.py                Validation: Softplus lane-changing mechanics (Section 5)
    V5_mass.py                   Validation: global mass conservation theorem (Section 7)
    V6_stiffness.py              Validation: stiffness ratio and operator splitting (Sections 8-9)
    run_all.py                   Master validation runner
    results.md                   Full validation report with numerical results
    validation_summary.json      Machine-readable validation summary
    figures/                     Output figures from validation modules
    project_overview.md          Project structure and validation mapping
```

## Key Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| X | 150 | Number of spatial cells |
| L | 3 | Number of lanes |
| N | 15 | Number of speed categories (2 to 30 m/s) |
| M | 2 | Vehicle classes (Passenger Car, Heavy Duty Truck) |
| rho_max | 0.15 PCE/m | Absolute jam density |
| CFL | 0.750 | Courant-Friedrichs-Lewy number |

## Validation Results

All 30 validation checks pass. The five core research claims are supported:

- Softplus lateral transition rates are demonstrably smoother than hard-max operators (C-infinity continuity confirmed, variance ratio 0.077)
- The singular asymptotic barrier B(Omega) at the benchmark bottleneck evaluates to 4,436,552 at Omega = 0.145, preventing capacity penetration
- Explicit Euler integration diverges to 3.89e+20 in 5 steps; Thomas algorithm produces a stable result with residual 9.21e-20
- Per-class mass conservation relative error is below 0.6% over 800 timesteps
- FVM positivity is maintained globally; shockwave propagation is observable in the space-time density map

## Requirements

```
python >= 3.9
numpy
h5py
matplotlib
```

## Usage

Generate the benchmark dataset:

```bash
python code/generate_dataset.py
```

Run all validation modules:

```bash
cd code
python run_all.py
```

## Reference

Yuan, Mingchen. "Autonomous Multiclass TRM: Strict Kinematics, Spatial FVM, and Resolution of Extreme Stiffness." (2026)
