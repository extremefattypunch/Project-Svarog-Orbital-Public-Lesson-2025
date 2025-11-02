<div align="center">

# Project Svarog Orbital Dynamics & Solar Sail Propagation

**High‑fidelity, GPU‑ready, allocation‑free flight dynamics kernels for solar‑sail and conventional spacecraft propagation in Julia.**

_Authoritative ODE right‑hand side. Unified CPU/GPU physics. Extensible force & torque architecture._

</div>

---

## Table of Contents
1. Vision & Motivation
2. Core Design Principles (Non‑Negotiables)
3. High‑Level Architecture
4. Module Deep Dive
5. API Documentation & Usage Guide
6. State Representation & Mathematical Conventions
7. Physics Models (Forces & Attitude) – Detailed Derivations
8. GPU vs CPU Execution Model
9. Installation & Environment Setup (All Platforms)
10. Quick Start (90‑Second Run)
11. Example Scenarios & Recipes
12. Extended Setup Examples & Specialized Use Cases
13. Extending the System (Adding a New Force / Torque)
14. Performance, Allocation Strategy & Benchmarking
15. Validation Philosophy & Testing (Current Status)
16. Roadmap & Future Enhancements
17. Troubleshooting Guide
18. Contributing & Code Style
19. Glossary
20. References

---

## 1. Vision & Motivation
Project Svarog is a student‑led mission concept targeting the development of a solar‑sailing CubeSat with long‑range ambitions, ultimately exploring interstellar precursor dynamics. This repository implements a modern, type‑stable, *GPU-friendly* dynamics core in Julia. The goal: **a single authoritative ODE RHS** powering all translational + (future) rotational dynamics, with clear boundaries between real‑time physics kernels and CPU‑only tooling (SPICE, I/O, plotting, analysis).

Key outcomes:
* Deterministic physics core (no hidden side effects, no I/O, no ephemeris calls mid‑step).
* Parametric in `T<:AbstractFloat` (switch `Float64` ↔ `Float32` seamlessly for GPU).</br>
* Fast and allocation‑free inside the integrator loop.
* Explicit, composable models (gravity, SRP, drag, albedo now; torques ready to plug in later).

---

## 2. Core Design Principles (Non‑Negotiables)
| Principle | Summary | Rationale |
|-----------|---------|-----------|
| Single Source of Truth | `dynamics!` in `Propagator.jl` is *the* ODE RHS | Prevent physics drift across CPU/GPU or tools |
| GPU-Safe Kernels | No SPICE, Dates, I/O, heap allocs in hot path | Determinism & performance |
| Parametric Types | All numerics use `T<:AbstractFloat` | Mixed precision & performance tuning |
| Separation of Concerns | Forces (Gravity.jl), Attitude (Attitude.jl), Ephemeris (SpiceUtils.jl) | Maintainability |
| Legacy Isolation | Old APIs (e.g. `gravity_acceleration`) clearly tagged | Gradual migration |
| Explicit Data Contracts | `DynamicsParams` packs immutable runtime constants | GPU transfer safety |
| Idempotent Setup | Precompute external data *before* propagation | Reproducibility |

---

## 3. High‑Level Architecture
```
+-------------------+        +------------------+
|  SpiceUtils (CPU) |  --->  |  Precomputed Sun  |  (SVector{3,T})
+-------------------+        +------------------+
           |                             \
           v                              v
 +----------------+        +------------------------------+
 | DynamicsParams |  --->  |    dynamics!(...) ODE RHS    |  ---> Integrator (Tsit5 / GPU Tsit5)
 +----------------+        +------------------------------+
           ^                              |
  +--------|---------+          +---------|------------------------+
  |  Gravity Kernels |          | Attitude (quat normalize / rot)  |
  +------------------+          +----------------------------------+
```

### Data Flow
1. (Optional) Ephemeris preprocessed (Sun position) on CPU.
2. `Propagator.make_params` condenses model selections into `DynamicsParams`.
3. `dynamics!` executes deterministic kernels – gravity, SRP, drag – and attitude kinematics.
4. Integrator (CPU or DiffEqGPU) advances state; callback keeps quaternion unit norm.

---

## 4. Module Deep Dive
| Module | Purpose | GPU Safe? | Notes |
|--------|---------|-----------|-------|
| `Constants.jl` | Scalar + vector constants (via functions) | Yes | Removed unused legacy exports (e.g. `OMEGA_EARTH_VEC`) |
| `CentralBody.jl` | Environment structs: `EarthEnv`, `SunEnv` | Yes | Pure POD; no time‑varying state |
| `Attitude.jl` | Quaternion ops (normalize, derivative, rotations) | Yes | Canonical – do **not** reimplement elsewhere |
| `Gravity.jl` | Force models + low‑level kernels | Yes (except optional cache use for SRP model) | SRP, drag, albedo, gravity (point mass + J2) |
| `Torque.jl` | Torque models (SRP torque, gravity gradient, etc.) | Partially | Some paths CPU-only if using ephemeris cache |
| `Propagator.jl` | ODE definition & integration orchestration | Yes (hot path) | `DynamicsParams` & `dynamics!` authoritative |
| `Illumination.jl` | Shadow / eclipse factor | Yes | Earth-only wrapper + generic version |
| `SpiceUtils.jl` | Ephemeris caching, SPICE I/O | No | *Never* call inside `dynamics!` |
| `Plotting.jl`, `Analysis.jl` | Post-processing & visualization | No | Free to allocate / use SPICE |
| `legacy/` | Historical scripts & prototypes | No | Reference only |

---

## 5. API Documentation & Usage Guide

This section enumerates the core public API surface. Anything not mentioned here should be treated as internal or provisional. Legacy symbols are explicitly marked.

### 5.1 Core Types

| Type | Location | Purpose | Notes |
|------|----------|---------|-------|
| `StateVector{T}` | `OrbitState.jl` | Holds `r,v,q,ω` | Quaternion auto‑normalized on construction |
| `PropagatorType{T,CB}` | `Propagator.jl` | Container configuring models, tolerances, inertia, callbacks | Add forces via `add_force!` |
| `DynamicsParams{T,CB}` | `Propagator.jl` | Packed, GPU‑safe ODE parameters | Built by `make_params` (not usually hand‑made) |
| `EarthEnv{T}` / `SunEnv{T}` | `CentralBody.jl` | Immutable environment constants | POD, GPU‑safe |
| `GravityModel{T}` | `Gravity.jl` | High‑level gravity force (degree selects J2) | Wraps kernel `gravity_accel` |
| `SRPModel{T}` | `Gravity.jl` | Solar radiation pressure | Optional ephemeris cache (CPU‑only when used) |
| `AtmosphericDragModel{T}` | `Gravity.jl` | Free‑molecular Earth drag | Earth‑only assumption |
| `AlbedoModel{T}` | `Gravity.jl` | Simplified reflected pressure | Experimental |
| `SRPParams{T}` / `DragParams{T}` | `Gravity.jl` | Kernel parameter packs | Passed to *_accel functions |
| `AbstractForceModel{T}` | `Gravity.jl` | Supertype for force polymorphism | Implement `acceleration` |
| `AbstractTorqueModel{T}` | `Torque.jl` | Supertype for torque sources | Implement `torque` |

### 5.2 Core Functions (Authoritative)

| Function | Signature (simplified) | Returns | CPU/GPU | Notes |
|----------|------------------------|---------|--------|-------|
| `propagate` | `(prop, state0, (t0,tf); kwargs...)` | `TrajectoryData` | Both | Main entry point |
| `dynamics!` | `(du,u,p,t)` | `nothing` | Both | ODE RHS (do not fork) |
| `make_params` | `(prop; sun_pos=...)` | `DynamicsParams` | CPU build / GPU use | Precompute before solve |
| `gravity_accel` | `(body, r; degree=0)` | `SVector{3,T}` | Both | Point mass + optional J2 |
| `srp_accel` | `(params, r, q, sun_pos)` | `SVector{3,T}` | Both | Requires illumination call internally |
| `drag_accel` | `(params, r, v, q, body, t)` | `SVector{3,T}` | Both | Earth atmosphere approximation |
| `illumination_factor` | `(r_sc, sun_pos, body)` | `T∈[0,1]` | Both | General shadow model |
| `shadow_factor` | `(r_sc, sun_pos)` | `T∈[0,1]` | Both | Earth convenience wrapper |
| `quat_normalize` | `(q)` | `SVector{4,T}` | Both | Use everywhere for robustness |
| `quat_derivative` | `(q, ω)` | `SVector{4,T}` | Both | Attitude kinematics |
| `quat_to_rotmat` | `(q)` | `SMatrix{3,3,T}` | Both | Body→inertial |

### 5.3 Legacy / Transitional Functions

| Function | Status | Replacement |
|----------|--------|-------------|
| `gravity_acceleration(μ,J2,Re,r)` | LEGACY | `gravity_accel(body,r; degree)` |
| `srp_acceleration(area,mass,C_r,r,q,sun)` | LEGACY | `srp_accel(SRPParams(...), r, q, sun)` |
| `quaternion_to_rotation_matrix(q)` | Wrapper | `quat_to_rotmat(q)` |

### 5.4 Constructing a Propagator
```julia
using ProjectSvarogOrbital
using StaticArrays

cb = CentralBody.EarthEnv{Float64}()
models = [
  GravityModel(cb; degree=2),
  SRPModel(4.0, 8.0, 1.6),
  AtmosphericDragModel(1.5, 8.0, 2.2)  # optional
]
prop = PropagatorType(cb; models=models, degree=2)

# Keplerian initial conditions
a   = 7000e3
e   = 0.001
i   = 63.4
Ω   = 0.0
ω   = 0.0
ν   = 0.0
state0 = StateVector(a, e, i, Ω, ω, ν, cb.mu)

traj = propagate(prop, state0, (0.0, 86400.0))
```

### 5.5 Adding / Composing Forces After Construction
```julia
add_force!(prop, AlbedoModel(4.0, 8.0, 0.3))
traj2 = propagate(prop, state0, (0.0, 2*86400.0))
```

### 5.6 Direct Kernel Use (Low‑Level)
Useful for batch evaluation, Monte Carlo sensitivity, or custom integrators.
```julia
using StaticArrays
cb = CentralBody.EarthEnv{Float64}()
r  = SVector(7000e3, 0.0, 0.0)
v  = SVector(0.0, 7_500.0, 1_000.0)
q  = SVector(1.0, 0.0, 0.0, 0.0)
sun = SVector(1.495978707e11, 0.0, 0.0)

a_grav = gravity_accel(cb, r; degree=2)
a_srp  = srp_accel(SRPParams(4.0, 8.0, 1.6), r, q, sun)
a_drag = drag_accel(DragParams(1.5, 8.0, 2.2), r, v, q, cb, 0.0)
a_total = a_grav + a_srp + a_drag
```

### 5.7 Building Parameters for Custom Integrators
If you want to bypass `propagate` and call DifferentialEquations.jl manually:
```julia
using DifferentialEquations
p = make_params(prop; sun_pos = SVector(1.495978707e11, 0, 0))
u0 = vcat(state0.r, state0.v, state0.q, state0.ω)
prob = ODEProblem(dynamics!, u0, (0.0, 86400.0), p)
sol = solve(prob, Tsit5(); reltol=prop.tol, abstol=prop.tol)
```

### 5.8 Quaternion Safety Pattern
When injecting attitude changes (e.g. control law):
```julia
q_new = Attitude.quat_normalize(q_old + δq)  # NEVER skip normalization after modification
```

### 5.9 Extending with a New Force Model – Minimal Template
```julia
struct MyPhotonForce{T} <: AbstractForceModel{T}
  area::T; mass::T; coeff::T
end

function acceleration(m::MyPhotonForce{T}, r::SVector{3,T}, v::SVector{3,T},
            q::SVector{4,T}, t::T) where {T<:AbstractFloat}
  n = Attitude.quat_to_rotmat(q) * PLATE_NORMAL_BODY(T)
  flux = T(1361.0) * (AU(T)/norm(r))^2
  pres = flux / C_LIGHT(T)
  return pres * m.area * m.coeff / m.mass * n
end
```
Add it to a propagator via `add_force!(prop, MyPhotonForce(area, mass, coeff))`.

### 5.10 Zero‑Allocation Checklist (Manual)
```julia
using BenchmarkTools
du = similar(u0)
@btime dynamics!($du, $u0, $p, 0.0)  # Expect ≈ 0 allocations
```
If allocations appear:
1. Inspect new force model for dynamic `Vector` creation.
2. Ensure all temporaries are `SVector` / `SMatrix`.
3. Avoid splatting varargs in performance critical loops.

### 5.11 Common Pitfalls
| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Forgot quaternion normalize | Drift / NaNs after long run | Always call `quat_normalize` |
| Added SPICE call in kernel | GPU failure or huge slowdown | Move to setup phase (`make_params`) |
| Used random numbers | Non‑deterministic results | Precompute seeds off‑kernel or remove |
| Allocating large arrays each step | Performance cliff | Reuse buffers or rely on SVector |

### 5.12 Versioning & Stability
| Stability Level | Components |
|-----------------|------------|
| Stable | `PropagatorType`, `propagate`, `gravity_accel`, `srp_accel`, `drag_accel`, `StateVector`, attitude functions |
| Beta | Albedo model, torque models (not in RHS), illumination smoothing parameters |
| Experimental | Any force tagged with `# LEGACY` comment |

For new API proposals, open an issue referencing this section so we can classify stability level early.

---

## 6. State Representation & Mathematical Conventions
The propagated state (size 13) is:
```
u = [ r(1:3); v(4:6); q(7:10); ω(11:13) ]
```
Where:
* `r` – Position in inertial frame [m]
* `v` – Inertial velocity [m/s]
* `q = (w,x,y,z)` – Unit quaternion (body→inertial)
* `ω` – Body angular velocity [rad/s] (currently kinematic placeholder; ω̇=0 for now)

Quaternion derivative:
$$\dot{q} = \tfrac{1}{2} \; \Omega(\omega) \, q$$
with
$$\Omega(\omega) = \begin{bmatrix}0 & -\omega_x & -\omega_y & -\omega_z\\ \omega_x & 0 & \omega_z & -\omega_y\\ \omega_y & -\omega_z & 0 & \omega_x \\ \omega_z & \omega_y & -\omega_x & 0\end{bmatrix}.$$

Rotation matrix from quaternion uses the standard normalized quadratic form. Always pass via `Attitude.quat_to_rotmat` to avoid divergence.

---

## 7. Physics Models (Forces & Attitude) – Detailed Derivations

### 6.1 Central Body Gravity (Point Mass + J2)
Base acceleration:
$$\mathbf{a}_{\mu} = -\mu \frac{\mathbf{r}}{\|\mathbf{r}\|^3}$$

J2 perturbation term (z along spin axis):
$$\mathbf{a}_{J2} = \frac{3 J_2 \mu R_e^2}{2 r^5} \begin{bmatrix} x (5 z^2/r^2 - 1) \\ y (5 z^2/r^2 - 1) \\ z (5 z^2/r^2 - 3) \end{bmatrix}$$
We combine them in `gravity_accel(body, r; degree=...)` only if `degree ≥ 2` and `body.j2 > 0`.

### 6.2 Solar Radiation Pressure (SRP)
Flux scaling with distance:
$$P(r) = P_{1\text{AU}} \left(\frac{1\text{AU}}{d}\right)^2$$
Effective acceleration:
$$ \mathbf{a}_{\text{SRP}} = \frac{P \; A}{m c} (1 + C_r) \max(0, \cos \theta) \, I_{\text{illum}} \, \hat{n} $$
where \(\hat{n}\) is plate normal in inertial frame (`R * PLATE_NORMAL_BODY`). We clamp illumination using Earth's shadow factor.

### 6.3 Atmospheric Drag (Free Molecular Approximation)
Simplified density model uses tabulated exponential segments (1976 US Std Atmosphere slice). Relative wind accounts for Earth rotation:
$$\mathbf{v}_{rel} = \mathbf{v} - \boldsymbol{\Omega}_E \times \mathbf{r}$$
Sentman‑style blended coefficients produce drag & lift contributions; final acceleration:
$$\mathbf{a}_{\text{drag}} = \frac{A}{m} q_{dyn} (-C_D \hat{v} + C_L \hat{l})$$
with dynamic pressure \(q_{dyn}= \tfrac{1}{2} \rho v_{rel}^2\). Implemented allocation‑free in `drag_accel`.

### 6.4 Earth Albedo Pressure (Simplified)
Currently uses geometric inverse‑square scaling of the apparent Earth disc:

$$\left(\frac{R_E}{d}\right)^2$$

where $R_E$ is Earth's mean equatorial radius and $d = \|\mathbf{r}\|$. A simple specular (cos²) alignment is applied using the sail plate normal in inertial coordinates. Future improvements may incorporate: BRDF models, limb darkening, Earth IR re‑radiation, and eclipse (partial occultation) geometry.

### 6.5 Illumination / Shadow Model
Implements conical umbra / penumbra transition with smooth (`tanh`) blending. `shadow_factor` is an Earth‑only convenience wrapper; general form uses `illumination_factor(r_sc, sun_pos, body)`.

### 6.6 Attitude Kinematics
We track quaternion and (currently) hold angular velocity constant. Normalization enforced both inline (per RHS evaluation) and via a discrete renormalization callback to mitigate drift under finite precision.

---

## 8. GPU vs CPU Execution Model
| Aspect | CPU | GPU |
|--------|-----|-----|
| Integrator | `Tsit5()` | `GPUTsit5()` (DiffEqGPU) |
| Memory Model | Standard heap + stack | SArrays (stack‑like, avoids heap allocs) |
| Attitude Math | Same code path | Same (no divergence) |
| Ephemeris | Via SpiceUtils (pre-run) | Precomputed only – never in kernel |
| Debug Logging | Optional @debug | Usually suppressed (cost) |

To run on GPU you must ensure:
1. `DynamicsParams` contains only isbits fields.
2. No closures capturing non‑GPU objects inside `dynamics!`.
3. External data (e.g. Sun position) precomputed and passed as `SVector{3,T}`.

---

## 9. Installation & Environment Setup (All Platforms)

### 8.1 Prerequisites
| Component | Version | Notes |
|-----------|---------|-------|
| Julia | ≥ 1.10 (recommended) | Earlier 1.7+ likely works; performance improvements in newer releases |
| GPU (optional) | CUDA‑capable or ROCm | Requires DiffEqGPU backend availability |
| Git | Latest stable | For cloning |

### 8.2 Clone & Instantiate
```bash
git clone https://github.com/ImperialSpaceSociety/Project-Svarog-Orbital.git
cd Project-Svarog-Orbital
```
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### 8.3 Windows Notes
* Ensure Julia `bin` directory is on PATH.
* If using GPU: install CUDA toolkit matching driver. DiffEqGPU relies on CUDA.jl auto‑detection.

### 8.4 Linux Notes
* For NVIDIA: verify `nvidia-smi` works before launching Julia.
* Set `JULIA_NUM_THREADS` for multi-threaded CPU runs: `export JULIA_NUM_THREADS=$(nproc)`.

### 8.5 macOS Notes
* Apple Silicon: `Float32` may yield better GPU/Metal throughput (future backend considerations).

### 8.6 Development Mode
Add as a local dependency elsewhere:
```julia
using Pkg
Pkg.develop(path="/absolute/path/to/Project-Svarog-Orbital")
```

---

## 10. Quick Start (90‑Second Run)
```julia
julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
include("examples/earth_orbit.jl")  # or run from VS Code cell
```
Adjust orbital elements inside the example; run again to see updated trajectory. For SRP or drag examples, see `examples/earth_orbit_acc.jl` or `earth_ensemble_cpu.jl`.

---

## 11. Example Scenarios & Recipes
| File | Scenario | Highlights |
|------|----------|------------|
| `examples/earth_orbit.jl` | Baseline 2‑body propagation | Minimal kernel usage |
| `examples/earth_orbit_acc.jl` | Forces demonstration | Adds J2 / SRP / drag combos |
| `examples/earth_ensemble_cpu.jl` | Ensemble CPU run | Parameter sweeps |
| `examples/earth_ensemble_gpu.jl` | Ensemble GPU prototype | Scalability test |
| `examples/compare_cpu_gpu.jl` | Cross‑platform consistency | Float drift analysis |
| `examples/visualisation.jl` | Plotting workflow | CPU‑only analysis layer |

### Recipe: Add SRP to a Propagator
```julia
using ProjectSvarogOrbital
cb   = CentralBody.EarthEnv{Float64}()
prop = PropagatorType(cb; models=[GravityModel(cb; degree=2), SRPModel(4.0, 8.0, 1.6)])
state0 = StateVector(a, e, i, Ω, ω, ν, cb.mu)
traj = propagate(prop, state0, (0.0, 86400.0))
```

### Recipe: GPU Run (Prototype)
```julia
traj = propagate(prop, state0, (0.0, 86400.0); gpu=true)
```
Ensure all active force models are GPU‑safe (avoid torque models using ephemeris cache).

---
## 12. Extended Setup Examples & Specialized Use Cases

This section provides concrete problem setup snippets beyond the basic examples. They illustrate how to: (a) isolate specific physics, (b) prepare GPU ensembles, (c) run attitude‑only studies, (d) compare force model impacts, and (e) stage future heliocentric extensions.

### 12.1 Minimal Earth Gravity + J2 Only
```julia
using ProjectSvarogOrbital, StaticArrays, DifferentialEquations
T  = Float64
cb = CentralBody.EarthEnv{T}()
models = [GravityModel(cb; degree=2)]  # J2 enabled
prop = PropagatorType(cb; models=models, degree=2, tol=1e-9)

# Simple circular LEO
a,e,i,Ω,ω,ν = 7000e3, 0.0, deg2rad(51.6), 0.0, 0.0, 0.0
state0 = StateVector(a,e,i,Ω,ω,ν, cb.mu)
traj = propagate(prop, state0, (0.0, 3*86400.0))
```

### 12.2 Add SRP & Drag (Force Mix Toggle)
```julia
srp  = SRPModel(4.0, 8.0, 1.6)
drag = AtmosphericDragModel(1.5, 8.0, 2.2)
add_force!(prop, srp)
add_force!(prop, drag)
traj_forces = propagate(prop, state0, (0.0, 3*86400.0))
```

### 12.3 SRP On vs Off Comparison
```julia
prop_srp_on  = deepcopy(prop)
prop_srp_off = PropagatorType(cb; models=[GravityModel(cb; degree=2), drag])

traj_on  = propagate(prop_srp_on,  state0, (0.0, 10*86400.0))
traj_off = propagate(prop_srp_off, state0, (0.0, 10*86400.0))
Δr_final = traj_on.r[end] - traj_off.r[end]
println("SRP-induced terminal position delta (m): ", norm(Δr_final))
```

### 12.4 Drag Sensitivity Sweep (LEO Altitude Band)
```julia
alts = 650e3:50e3:900e3
function make_state(a_alt)
  StateVector(a_alt, 0.001, deg2rad(51.6), 0.0, 0.0, 0.0, cb.mu)
end
decay_rates = map(alts) do alt
  st = make_state(alt + 6371e3)
  sol = propagate(prop, st, (0.0, 2*86400.0))
  (alt, sol.a[end])  # Example: record semi-major axis evolution
end
```

### 12.5 GPU Ensemble (Prototype)
```julia
using DiffEqGPU
cb = CentralBody.EarthEnv{Float32}()
base = PropagatorType(cb; models=[GravityModel(cb; degree=2), SRPModel(4f0, 8f0, 1.6f0)], degree=2)
N = 512
states = [StateVector(7000e3 + 10*i, 0.0005f0, deg2rad(51.6f0), 0f0, 0f0, 0f0, cb.mu) for i in 0:(N-1)]
p = make_params(base; sun_pos = SVector{3,Float32}(1.4959787f11,0,0))
u0s = [vcat(st.r, st.v, st.q, st.ω) for st in states]
prob = ODEProblem(dynamics!, u0s[1], (0f0, 86400f0), p)  # Template for ensemble
ens  = EnsembleProblem(prob, prob_func = (prob,i,repeat)-> remake(prob, u0=u0s[i]))
sol  = solve(ens, GPUTsit5(), EnsembleGPUArray(), trajectories=N)
```

### 12.6 Attitude‑Only Kinematic Study (Freeze Translation)
Purpose: Examine quaternion evolution and (future) torque impacts without orbital motion.
```julia
using DifferentialEquations
cb = CentralBody.EarthEnv{Float64}()
prop_att = PropagatorType(cb; models=[GravityModel(cb; degree=0)], degree=0)  # GravityModel included but translation frozen
p = make_params(prop_att; sun_pos=SVector{3,Float64}(1.495978707e11,0,0))

# Initial quaternion & a non-zero angular velocity
ω0 = SVector{3,Float64}(0.05, 0.02, -0.01)
a,e,i,Ω,ω,ν = 7000e3,0.0,0.0,0.0,0.0,0.0
st = StateVector(a,e,i,Ω,ω,ν, cb.mu)
u0 = vcat(st.r, st.v, st.q, ω0)

function dynamics_attitude_only!(du,u,p,t)
  dynamics!(du,u,p,t)            # Compute full RHS
  # Zero translational derivatives (ṙ, v̇)
  @inbounds for k in 1:6
    du[k] = 0
  end
  # ω̇ remains zero (torque integration pending roadmap item)
end

prob = ODEProblem(dynamics_attitude_only!, u0, (0.0, 7200.0), p)
sol  = solve(prob, Tsit5())
```
Note: Once torque integration (ω̇ ≠ 0) is added, replace the zeroing logic with physical torque derivation.

### 12.7 Mixed Force Subset (SRP Only, Drag Off)
```julia
prop_srp_only = PropagatorType(cb; models=[GravityModel(cb; degree=2), SRPModel(4.0,8.0,1.6)], degree=2)
traj_srp = propagate(prop_srp_only, state0, (0.0, 5*86400.0))
```

### 12.8 Manual Quaternion Steering (Hypothetical Control Law)
```julia
function steer_quaternion!(u,t)
  q = @SVector u[7:10]
  # Simple spin about z by small incremental rotation (placeholder)
  δq = SVector(1.0, 0.0, 0.0, 0.002)  # Not a proper unit quaternion increment
  q_new = quat_normalize(q + δq)
  @inbounds for i in 1:4
    u[6+i] = q_new[i]
  end
end
```
Integrate with a DiscreteCallback applying `steer_quaternion!` every N seconds.

### 12.9 Placeholder: Heliocentric Transfer (Future Feature)
```julia
# FUTURE FEATURE: Requires generalized central body (e.g. Sun as primary)
# TODO: Define HelioEnv{T} with mu_sun, maybe planetary ephemeris pack.
# TODO: Implement third-body gravity (Earth, Jupiter influences) via precomputed arrays.
cb_sun = nothing  # Placeholder
models_helio = [] # Will include GravityModel(cb_sun), SRPModel(...), future planetary terms
# prob = ODEProblem(dynamics!, u0_helio, (t0, tf), params_helio)  # To be specified
```
Status: Deferred until roadmap items (Third‑body gravity & adaptive Sun vector) are complete.

### 12.10 Ephemeris Precomputation Pattern (CPU Side)
```julia
using .SpiceUtils  # CPU-only
sun_vec = SpiceUtils.sun_pos(J2000_ET_start)  # Returns SVector{3,T}
p = make_params(prop; sun_pos=sun_vec)
```
Guarantee this happens before any GPU solve to avoid illegal SPICE calls mid‑kernel.

### 12.11 Minimal Benchmark Harness
```julia
using BenchmarkTools
du = similar(u0)
@btime dynamics!($du, $u0, $p, 0.0)  # Expect ~0 allocations
```

### 12.12 Monte Carlo Constellation Spread (Concept)
```julia
N = 256
rand_phase = 2π .* rand(N)
states_mc = [StateVector(a, e, i, Ω, ω, νp, cb.mu) for νp in rand_phase]
u0s = [vcat(s.r, s.v, s.q, s.ω) for s in states_mc]
ens_prob = EnsembleProblem(prob, prob_func=(prob,i,_) -> remake(prob, u0=u0s[i]))
sols = solve(ens_prob, Tsit5(); trajectories=N)
```
Capture dispersion metrics by sampling final positions/velocities.

### 12.13 Debugging a Single Step (Kernel Introspection)
```julia
du = similar(u0)
dynamics!(du, u0, p, 0.0)
@show du[1:6]  # Translational derivatives
@show du[7:10] # Quaternion derivative
@show du[11:13] # Angular velocity derivative (currently zeros)
```

### 12.14 Force Model Isolation Harness
```julia
function isolate(model, state0, cb; span=(0.0, 3600.0))
  prop_iso = PropagatorType(cb; models=[model], degree=0)
  propagate(prop_iso, state0, span)
end
sol_srp_only = isolate(SRPModel(4.0,8.0,1.6), state0, cb)
```

---
**Reminder:** The authoritative path stays inside `dynamics!`. Alternate wrappers should call it rather than reimplement physics.


## 13. Extending the System (Adding a New Force)
1. Define a subtype: `struct MyForceModel{T} <: AbstractForceModel{T} ... end`.
2. Provide: `acceleration(model::MyForceModel{T}, r, v, q, t)` returning `SVector{3,T}`.
3. If it can be used in dynamics! frequently, factor a *kernel* version `myforce_accel(params, ...)` with a POD param struct.
4. Modify `make_params` only if you need a new flag / parameters group (avoid dynamic dispatch in the hot loop).
5. Confirm zero allocations: `@allocated dynamics!(du, u, params, t)` ≈ 0.

### Force Kernel Checklist
- No random number generation.
- No global constants mutated.
- No SPICE / file I/O / logging macros inside inner math.
- All intermediate small vectors as `SVector` to remain stack‑friendly.

---

## 14. Performance, Allocation Strategy & Benchmarking
* **SVector Everywhere:** Force kernels rely on `StaticArrays` for fixed-size math (3‑vectors, quaternions).
* **No Heap in Hot Path:** `dynamics!` constructs only `SVector{3,T}` / `SVector{4,T}`; no `Vector` growth.
* **Quaternion Normalization:** Done once per step; extremely low overhead.
* **J2 Branch:** Short‑circuited if `degree < 2` or `body.j2 == 0`.
* **Potential Optimization:** Precompute repeated scaling factors for large ensembles.

Benchmark suggestion (user‑run):
```julia
using BenchmarkTools
T = Float64
cb = CentralBody.EarthEnv{T}(); q = SVector{4,T}(1,0,0,0)
u = @SVector rand(T,13); du = similar(u)
p = DynamicsParams(cb, 2, false, SRPParams(0,1,1), false, DragParams(0,1,2.2), SVector{3,T}(1.5e11,0,0))
@btime dynamics!($du, $u, $p, 0.0)
```

---

## 15. Validation Philosophy & Testing (Current Status)
**Note:** Full test suite refactor is in progress; some tests are expected to fail during transitional architecture consolidation.

Planned test tiers:
| Tier | Goal | Example |
|------|------|---------|
| Unit | Kernel math correctness | J2 term sign & scaling |
| Property | Invariants | Energy drift (gravity only) |
| Cross-Platform | CPU vs GPU state difference < tolerance | Ensemble midpoint comparison |
| Allocation | Zero heap in dynamics! | `@allocated` checks |

If you contribute new physics: **include at least one unit + one property test**.

---

## 16. Roadmap & Future Enhancements
| Stage | Feature | Status |
|-------|---------|--------|
| A | Torque integration into `dynamics!` (ω̇) | Pending |
| B | Third‑body gravity (precomputed ephemerides) | Design |
| C | Adaptive Sun vector interpolation (piecewise) | Planned |
| D | Sail attitude control laws (closed loop) | Partial (Control module) |
| E | High‑fidelity atmosphere (NRLMSISE‑00 binding) | Planned |
| F | Advanced SRP (multi‑panel, optical properties) | Planned |
| G | Albedo + IR flux model rework | Planned |
| H | Validation vs external tools (GMAT, Orekit) | Planned |

---

## 17. Troubleshooting Guide
| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| Quaternion drifts | Custom code bypassing `quat_normalize` | Always use Attitude functions |
| GPU kernel fails | Non‑isbits field in params | Inspect `DynamicsParams` additions |
| SRP off by factor | Incorrect plate normal or sun distance | Check `PLATE_NORMAL_BODY` and AU scaling |
| Drag = 0 always | Altitude below/above table bounds | Confirm orbit alt range 100–1000 km for simplified model |
| Shadow always 1 | Geometry sign test failing | Verify r·sun_pos negative on night side |

---

## 18. Contributing & Code Style
1. Keep new physics GPU-safe unless explicitly CPU-only.
2. Prefer `SVector{3,T}` over `Vector{T}` in kernels.
3. Add docstrings for any public API.
4. Tag legacy code with `# LEGACY` instead of deleting immediately.
5. Run `Pkg.instantiate()` before committing (lock dependencies reproducibly).
6. Use descriptive PR titles: `feat: add third-body gravity kernel`.

### Minimal PR Checklist
- [ ] Added/updated docstrings
- [ ] No allocations in hot path (`dynamics!` diff inspected)
- [ ] GPU-safe (if intended)
- [ ] Added tests or rationale for omission

---

## 19. Glossary
| Term | Meaning |
|------|---------|
| SRP | Solar Radiation Pressure |
| J2 | Second zonal harmonic of gravity field |
| POD | Plain Old Data (no pointers to managed heap) |
| RHS | Right-Hand Side of ODE (`du/dt = f(u,t)`) |
| Ephemeris Cache | Precomputed body positions (Sun etc.) accessible without SPICE call |

---

## 20. References
1. P. Fil et al., *Mission Concept and Development of the First Interstellar CubeSat Powered by Solar Sailing Technology*, JBIS, 2023.
2. Vallado, *Fundamentals of Astrodynamics and Applications*.
3. Wertz, *Space Mission Engineering: SMAD*.
4. Montenbruck & Gill, *Satellite Orbits*.
5. Sentman, *Free Molecule Flow Theory and Its Application to the Determination of Aerodynamic Forces*.

---

## Acknowledgments
Student contributors, mentors, and the broader open‑source astrodynamics community. DifferentialEquations.jl and StaticArrays.jl projects underpin much of the performance here.

---

## License
MIT – see `LICENSE`.

---

> “Make the one true model correct and fast, then make everything else a view of it.” – Internal Principle

---

**(Sections moved earlier: API now 5, Extended Setup now 12.)**


