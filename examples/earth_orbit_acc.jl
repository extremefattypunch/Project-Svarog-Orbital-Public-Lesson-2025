#!/usr/bin/env julia
#
# earth_orbit_acc.jl  —  Single-trajectory propagation (13-state Cartesian)
# Float32 version (script-only changes; library untouched)

using SolarSailPropagator
using LinearAlgebra, StaticArrays, Dates, Logging, JLD2, InteractiveUtils
using TerminalLoggers
using Base.Threads, Printf, Plots

global_logger(TerminalLogger(stderr, Logging.Debug))

# Use Float32 throughout this script (set to Float64 if higher precision needed)
const T = Float32

# 0.  Run-time options
const YEARS        = T(1) / T(4)     # 0.25 years
const DT          = T(100)
const WARMUP_DT   = T(1)
const PROGRESS_BAR = true
const RATE        = T(1.0)
const PITCH       = T(50)
const MODEL_TORQUES = false
# TEMP_FIXED_SUN: begin temporary fixed-Sun configuration block
# Temporary flag: run with fixed Sun (no cache/SPICE) for GPU friendliness
const FIXED_SUN = false


# 1.  Helpers
function warmup!(prop, state, law)
    propagate(prop, state, (T(0), T(10));
              dt = WARMUP_DT, adaptive = false, law = law)
    return nothing
end

# 2.  Sun-position cache (optional). If FIXED_SUN=true we skip building/loading.
# TEMP_FIXED_SUN: begin conditional cache bypass
cache = nothing
if !FIXED_SUN
    cache_file = joinpath(@__DIR__, "sun_cache.jld2")
    cache = load_cache(cache_file; T=T)
    if cache === nothing
        cache = SunCache(T(86_400); T=T)
        build_cache!(cache, T(0), T(3 * 365.25 * 86_400))
        save_cache(cache, cache_file)
    end
end
# TEMP_FIXED_SUN: end conditional cache bypass

# 3.  Initial conditions
earth = EarthEnv{T}()   # <— Float32 central body

# Orbital elements
a_km = T(6378.137) + T(1000.0)  # semi-major axis in km
e     = T(0.001)
i     = T(99.79)               # degrees
Ω     = T(58.6)
ω     = T(0.0)
ν     = T(0.0)
r, v = coe2rv(a_km * T(1e3), e, i, Ω, ω, ν, earth.mu)

# Attitude and angular velocity
q0 = fixed_in_plane_attitude(r, v; angle_deg = PITCH)
ω0 = SVector{3,T}(T(0), T(0), RATE)

state0 = StateVector{T}(r, v, q0, ω0)

# 4.  Propagator setup
# Cross-braced solar sail, total mass m, arm length L
m = T(10.0)    # total mass
L = T(3.0)     # length of each rod (x and y brace)
I_xx = (T(1)/T(24)) * m * (L^2)
I_yy = I_xx
I_zz = (T(1)/T(12)) * m * (L^2)

I_body = SMatrix{3,3,T}(Diagonal(SVector{3,T}(I_xx, I_yy, I_zz)))

prop = PropagatorType(earth; degree = 2, I_body = I_body)

area, mass, C_r = (L^2), m, T(1.7)
lever = SVector{3,T}(T(0.00), T(0.00), T(0.05))  # CoG to CoP

C_d = T(2.2)

# Use typed cache to avoid CSPICE calls in the loop
add_force!(prop, SRPModel(area, mass, C_r; cache = cache))
add_force!(prop, AtmosphericDragModel(area, mass, C_d))
add_force!(prop, AlbedoModel(area, mass, T(0.3)))

if MODEL_TORQUES
    add_torque!(prop, SRPTorqueModel(area, mass, C_r, lever; cache = cache))
    add_torque!(prop, GravityGradientTorqueModel(earth, I_body))
    add_torque!(prop, AtmosphericDragTorqueModel(area, mass, C_d, lever))
end


law = nothing  # cache may be nothing (fixed Sun mode)



# 5.  Compilation warm-up
@info "Warming up JIT …  " Threads.nthreads() " threads"
warmup!(prop, state0, law)

# 6.  Main propagation
t_end  = YEARS * T(365) * T(86_400)
@info "Propagating for $(Float64(YEARS)) year(s)  →  $(Float64(t_end)/86_400) days"

t0 = time()
traj = with_logger(TerminalLogger()) do
    propagate(prop, state0, (T(0), t_end);
        dt       = DT,
        adaptive = false,
        progress = true,
        law = law
    )
end

elapsed = time() - t0
@info @sprintf("Propagation finished in %.1f s (%.1f min)",
               elapsed, elapsed/60)

# 7.  Final orbital elements & plots
final_state = traj.states[end]
a_f,e_f,i_f,Ω_f,ω_f,ν_f = rv2coe(final_state.r, final_state.v, earth.mu)
@printf("\nFinal elements after %.1f days:\n", Float64(t_end)/86_400)
@printf("  a  = %.1f km\n  e  = %.4f\n  i  = %.3f°\n", Float64(a_f)/1e3, Float64(e_f), Float64(i_f))
@printf("  Ω  = %.3f°   ω = %.3f°   ν = %.3f°\n", Float64(Ω_f), Float64(ω_f), Float64(ν_f))

@printf("\nFinal state vector:\n")
@printf("  r = [%.3f, %.3f, %.3f] m\n", Float64.(final_state.r)...)
@printf("  v = [%.3f, %.3f, %.3f] m/s\n", Float64.(final_state.v)...)
@printf("  q = [%.5f, %.5f, %.5f, %.5f]\n", Float64.(final_state.q)...)
@printf("  ω = [%.5f, %.5f, %.5f] rad/s\n\n", Float64.(final_state.ω)...)

# Plotting
display(plot_orbit_3d(traj, earth))
display(plot_altitude(traj, earth))
display(plot_orbital_elements(traj, earth))
display(plot_attitude(traj))
display(plot_force_contributions(traj, prop))
display(plot_torque_contributions(traj, prop))
display(plot_srp_contribution(traj, prop))
display(plot_srp_transverse_work(traj, prop, T(0)))
display(plot_srp_transverse_work_per_orbit(traj, prop))
display(plot_srp_radial_work_per_orbit(traj, prop))
display(plot_srp_normal_work_per_orbit(traj, prop))
