#!/usr/bin/env julia
# earth_ensemble_gpu.jl — GPU-accelerated ensemble propagation using DiffEqGPU.jl
# RAAN × attitude combinations, GPU parallel (1 thread per trajectory)

using SolarSailPropagator
using LinearAlgebra, StaticArrays, Dates, Logging, JLD2
using TerminalLoggers, ProgressMeter
using Base.Threads, Printf
using CUDA, DiffEqGPU
using DifferentialEquations

##############################################################################
# 0.  Run-time options
##############################################################################
const ALTITUDES   = [600.0]         # in km
const N_Ω         = 30
const N_ANGLE     = 30
const YEARS       = 1/3
const OUTER_DT    = 100.0
const OUT_DIR     = joinpath(@__DIR__, "trajectories_ensemble_updated_gpu")
mkpath(OUT_DIR)

# Use Float64 for better numerical stability (Float32 causes precision issues)
const T = Float64

##############################################################################
# 1.  Helpers
##############################################################################
function warmup_gpu!(prob_func, prob, trajectories)
    # Small warm-up ensemble to compile GPU kernels
    warmup_trajectories = min(4, trajectories)
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    solve(ensemble_prob, Tsit5(), EnsembleGPUArray(T), trajectories=warmup_trajectories, 
          dt=T(1.0), adaptive=false, callback=nothing)
    return nothing
end

##############################################################################
# 2.  Hard-coded Sun position (no cache, GPU-friendly)
##############################################################################
# Fixed sun at +X 1 AU for GPU compatibility
const FIXED_SUN_POS = SVector{3,T}(T(149597870700.0), zero(T), zero(T))  # 1 AU in meters

##############################################################################
# 3.  Problem function for EnsembleProblem
##############################################################################
function prob_func(prob, i, repeat)
    # Calculate RAAN and angle from linear index i
    i_raan = ((i - 1) % N_Ω) + 1
    i_angle = div(i - 1, N_Ω) + 1
    
    Ω = T((i_raan - 1) * 360.0 / N_Ω)
    angle_deg = T((i_angle - 1) * 180.0 / N_ANGLE)
    
    # Generate initial conditions for this trajectory
    earth = EarthEnv{T}()
    μ = earth.mu
    alt_km = T(ALTITUDES[1])  # Only using first altitude for now
    
    a_km, e, i, ω, ν = T(6378.137) + alt_km, T(0.0), T(97.97), T(0.0), T(0.0)
    r, v = coe2rv(a_km * T(1e3), e, i, Ω, ω, ν, μ)
    q0 = fixed_in_plane_attitude(r, v; angle_deg = angle_deg)
    ω0 = SVector{3,T}(T(0), T(0), T(0))  # No initial rotation
    
    u0 = [r[1], r[2], r[3], v[1], v[2], v[3], q0[1], q0[2], q0[3], q0[4], ω0[1], ω0[2], ω0[3]]
    
    # Create a minimal PropagatorType to extract parameters
    temp_prop = PropagatorType(earth; degree = 2)
    area = T(36.0)  # 6m x 6m sail
    mass = T(10.0)
    C_r = T(1.7)
    add_force!(temp_prop, SRPModel(area, mass, C_r))
    
    # Use shared parameter function
    p = make_params(temp_prop; sun_pos = FIXED_SUN_POS)
    
    remake(prob; u0=u0, p=p)
end

##############################################################################
# 6.  Main execution
##############################################################################
for alt_km in ALTITUDES
    this_out_dir = joinpath(OUT_DIR, "$(Int(alt_km))km_$(Int(OUTER_DT))dt")
    mkpath(this_out_dir)

    @info "Beginning GPU ensemble propagation at $(alt_km) km altitude"
    total_trajectories = N_Ω * N_ANGLE
    
    # Setup base problem with shared dynamics
    earth = EarthEnv{T}()
    t_end = T(YEARS * 365 * 86_400)
    tspan = (T(0), t_end)
    
    # Create dummy propagator to get parameters
    temp_prop = PropagatorType(earth; degree = 2)
    area = T(36.0)  # 6m x 6m sail
    mass = T(10.0)
    C_r = T(1.7)
    add_force!(temp_prop, SRPModel(area, mass, C_r))
    p_dummy = make_params(temp_prop; sun_pos = FIXED_SUN_POS)
    
    # Dummy initial condition (will be overridden by prob_func)
    u0_dummy = zeros(T, 13)
    
    prob = ODEProblem(dynamics!, u0_dummy, tspan, p_dummy)
    
    @info "Warming up GPU kernels..."
    warmup_gpu!(prob_func, prob, total_trajectories)
    
    # Define reduction function to save batches to disk
    function my_reduction(u, batch, I)
        for (j, global_i) in enumerate(I)
            sol = batch[j]
            i_raan = ((global_i - 1) % N_Ω) + 1
            i_angle = div(global_i - 1, N_Ω) + 1
            
            Ω = (i_raan - 1) * 360.0 / N_Ω
            angle_deg = (i_angle - 1) * 180.0 / N_ANGLE
            
            out_file = joinpath(this_out_dir, @sprintf("%06.1f_%06.1f_traj.jld2", Ω, angle_deg))
            
            # Convert solution to TrajectoryData format
            states = Vector{StateVector{T}}(undef, length(sol.u))
            @inbounds for k in eachindex(sol.u)
                uk = sol.u[k]
                r = SVector{3,T}(uk[1], uk[2], uk[3])
                v = SVector{3,T}(uk[4], uk[5], uk[6])
                q = SVector{4,T}(uk[7], uk[8], uk[9], uk[10])
                ω = SVector{3,T}(uk[11], uk[12], uk[13])
                states[k] = StateVector{T}(r, v, q, ω)
            end
            
            traj = TrajectoryData{T}(sol.t, states)
            JLD2.@save out_file traj
        end
        return u, false
    end
    
    @info "Running GPU ensemble..."
    t0 = time()
    
    backend = CUDA.CUDABackend()
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func, reduction=my_reduction, u_init=[])
    sim = solve(ensemble_prob, Tsit5(), DiffEqGPU.EnsembleGPUArray(T), 
                trajectories=total_trajectories,
                dt=T(OUTER_DT), 
                adaptive=false,
                reltol=T(1e-8),
                abstol=T(1e-10),
                batch_size = 16,  # Reduced for memory management
                callback=nothing)  # No callbacks on GPU
    
    elapsed = time() - t0
    @info @sprintf("GPU ensemble completed in %.1f s (%.1f min)", elapsed, elapsed/60)
    
    @info @sprintf("All %d trajectories at %.0f km saved", total_trajectories, alt_km)
end

@info "GPU ensemble propagation complete!"
