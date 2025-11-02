module Propagator

using LinearAlgebra, DifferentialEquations, StaticArrays, ProgressMeter
using DiffEqGPU  # For GPU support
using ..Torque
using ..Gravity
using ..CentralBody
using ..OrbitState
using ..Control
using ..Attitude

export PropagatorType, propagate, TrajectoryData, add_force!, add_torque!, dynamics!, make_params, DynamicsParams

# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────
struct TrajectoryData{T<:AbstractFloat}
    times  :: Vector{T}
    states :: Vector{StateVector{T}}
end

struct PropagatorType{T<:AbstractFloat, CB<:AbstractCentralBody{T}}
    central_body :: CB
    models       :: Vector{AbstractForceModel{T}}
    torques      :: Vector{AbstractTorqueModel{T}}
    I_body       :: SMatrix{3,3,T}
    I_inv        :: SMatrix{3,3,T}
    tol          :: T
    max_step     :: T
end

# Constructor that keeps everything in T and adds default gravity if needed
function PropagatorType(cb::AbstractCentralBody{T};
                        models   :: Vector{AbstractForceModel{T}} = AbstractForceModel{T}[],
                        torques  :: Vector{AbstractTorqueModel{T}} = AbstractTorqueModel{T}[],
                        degree   :: Int = 0,
                        I_body   :: SMatrix{3,3,T} = SMatrix{3,3,T}(Diagonal(SVector{3,T}(one(T), one(T), one(T)))),
                        tol      :: T = T(1e-9),
                        max_step :: T = T(Inf)
                       ) where {T<:AbstractFloat}

    isempty(models) && push!(models, GravityModel(cb; degree=degree))
    return PropagatorType{T,typeof(cb)}(cb, models, torques, I_body, inv(I_body), tol, max_step)
end

# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers are now centralized in Attitude/Gravity modules

"""
dynamics!(du, u, p::DynamicsParams, t)

Unified translational + attitude kinematics ODE right-hand side.

Characteristics:
    * Pure math, allocation-free, GPU-safe.
    * Uses canonical kernels from `Gravity` (`gravity_accel`, `srp_accel`, `drag_accel`).
    * Uses quaternion utilities from `Attitude` (`quat_normalize`, `quat_derivative`).
    * Performs no SPICE calls, no Dates / I/O / serialization, no heap allocations.
    * Authoritative flight dynamics model – all future physics must be integrated
        here via extensions to `DynamicsParams`.

State layout (u):
    1:3   position r [m]
    4:6   velocity v [m/s]
    7:10  quaternion q (w,x,y,z) – maintained unit-norm
    11:13  body angular velocity ω [rad/s] (currently kinematic only; ω̇=0)
"""
function dynamics!(du, u, p, t)
    T = eltype(u)
    r = SVector{3,T}(u[1], u[2], u[3])
    v = SVector{3,T}(u[4], u[5], u[6])
        q = Attitude.quat_normalize(SVector{4,T}(u[7], u[8], u[9], u[10]))
    ω = SVector{3,T}(u[11], u[12], u[13])

    # Gravity kernel
    a_total = Gravity.gravity_accel(p.cb, r; degree = p.degree)

    # SRP kernel
    if p.use_srp
        a_total += Gravity.srp_accel(p.srp, r, q, p.sun_pos)
    end

    # Drag kernel (optional)
    if p.use_drag
        a_total += Gravity.drag_accel(p.drag, r, v, q, p.cb, T(t))
    end

    # Quaternion derivative (centralized)
    qdot = Attitude.quat_derivative(q, ω)

    # Write results in-place
    du[1] = v[1]; du[2] = v[2]; du[3] = v[3]
    du[4] = a_total[1]; du[5] = a_total[2]; du[6] = a_total[3]
    du[7] = qdot[1]; du[8] = qdot[2]; du[9] = qdot[3]; du[10] = qdot[4]
    du[11] = zero(T); du[12] = zero(T); du[13] = zero(T)  # TODO: add ω̇ integration (torque / inertia model)

    return nothing
end

# Out-of-place wrapper for backward compatibility (using tuple parameters)
function dynamics(u, p, t)
    T = eltype(u)
    du = similar(u)
    dynamics!(du, u, p, t)
    return du
end

# Keep quaternion unit-norm, preserving T
function renormalize_callback!(integrator)
    u = integrator.u
    T = eltype(u)
    q = Attitude.quat_normalize(SVector{4,T}(u[7:10]))
    integrator.u = SVector{13,T}(u[1:6]..., q..., u[11:13]...)
end

# ──────────────────────────────────────────────────────────────────────────────
# Main propagation
# ──────────────────────────────────────────────────────────────────────────────
# Single propagate function for both CPU and GPU execution
function propagate(prop::PropagatorType{T},
                   state0::StateVector{T},
                   tspan::Tuple{T,T};
                   alg = Tsit5(),
                   dt::Union{Nothing,T}=nothing,
                   law::Union{Nothing,AbstractControlLaw{T}}=nothing,
                   progress::Bool=false,
                   progress_steps::Int=100,
                   sun_pos::SVector{3,T} = SVector{3,T}(T(149597870700.0), zero(T), zero(T)),
                   gpu::Bool = false,
                   kwargs...) where {T<:AbstractFloat}

    u0 = [state0.r[1], state0.r[2], state0.r[3], 
          state0.v[1], state0.v[2], state0.v[3],
          state0.q[1], state0.q[2], state0.q[3], state0.q[4],
          state0.ω[1], state0.ω[2], state0.ω[3]]
    
    # Convert PropagatorType to GPU-safe params struct
    p = make_params(prop; sun_pos = sun_pos)
    prob = ODEProblem(dynamics!, u0, tspan, p)

    # Choose algorithm based on GPU flag
    chosen_alg = gpu ? GPUTsit5() : alg

    renorm_cb = DiscreteCallback((u, t, integrator) -> true, renormalize_callback!)
    
    cb = nothing
    if law !== nothing
        function control_callback!(integrator)
            u = integrator.u
            t = integrator.t
            r = SVector{3,T}(u[1:3])
            v = SVector{3,T}(u[4:6])
            q = SVector{4,T}(u[7:10])
            ω = SVector{3,T}(u[11:13])
            q_new = apply_control!(law, r, v, q, ω, t)
            integrator.u = SVector{13,T}(u[1:6]..., q_new..., u[11:13]...)
        end
        control_cb = DiscreteCallback((u,t,integrator)->true, control_callback!; save_positions=(false, false))
        cb = CallbackSet(control_cb, renorm_cb)
    else
        cb = renorm_cb
    end

    common = (reltol = prop.tol, abstol = prop.tol,
              dtmax  = prop.max_step, maxiters = 100000000)

    sol = if dt === nothing
        solve(prob, chosen_alg; callback=cb, progress=progress,
              progress_steps=progress_steps, common..., kwargs...)
    else
        solve(prob, chosen_alg; callback=cb, dt=dt, progress=progress,
              progress_steps=progress_steps, common..., kwargs...)
    end

    # Rewrap the solution into typed StateVector{T}s
    states = Vector{StateVector{T}}(undef, length(sol.u))
    @inbounds for i in eachindex(sol.u)
        u = sol.u[i]
        r = SVector{3,T}(u[1], u[2], u[3])
        v = SVector{3,T}(u[4], u[5], u[6])
        q = SVector{4,T}(u[7], u[8], u[9], u[10])
        ω = SVector{3,T}(u[11], u[12], u[13])
        states[i] = StateVector{T}(r, v, q, ω)
    end

    # Optional debug without leaking types
    t0, tf = sol.t[1], sol.t[end]
    s0, sf = states[1], states[end]
    a0 = zero(SVector{3,T}); af = zero(SVector{3,T})
    τ0 = zero(SVector{3,T}); τf = zero(SVector{3,T})
    for m in prop.models
        a0 += acceleration(m, s0.r, s0.v, s0.q, t0)
        af += acceleration(m, sf.r, sf.v, sf.q, tf)
    end
    for m in prop.torques
        τ0 += torque(m, s0.r, s0.v, s0.q, s0.ω, t0)
        τf += torque(m, sf.r, sf.v, sf.q, sf.ω, tf)
    end
    @debug "Initial acceleration = [$(Float64(a0[1])),$(Float64(a0[2])),$(Float64(a0[3]))] m/s²"
    @debug "Final   acceleration = [$(Float64(af[1])),$(Float64(af[2])),$(Float64(af[3]))] m/s²"
    @debug "Initial torque       = [$(Float64(τ0[1])),$(Float64(τ0[2])),$(Float64(τ0[3]))] N·m"
    @debug "Final   torque       = [$(Float64(τf[1])),$(Float64(τf[2])),$(Float64(τf[3]))] N·m"

    # Force progress bar cleanup if it was used
    if progress
        print("\r" * " "^80 * "\r")  # Clear the progress bar line
        flush(stdout)
    end

    return TrajectoryData{T}(sol.t, states)
end

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
@inline add_force!(prop::PropagatorType{T}, model::AbstractForceModel{T}) where {T} =
    (push!(prop.models, model); prop)

@inline add_torque!(prop::PropagatorType{T}, model::AbstractTorqueModel{T}) where {T} =
    (push!(prop.torques, model); prop)

"""
DynamicsParams{T,CB}

Bundled, GPU-safe parameters consumed by `dynamics!`.

Fields:
    * cb::CB                Central body environment (e.g. `EarthEnv{T}`)
    * degree::Int           Gravity degree (0 => point mass; ≥2 => include J2 if available)
    * use_srp::Bool         Enable Solar Radiation Pressure term
    * srp::SRPParams{T}     Sail / optical properties (area, mass, C_r)
    * use_drag::Bool        Enable atmospheric drag term (Earth-only currently)
    * drag::DragParams{T}   Drag area / mass / coefficient
    * sun_pos::SVector{3,T} Sun position (assumed constant over integration window)

Contract:
    * Absolutely no SPICE, Dates, file I/O, or heap-backed objects.
    * All fields must be Plain Old Data so the struct is GPU-transferable.
"""
struct DynamicsParams{T<:AbstractFloat, CB<:AbstractCentralBody{T}}
        cb::CB
        degree::Int
        use_srp::Bool
        srp::Gravity.SRPParams{T}
        use_drag::Bool
        drag::Gravity.DragParams{T}
        sun_pos::SVector{3,T}
end

# Helper to build DynamicsParams from PropagatorType
function make_params(prop::PropagatorType{T};
                     sun_pos::SVector{3,T} = SVector{3,T}(T(149597870700.0), zero(T), zero(T))) where {T}

    # Determine gravity degree from any GravityModel in models (fallback 0)
    degree = 0
    for m in prop.models
        if m isa GravityModel{T}
            degree = max(degree, m.degree)
        end
    end

    # SRP params
    use_srp = false
    srp = Gravity.SRPParams{T}(zero(T), one(T), one(T))
    for m in prop.models
        if m isa SRPModel{T}
            srp = Gravity.SRPParams{T}(m.area, m.mass, m.C_r)
            use_srp = true
            break
        end
    end

    # Drag params
    use_drag = false
    drag = Gravity.DragParams{T}(zero(T), one(T), one(T))
    for m in prop.models
        if m isa AtmosphericDragModel{T}
            drag = Gravity.DragParams{T}(m.area, m.mass, m.C_d)
            use_drag = true
            break
        end
    end

    return DynamicsParams{T, typeof(prop.central_body)}(
        prop.central_body,
        degree,
        use_srp,
        srp,
        use_drag,
        drag,
        sun_pos,
    )
end

end # module
