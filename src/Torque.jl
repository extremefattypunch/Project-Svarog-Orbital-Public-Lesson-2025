module Torque

using StaticArrays, LinearAlgebra
using ..CentralBody
using ..Constants
using ..SpiceUtils
using ..OrbitState
using ..Gravity
using ..Attitude

export AbstractTorqueModel, torque, SRPTorqueModel, plate_inertia, rod_inertia, parallel_axis, GravityGradientTorqueModel, AtmosphericDragTorqueModel, AlbedoTorqueModel

function plate_inertia(m::T, a::T, b::T) where {T<:AbstractFloat}
    I_x = m * (b^2) / T(12)
    I_y = m * (a^2) / T(12)
    I_z = m * ((a^2) + (b^2)) / T(12)
    SMatrix{3,3}(Diagonal(SVector(I_x, I_y, I_z)))
end

function rod_inertia(m::T, L::T) where {T<:AbstractFloat}
    I_rod = m * (L^2) / T(12)
    SMatrix{3,3}(Diagonal(SVector(T(0.0), I_rod, I_rod)))
end

function parallel_axis(Ic::SMatrix{3,3,T}, m::T, d::SVector{3,T}) where {T<:AbstractFloat}
    d2 = dot(d,d)
    return Ic .+ m * (d2 * SMatrix{3,3,T}(I) - d*d')
end

abstract type AbstractTorqueModel{T<:AbstractFloat} end

# Abstract interface method with return type annotation
function torque(model::AbstractTorqueModel{T},
                r::SVector{3,T}, v::SVector{3,T},
                q::SVector{4,T}, ω::SVector{3,T}, t::T)::SVector{3,T} where {T<:AbstractFloat}
    error("torque not implemented for $(typeof(model))")
end

struct SRPTorqueModel{T<:AbstractFloat} <: AbstractTorqueModel{T}
    area::T
    mass::T
    Cr::T
    lever_arm::SVector{3,T}
    cache::Union{Nothing,SpiceUtils.SunCache{T}}
    function SRPTorqueModel(area::T, mass::T, Cr::T, lever_arm::SVector{3,T}; cache=nothing) where {T<:AbstractFloat}
        new{T}(area, mass, Cr, lever_arm, cache)
    end
end

function torque(model::SRPTorqueModel{T},
                r::SVector{3,T},
                v::SVector{3,T},
                q::SVector{4,T},
                ω::SVector{3,T},
                t::T) where {T<:AbstractFloat}
    # NOTE: If `cache === nothing`, we assume a fixed Sun at +X (1 AU). This path is
    # GPU-safe. If `cache` is provided we index ephemeris via `SpiceUtils.sun_pos`,
    # which is CPU-only (do NOT call this method from inside a GPU ODE kernel when
    # cache ≠ nothing). Future refactor: precompute sun vector and pass numerically.
    sun_pos = model.cache === nothing ? SVector{3,T}(Constants.AU(T), zero(T), zero(T)) : SpiceUtils.sun_pos(model.cache, t)
    ρ = sun_pos - r
    d = norm(ρ)
    r̂ = ρ / d
    R = Attitude.quat_to_rotmat(q)
    n = R * Constants.PLATE_NORMAL_BODY(T)
    cosα = dot(n, r̂)
    cosα <= T(0) && return SVector{3,T}(T(0), T(0), T(0))
    P = Constants.SRP_PRESSURE_1AU(T) * (Constants.AU(T)/d)^2
    F = P * model.area * model.Cr * (cosα^2) * n
    r_lever = R * model.lever_arm
    return cross(r_lever, F)
end

struct GravityGradientTorqueModel{T<:AbstractFloat} <: AbstractTorqueModel{T}
    central_body::AbstractCentralBody{T}
    I_body::SMatrix{3,3,T}
end

function torque(model::GravityGradientTorqueModel{T},
                r::SVector{3,T},
                v::SVector{3,T},
                q::SVector{4,T},
                ω::SVector{3,T},
                t::T) where {T<:AbstractFloat}
    R = Attitude.quat_to_rotmat(q)
    r_body = R' * r
    r_mag = norm(r)
    r_body_u = r_body / r_mag
    μ = model.central_body.mu
    factor = T(3)*μ / (r_mag^3)
    τ_body = factor * cross(r_body_u, model.I_body * r_body_u)
    return τ_body
end

struct AtmosphericDragTorqueModel{T<:AbstractFloat} <: AbstractTorqueModel{T}
    area::T
    mass::T
    C_d::T
    lever_arm::SVector{3,T}
end

function torque(model::AtmosphericDragTorqueModel{T},
                r::SVector{3,T},
                v::SVector{3,T},
                q::SVector{4,T},
                ω::SVector{3,T},
                t::T) where {T<:AbstractFloat}

    # reuse the force routine (identical inputs)
    F_acc = acceleration(AtmosphericDragModel(model.area, model.mass, model.C_d),
                         r, v, q, t) * model.mass     # convert back to Newtons

    # updated, attitude-dependent lever arm ----------------------
    R = Attitude.quat_to_rotmat(q)
    r_lever = R * model.lever_arm                     # body-frame → inertial

    return cross(r_lever, F_acc)
end

struct AlbedoTorqueModel{T<:AbstractFloat} <: AbstractTorqueModel{T}
    area::T
    mass::T
    albedo::T
    lever_arm::SVector{3,T}  # vector from center of mass to surface center
end

function torque(m::AlbedoTorqueModel{T}, r::SVector{3,T}, v::SVector{3,T}, q::SVector{4,T}, ω::SVector{3,T}, t::T) where {T<:AbstractFloat}
    r_sc = r
    d = norm(r_sc)
    if d <= Constants.EARTH_RADIUS(T)
        return SVector{3,T}(T(0.0), T(0.0), T(0.0))
    end

    # Force magnitude same as force model
    P_sun = Constants.SOLAR_CONSTANT(T) * (Constants.AU(T) / d)^2
    F_mag = P_sun * m.albedo * ((Constants.EARTH_RADIUS(T) / d)^2) * m.area

    # Inertial lever arm direction
    R = Attitude.quat_to_rotmat(q)
    r_lever = R * m.lever_arm

    # Force vector (directed outward from Earth center)
    F_vec = F_mag * (r_sc / d)

    return cross(r_lever, F_vec)
end

end # module
