module Control

using LinearAlgebra, StaticArrays, Dates
using ..SpiceUtils
using ..Constants
using ..Attitude

export AbstractControlLaw, apply_control!, SunFeatherLaw, fixed_in_plane_attitude, angular_velocity_along_body_z

# ---------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------
abstract type AbstractControlLaw{T<:AbstractFloat} end

"""
    apply_control!(law, r, v, q, ω, t) -> q_new

Compute a corrected quaternion according to the control strategy `law`.
"""
function apply_control!(law::AbstractControlLaw{T},
                        r::SVector{3,T},
                        v::SVector{3,T},
                        q::SVector{4,T},
                        ω::SVector{3,T},
                        t::T) where {T<:AbstractFloat}
    error("apply_control! not implemented for $(typeof(law))")
end

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# Use Attitude.rotmat_to_quat instead of local conversion

# Return an arbitrary unit vector perpendicular to `v`
function perp_vector(v::SVector{3,T}) where {T<:AbstractFloat}
    p = cross(v, SVector{3,T}(T(0.0), T(0.0), T(1.0)))
    if norm(p) < T(1e-6)
        p = cross(v, SVector{3,T}(T(0.0), T(1.0), T(0.0)))
    end
    return normalize(p)
end

# ---------------------------------------------------------------------
# Example control law
# ---------------------------------------------------------------------

"""
        SunFeatherLaw(; cache=nothing)

Simple control strategy: if spacecraft radial velocity relative to the Sun is
outbound, orient the sail normal toward the Sun (maximum thrust). If inbound,
feather the sail by aligning the normal perpendicular to the Sun line.

Notes
- CPU/GPU: For GPU runs, set `cache=nothing` so this law uses a fixed Sun
    at +X (1 AU). Using a SPICE-backed cache in a callback is CPU-only.
- This function allocates only stack-based StaticArrays and performs
    pure math; it calls no I/O. If `cache` is provided, the cache lookup
    must occur on CPU.
"""
struct SunFeatherLaw{T<:AbstractFloat} <: AbstractControlLaw{T}
    # Tie cache element type to T to avoid Float64 in Float32 runs
    cache::Union{Nothing,SpiceUtils.SunCache{T}}
end

# Explicit outer constructor to choose element type parameter T
SunFeatherLaw{T}(; cache=nothing) where {T} = SunFeatherLaw{T}(cache)

# Default to Float64 when not specified
SunFeatherLaw(; cache=nothing) = SunFeatherLaw{Float64}(cache)

function apply_control!(law::SunFeatherLaw{T},
                        r::SVector{3,T},
                        v::SVector{3,T},
                        q::SVector{4,T},
                        ω::SVector{3,T},
                        t::T) where {T<:AbstractFloat}

    # TEMP_FIXED_SUN: begin temporary fallback (Sun fixed at +X 1 AU if cache==nothing)
    # If cache provided, use cached Sun position (normal path). Otherwise use fixed +X 1 AU Sun
    sun_pos = law.cache === nothing ? SVector{3,T}(Constants.AU(T), zero(T), zero(T)) : SpiceUtils.sun_pos(law.cache, t)
    # TEMP_FIXED_SUN: end temporary fallback

    r_hat = normalize(sun_pos - r)
    v_rad = dot(v, r_hat)

    # Desired sail normal
    z_axis = v_rad > T(0) ? r_hat : perp_vector(r_hat)

    # Build a right-handed frame with ẑ = normal
    x_axis = perp_vector(z_axis)
    y_axis = cross(z_axis, x_axis)
    R = SMatrix{3,3,T}(hcat(x_axis, y_axis, z_axis))

    return rotmat_to_quat(R)
end

function fixed_in_plane_attitude(r::SVector{3,T},
                                 v::SVector{3,T};
                                 angle_deg = T(45.0)) where {T<:AbstractFloat}

    ĥ = normalize(cross(r, v))         # angular-momentum direction
    r̂ = normalize(r)

    tangential = normalize(cross(ĥ, r̂))
    θ = deg2rad(angle_deg)

    ẑd = normalize(cos(θ)*r̂ + sin(θ)*tangential)
    ŷd = normalize(cross(ẑd, ĥ))
    x̂d = normalize(cross(ŷd, ẑd))

    R = SMatrix{3,3,T}(hcat(x̂d, ŷd, ẑd))   # ensure element type is T
    return rotmat_to_quat(R)
end

function angular_velocity_along_body_z(q::SVector{4,T}, rate::T) where {T<:AbstractFloat} #TODO: Check implementation
    # rotation matrix body → inertial
    R = quat_to_rotmat(q)

    # body z-axis expressed in inertial frame
    z_hat_inertial = R[:, 3]

    return rate * z_hat_inertial        # this is an SVector{3,T}
end

end # module