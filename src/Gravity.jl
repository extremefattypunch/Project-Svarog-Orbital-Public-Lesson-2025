module Gravity

# ---------------------------------------------------------------------------
# Force Models Module
# ---------------------------------------------------------------------------
# Units: SI throughout (m, s, kg, N, W). Positions are inertial ECI/J2000 frame
# unless otherwise specified. Velocities same frame. Quaternions q = (q0,q1,q2,q3)
# are scalar-first, right-handed, active rotations from body frame to inertial
# frame (i.e. v_inertial = R(q) * v_body). Plate normal `Constants.PLATE_NORMAL_BODY(T)`
# is defined in the body frame and rotated to inertial via Attitude.quat_to_rotmat.
# An identity quaternion is auto-injected by convenience acceleration overloads
# that omit attitude; be explicit when attitude matters.
# ---------------------------------------------------------------------------

using LinearAlgebra, StaticArrays
using ..Constants
using ..CentralBody
using ..SpiceUtils
using ..OrbitState
using ..Illumination
using ..Attitude

export AbstractForceModel, GravityModel, SRPModel, acceleration, AtmosphericDragModel, nrlmsise_density, AlbedoModel, gravity_acceleration, srp_acceleration,
       SRPParams, DragParams, gravity_accel, srp_accel, drag_accel

"""
        AbstractForceModel{T}

Abstract supertype for all force models providing an `acceleration(model, r, v, q, t)` method.
Interface contract:
    r :: SVector{3,T}  position [m] inertial (ECI/J2000)
    v :: SVector{3,T}  velocity [m/s] inertial
    q :: SVector{4,T}  attitude quaternion (scalar-first) body->inertial; if omitted identity is assumed.
    t :: T             time variable (seconds from chosen epoch) or simulation time.

Must return SVector{3,T} acceleration [m/s^2] in the same inertial frame.
Implementations should be allocation-free and GPU-friendly.
"""
abstract type AbstractForceModel{T<:AbstractFloat} end

# Abstract interface methods with return type annotations
function acceleration(model::AbstractForceModel{T},
                      r::SVector{3,T}, v::SVector{3,T},
                      q::SVector{4,T}, t::T)::SVector{3,T} where {T<:AbstractFloat}
    error("acceleration not implemented for $(typeof(model))")
end

@inline function acceleration(model::AbstractForceModel{T},
                              r::SVector{3,T}, v::SVector{3,T}, t::T)::SVector{3,T} where {T<:AbstractFloat}
    acceleration(model, r, v, SVector{4,T}(one(T), zero(T), zero(T), zero(T)), t)
end

struct GravityModel{T<:AbstractFloat} <: AbstractForceModel{T}
    central_body::AbstractCentralBody{T}
    degree::Int              # 0 => point-mass; 2 => include J2 only
    function GravityModel(central_body::AbstractCentralBody{T}; degree::Int=0, use_J2::Bool=false) where {T<:AbstractFloat}
        if degree != 0 && degree != 2
            error("GravityModel currently supports only degree=0 (point mass) or degree=2 (J2). Requested degree=$(degree)")
        end
        # If use_J2 provided, override degree argument for clarity
        d = use_J2 ? 2 : degree
        new{T}(central_body, d)
    end
end

function acceleration(model::GravityModel{T},
                      r::SVector{3,T}, v::SVector{3,T},
                      q::SVector{4,T}, t::T) where {T<:AbstractFloat}
    return gravity_accel(model.central_body, r; degree=model.degree)
end

acceleration(model::GravityModel{T}, r::SVector{3,T}, v::SVector{3,T}, t::T) where {T<:AbstractFloat} =
    acceleration(model, r, v, SVector{4,T}(one(T), zero(T), zero(T), zero(T)), t)

function spherical_harmonics_acceleration(model::GravityModel{T}, r::SVector{3,T}) where {T<:AbstractFloat}
    if model.degree >= 2
        return gravity_accel(model.central_body, r; degree=2) - gravity_accel(model.central_body, r; degree=0)
    else
        return zero(SVector{3,T})
    end
end

# LEGACY: Prefer `gravity_accel(body, r; degree=...)` downstream. This helper remains
# temporarily for backward compatibility. Do not call from new code.
function gravity_acceleration(μ::T, J2::T, Re::T, r::SVector{3,T}) where {T<:AbstractFloat}
    Base.depwarn("gravity_acceleration(μ,J2,Re,r) is deprecated; use gravity_accel(body, r; degree= (J2==0 ? 0 : 2)) with a central body instance", :gravity_acceleration)
    r_mag = norm(r)
    a_gravity = -μ * r / (r_mag^3)
    
    # J2 perturbation
    if J2 != zero(T)
        z_frac = r[3] / r_mag
        factor = T(1.5) * J2 * (Re / r_mag)^2
        a_j2 = factor * μ / r_mag^3 * SVector{3,T}(
            r[1] * (T(5) * z_frac^2 - T(1)),
            r[2] * (T(5) * z_frac^2 - T(1)),
            r[3] * (T(5) * z_frac^2 - T(3))
        )
        a_gravity += a_j2
    end
    
    return a_gravity
end

# Canonical gravity kernel using body fields; degree=0 (point mass) or >=2 (includes J2 if body.j2>0)
"""
        gravity_accel(body, r; degree=0) -> SVector{3,T}

Point-mass (degree=0) or J2-augmented (degree=2) gravitational acceleration.

Arguments:
    body   :: AbstractCentralBody – provides μ, radius, j2
    r      :: SVector{3,T} position [m] inertial frame
    degree :: Int (0 or 2). 2 adds only the J2 term if body.j2 > 0.

Returns: inertial acceleration [m/s^2].

Notes:
    * Higher-order harmonics (n>2) are not yet implemented.
    * Will ignore J2 if body.j2 == 0 or body.radius == 0.
"""
function gravity_accel(body::AbstractCentralBody{T}, r::SVector{3,T}; degree::Int=0) where {T<:AbstractFloat}
    μ = body.mu
    r_mag = norm(r)
    a = -μ * r / (r_mag^3)
    if degree >= 2 && body.j2 != zero(T) && body.radius != zero(T)
        z_frac = r[3] / r_mag
        factor = T(1.5) * body.j2 * (body.radius / r_mag)^2
        a += factor * μ / r_mag^3 * SVector{3,T}(
            r[1] * (T(5) * z_frac^2 - T(1)),
            r[2] * (T(5) * z_frac^2 - T(1)),
            r[3] * (T(5) * z_frac^2 - T(3))
        )
    end
    return a
end

# LEGACY: Prefer `srp_accel(SRPParams(...), r, q, sun_pos)` downstream. This wrapper
# remains for compatibility with older high-level model code.
function srp_acceleration(area::T, mass::T, C_r::T, r::SVector{3,T}, q::SVector{4,T}, 
                          sun_pos::SVector{3,T}) where {T<:AbstractFloat}
    return srp_accel(SRPParams{T}(area, mass, C_r), r, q, sun_pos)
end

"""
    SRPParams{T}

Parameters for simple solar radiation pressure (SRP) model.
Fields:
  area :: T  sail/craft effective area [m^2]
  mass :: T  spacecraft mass [kg]
  C_r  :: T  reflectivity coefficient (1=perfect absorber, up to ~1.8 ideal specular)
"""
struct SRPParams{T<:AbstractFloat}
    area::T
    mass::T
    C_r::T
end

"""
        srp_accel(params, r, q, sun_pos) -> SVector{3,T}

Solar radiation pressure acceleration using a single plate normal.

Arguments:
    params  :: SRPParams
    r       :: spacecraft position [m] inertial
    q       :: attitude quaternion (scalar-first) body->inertial
    sun_pos :: Sun position [m] inertial (same frame as r)

Notes:
    * Uses Illumination.shadow_factor for eclipse; returns zero if shadowed.
    * Pressure scales ~ 1/R^2 with Sun-spacecraft distance.
    * Force direction taken as the plate normal projected by cosine incidence and scaled by (1 + C_r).
"""
function srp_accel(params::SRPParams{T}, r::SVector{3,T}, q::SVector{4,T}, sun_pos::SVector{3,T}) where {T<:AbstractFloat}
    illum = Illumination.shadow_factor(r, sun_pos)
        illum <= zero(T) && return zero(SVector{3,T})

    sun_vec = sun_pos - r
    sun_dir = normalize(sun_vec)
    sun_distance = norm(sun_vec)

    P_sun = SOLAR_CONSTANT(T) * (AU(T) / sun_distance)^2
    pressure = P_sun / C_LIGHT(T)

    R = Attitude.quat_to_rotmat(q)
    n_inertial = R * PLATE_NORMAL_BODY(T)

    cos_θ = max(zero(T), dot(sun_dir, n_inertial))
    force_magnitude = illum * pressure * params.area * cos_θ * (T(1) + params.C_r)
    return (force_magnitude / params.mass) * n_inertial
end

struct SRPModel{T<:AbstractFloat} <: AbstractForceModel{T}
    area::T
    mass::T
    C_r::T
    cache::Union{Nothing,SpiceUtils.SunCache{T}}
    function SRPModel(area::T, mass::T, C_r::T; cache=nothing) where {T<:AbstractFloat}
        new{T}(area, mass, C_r, cache)
    end
end

## shadow_factor moved to Illumination module as the canonical implementation

function acceleration(model::SRPModel{T}, r::SVector{3,T}, v::SVector{3,T}, q::SVector{4,T}, t::T) where {T<:AbstractFloat}
    # Get sun position (use cache if available, otherwise fixed position)
    sun_vec = model.cache === nothing ?
              SVector{3,T}(AU(T), zero(T), zero(T)) :
              SpiceUtils.sun_pos(model.cache, t)
    # Use the unified GPU-friendly function (LEGACY wrapper for now)
    return srp_acceleration(model.area, model.mass, model.C_r, r, q, sun_vec)
end

acceleration(model::SRPModel{T}, r::SVector{3,T}, v::SVector{3,T}, t::T) where {T<:AbstractFloat} =
    acceleration(model, r, v, SVector{4,T}(T(1.0), T(0.0), T(0.0), T(0.0)), t)
"""
AtmosphericDragModel{T}

Earth-only free-molecular drag model.

Notes:
    * Assumes Earth's atmospheric density via `nrlmsise_density` (simple table model).
    * Assumes Earth rotation about +Z using the central body rotation rate when computing
        relative wind (currently hard-coded via `EarthEnv{T}()` inside acceleration).
    * GPU-safe: pure numeric operations; no SPICE, I/O, or heap allocations.
    * Future extension may add a `body` field to generalize beyond Earth.
"""
struct AtmosphericDragModel{T<:AbstractFloat} <: AbstractForceModel{T}
    area::T   # reference area [m^2]
    mass::T   # spacecraft mass [kg]
    C_d::T    # drag coefficient (used when angle model disabled or for baseline)
end

# https://www.pdas.com/bigtables.html source for density values 
"""
    nrlmsise_density(alt_km, lat, lon, t) -> T

Placeholder exponential-stitch density model (NOT real NRLMSISE-00).
Origin: stitched exponential fits around US Standard Atmosphere 1976
values at 600–1000 km (600,700,800,900,1000 km anchor points).
Valid range: table bounds (≈100–1000 km). Outside returns 0.
Inputs lat, lon, t currently ignored; kept for interface compatibility.
Returns mass density [kg/m^3].
"""
function nrlmsise_density(alt_km::T, lat::T, lon::T, t::T) where {T<:AbstractFloat}
    tbl = Constants.DENSITY_TABLE(T)
    if alt_km < tbl[1][1] || alt_km > tbl[end][1]
        return zero(T)
    end
    @inbounds for i in 1:length(tbl)-1
        h0, H, ρ0 = tbl[i]
        h1, _, _  = tbl[i+1]
        if alt_km >= h0 && alt_km < h1
            return ρ0 * exp(-(alt_km - h0) / H)
        end
    end
    h0, H, ρ0 = tbl[end]
    return ρ0 * exp(-(alt_km - h0) / H)
end

# ---------- helper ----------------------------------------------
"""
    sentman_coeffs(α; σ=0.9) -> (Cd, Cl)

Sentman free-molecule aerodynamic coefficients for combined diffuse/specular
reflection mix σ. α is angle between surface normal and free-stream velocity.
Returns (drag coefficient Cd, lift coefficient Cl) for panel formulation.
Reference: Sentman, A. L. "Free Molecule Flow Theory and Its Application to the Determination of Aerodynamic Forces" (1961).
"""
function sentman_coeffs(α::T; σ::T = T(0.9)) where {T<:AbstractFloat}
    S, C = sin(α), cos(α)
    Cn = (T(2) - σ) * S * C + σ * sqrt(T(π)) * S^2      # normal coeff
    Ct = T(2)*σ / sqrt(T(π)) * S                         # shear  coeff
    Cd = Cn * S + Ct * C
    Cl = Cn * C - Ct * S
    return Cd, Cl
end

function acceleration(model::AtmosphericDragModel,
                      r::SVector{3,T},
                      v::SVector{3,T},
                      q::SVector{4,T},
                      t::T) where {T<:AbstractFloat}
    # Earth-only implementation: environment hard-coded to EarthEnv.
    # Future generalisation may expose body env through the model.
    return drag_accel(DragParams{T}(model.area, model.mass, model.C_d), r, v, q, CentralBody.EarthEnv{T}(), t)
end

function acceleration(model::AtmosphericDragModel, r::SVector{3,T}, v::SVector{3,T}, t::T) where {T<:AbstractFloat}
    acceleration(model, r, v, SVector{4,T}(T(1.0), T(0.0), T(0.0), T(0.0)), t)
end

struct DragParams{T<:AbstractFloat}
    area::T
    mass::T
    C_d::T
end

"""
        drag_accel(params, r, v, q, body; t=0) -> SVector{3,T}

Free-molecular aerodynamic drag + lift using Sentman panel model.

Arguments:
    params :: DragParams (area [m^2], mass [kg], C_d baseline)
    r      :: position [m] inertial
    v      :: velocity [m/s] inertial
    q      :: attitude quaternion (scalar-first) body->inertial
    body   :: central body env (rotation_rate used; Earth assumed for density)
    t      :: time (currently unused in density placeholder)

Returns acceleration [m/s^2]. Zero if density negligible or velocity ~0.
Notes:
    * Uses placeholder exponential density (see nrlmsise_density).
    * Angle of attack derived from panel normal vs relative wind.
    * Lift direction degeneracy handled via epsilon threshold.
"""
function drag_accel(params::DragParams{T},
                                        r::SVector{3,T},
                                        v::SVector{3,T},
                                        q::SVector{4,T},
                                        body::AbstractCentralBody{T},
                                        t::T = zero(T)) where {T<:AbstractFloat}
    # 1. density
    alt_km = (norm(r) - Constants.EARTH_RADIUS(T)) / T(1e3)
    ρ = nrlmsise_density(alt_km, T(0), T(0), t)
        ρ == T(0) && return zero(SVector{3,T})

    # 2. relative wind using body rotation
    Ω = SVector{3,T}(zero(T), zero(T), body.rotation_rate)
    v_rel = v - cross(Ω, r)
    v_mag = norm(v_rel)
    v_mag == T(0) && return zero(SVector{3,T})
    v̂ = v_rel / v_mag

    # 3. body-fixed normal
    R = Attitude.quat_to_rotmat(q)
    n_b = R * Constants.PLATE_NORMAL_BODY(T)
    α = acos(clamp(-dot(n_b, v̂), T(-1), T(1)))

    # 4. coefficients and dynamic pressure
    Cd, Cl = sentman_coeffs(α)
    q_dyn = T(0.5) * ρ * (v_mag^2)

    # 5. lift direction
    e_L = cross(v̂, n_b)
    lift_unit = norm(e_L) < T(1e-12) ? zero(SVector{3,T}) : normalize(cross(e_L, v̂))

    # 6. acceleration
    F = params.area * q_dyn * (-Cd * v̂ + Cl * lift_unit)
    return F / params.mass
end
"""
        AlbedoModel{T}

Simplified Earth albedo / Earthshine model treating Earth as a Lambertian
reflector returning uniform reflected solar flux. Uses single plate normal.
Assumptions:
    * No eclipse coupling (does not reduce during Earth in shadow relative to Sun).
    * Scale ~ (R_earth / d)^2, ignores Earth phase angle vs Sun.
    * cosα^2 projection for reflected momentum exchange (heuristic).
Fields:
    area   :: illuminated area [m^2]
    mass   :: spacecraft mass [kg]
    albedo :: Earth effective albedo fraction (e.g. 0.3)
"""
struct AlbedoModel{T<:AbstractFloat} <: AbstractForceModel{T}
        area::T     # m²
        mass::T     # kg
        albedo::T   # dimensionless (e.g. 0.3)
end

function acceleration(m::AlbedoModel{T}, r::SVector{3,T}, v::SVector{3,T},
                      q::SVector{4,T}, t::T) where {T<:AbstractFloat}

    d = norm(r)
    if d <= Constants.EARTH_RADIUS(T)
        return zero(SVector{3,T})
    end

    # Radiation pressure from Earth (reflected Sunlight)
    # Step 1: normalize Earth-to-spacecraft vector
    r̂ = r / d

    # Step 2: get sail normal in inertial frame
    R = Attitude.quat_to_rotmat(q)
    n = R * Constants.PLATE_NORMAL_BODY(T)
    cosα = dot(n, r̂)
    cosα <= T(0) && return zero(SVector{3,T})

    # Step 3: reflected pressure magnitude
    P_sun = Constants.SOLAR_CONSTANT(T)             # W/m²
    P_reflected = P_sun / Constants.C_LIGHT(T)      # N/m²
    scale = P_reflected * m.albedo * (Constants.EARTH_RADIUS(T) / d)^2

    # Step 4: acceleration
    a_mag = scale * m.area * (cosα^2) / m.mass
    return a_mag * n
end


end # module
