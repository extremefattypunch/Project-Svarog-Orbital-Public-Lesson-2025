module Illumination

using StaticArrays, LinearAlgebra
using ..Constants
using ..CentralBody

export illumination_factor, shadow_factor

"""
    illumination_factor(r_sc, sun_pos, body) -> T in [0,1]

Conical umbra/penumbra illumination model (Basilisk-like). Returns the fraction
of incident solar flux reaching the spacecraft. 0 = full umbra, 1 = full sun.

Arguments:
- r_sc::SVector{3,T}: spacecraft position in inertial frame [m]
- sun_pos::SVector{3,T}: Sun position in same frame [m]
- body::AbstractCentralBody{T}: occluding body (e.g., Earth)
"""
function illumination_factor(r_sc::SVector{3,T},
                             sun_pos::SVector{3,T},
                             body::AbstractCentralBody{T}) where {T<:AbstractFloat}
    r_sun = sun_pos

    # Quick sunlight check: spacecraft on day-side relative to occluder
    if dot(r_sc, r_sun) > T(0)
        return T(1)
    end

    # Cone half-angles
    θ_sun   = asin(SUN_RADIUS(T)   / norm(r_sun))
    θ_body  = asin(body.radius               / norm(r_sun))
    f1 = θ_sun + θ_body  # penumbra half-angle
    f2 = θ_sun - θ_body  # umbra    half-angle

    # Projection of satellite onto shadow axis
    s0 = -dot(r_sc, r_sun) / norm(r_sun)

    # Radial distance from the axis
    l = sqrt(max(zero(T), norm(r_sc)^2 - s0^2))

    # Cone radii at satellite plane
    c1 = s0 + body.radius / sin(f1)
    c2 = s0 - body.radius / sin(f2)
    l1 = c1 * tan(f1)
    l2 = c2 * tan(f2)

    if abs(l) < abs(l2)
        return T(0)
    elseif abs(l) > abs(l1)
        return T(1)
    else
        # smooth transition with tanh
        x = (abs(l) - abs(l2)) / (abs(l1) - abs(l2))
        α = T(6)  # sharpness
        return T(0.5) * (T(1) + tanh((x - T(0.5)) * α))
    end
end

"""
shadow_factor(r_sc, sun_pos) -> T in [0,1]

Convenience wrapper that hard-codes Earth as the occluding body and treats the
Sun position as provided. This is intentionally narrow to keep GPU kernels
simple: if you need other occluders, call `illumination_factor` directly with a
different `AbstractCentralBody` instance. Assumes Earth-only for now.
"""
@inline function shadow_factor(r_sc::SVector{3,T}, sun_pos::SVector{3,T}) where {T<:AbstractFloat}
    earth = EarthEnv{T}()
    return illumination_factor(r_sc, sun_pos, earth)
end

end # module
