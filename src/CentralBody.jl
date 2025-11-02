module CentralBody

using ..Constants

export AbstractCentralBody, EarthEnv, SunEnv

"""
    AbstractCentralBody

Abstract type for central bodies (Earth, Sun, etc.).
"""
abstract type AbstractCentralBody{T<:AbstractFloat} end

"""
EarthEnv{T}

Lightweight, GPU-safe numeric environment for Earth. Pure data – no SPICE,
no time-varying state. All fields are parametric in `T<:AbstractFloat` to
support `Float32` / `Float64` kernels.

Fields:
    * mu::T            Standard gravitational parameter [m^3 s^-2]
    * radius::T        Equatorial radius [m]
    * j2::T            J2 zonal coefficient (dimensionless)
    * rotation_rate::T Sidereal rotation rate about +Z [rad s^-1]
"""
struct EarthEnv{T<:AbstractFloat} <: AbstractCentralBody{T}
    mu::T
    radius::T
    j2::T
    rotation_rate::T

    """EarthEnv{T}(): Construct with canonical constant values (see Constants)."""
    EarthEnv{T}() where {T<:AbstractFloat} = new{T}(
        Constants.EARTH_MU(T),
        Constants.EARTH_RADIUS(T),
        Constants.EARTH_J2(T),
        Constants.EARTH_ROTATION_RATE(T),
    )
end

"""
SunEnv{T}

Lightweight, GPU-safe numeric environment for the Sun. Pure constants.
No rotation / J2 modeled (fields set to zero for structural consistency).
"""
struct SunEnv{T<:AbstractFloat} <: AbstractCentralBody{T}
    mu::T
    radius::T
    j2::T
    rotation_rate::T

    """SunEnv{T}(): Construct with canonical constant values (see Constants)."""
    SunEnv{T}() where {T<:AbstractFloat} = new{T}(
        Constants.SUN_MU(T),
        Constants.SUN_RADIUS(T),
        zero(T),
        zero(T),
    )
end

# Default Float64 outer constructors for convenience
EarthEnv() = EarthEnv{Float64}()
SunEnv()   = SunEnv{Float64}()

end # module