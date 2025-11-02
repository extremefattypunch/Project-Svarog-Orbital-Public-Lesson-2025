module Constants

using StaticArrays

# Export constants
export
    # Physical constants
    G, C_LIGHT, SOLAR_CONSTANT, AU,
    # Earth constants
    EARTH_MASS, EARTH_MU, EARTH_RADIUS, EARTH_J2,
    # Sun constants
    SUN_MASS, SUN_MU, SUN_RADIUS,
    # Other celestial bodies
    MOON_MASS, MOON_MU, MOON_RADIUS,
    # SRP constants
    SRP_PRESSURE_1AU,
    # Earth rotation
    EARTH_ROTATION_RATE,
    # Plate normal vector
    PLATE_NORMAL_BODY,
    # Functions to retrieve density constants
    DENSITY_TABLE

# Universal physical constants
"""Gravitational constant [m³/kg/s²]"""
const _G::Float64 = 6.67430e-11
@inline G(::Type{T}) where T<:AbstractFloat = T(_G)

"""Speed of light [m/s]"""
const _C_LIGHT::Float64 = 2.99792458e8
@inline C_LIGHT(::Type{T}) where T<:AbstractFloat = T(_C_LIGHT)

"""Solar constant at 1 AU [W/m²]"""
const _SOLAR_CONSTANT::Float64 = 1361.0
@inline SOLAR_CONSTANT(::Type{T}) where T<:AbstractFloat = T(_SOLAR_CONSTANT)

"""Astronomical Unit [m]"""
const _AU::Float64 = 1.495978707e11
@inline AU(::Type{T}) where T<:AbstractFloat = T(_AU)

# Earth constants
"""Earth mass [kg]"""
const _EARTH_MASS::Float64 = 5.97219e24
@inline EARTH_MASS(::Type{T}) where T<:AbstractFloat = T(_EARTH_MASS)


"""Earth gravitational parameter (G*M) [m³/s²]"""
const _EARTH_MU::Float64 = _G * _EARTH_MASS
@inline EARTH_MU(::Type{T}) where T<:AbstractFloat = T(_EARTH_MU)

"""Earth equatorial radius [m]"""
const _EARTH_RADIUS::Float64 = 6378.1363e3  # Using the more accurate value from constants_LEO
@inline EARTH_RADIUS(::Type{T}) where T<:AbstractFloat = T(_EARTH_RADIUS)

"""Earth J2 coefficient (dimensionless)"""
const _EARTH_J2::Float64 = 1.08262668e-3
@inline EARTH_J2(::Type{T}) where T<:AbstractFloat = T(_EARTH_J2)

"""Earth rotation rate [rad/s]"""
const _EARTH_ROTATION_RATE::Float64 = 7.292115e-5
@inline EARTH_ROTATION_RATE(::Type{T}) where T<:AbstractFloat = T(_EARTH_ROTATION_RATE)

# (Removed unused OMEGA_EARTH_VEC; use scalar EARTH_ROTATION_RATE and construct locally.)

"""Sun mass [kg]"""
const _SUN_MASS::Float64 = 1.9885e30
@inline SUN_MASS(::Type{T}) where T<:AbstractFloat = T(_SUN_MASS)

"""Sun gravitational parameter (G*M) [m³/s²]"""
const _SUN_MU::Float64 = _G * _SUN_MASS
@inline SUN_MU(::Type{T}) where T<:AbstractFloat = T(_SUN_MU)

"""Sun radius [m]"""
const _SUN_RADIUS::Float64 = 696340000.0  # From original constants.jl (R_B)
@inline SUN_RADIUS(::Type{T}) where T<:AbstractFloat = T(_SUN_RADIUS)

# Moon constants
"""Moon mass [kg]"""
const _MOON_MASS::Float64 = 7.342e22
@inline MOON_MASS(::Type{T}) where T<:AbstractFloat = T(_MOON_MASS)

"""Moon gravitational parameter (G*M) [m³/s²]"""
const _MOON_MU::Float64 = _G * _MOON_MASS
@inline MOON_MU(::Type{T}) where T<:AbstractFloat = T(_MOON_MU)

"""Moon radius [m]"""
const _MOON_RADIUS::Float64 = 1.7374e6
@inline MOON_RADIUS(::Type{T}) where T<:AbstractFloat = T(_MOON_RADIUS)

# SRP constants
"""Solar radiation pressure at 1 AU [N/m²]"""
const _SRP_PRESSURE_1AU::Float64 = _SOLAR_CONSTANT / _C_LIGHT # N/m²
@inline SRP_PRESSURE_1AU(::Type{T}) where T<:AbstractFloat = T(_SRP_PRESSURE_1AU)

"""Normal vector for a flat plate in the body frame [m]"""
@inline PLATE_NORMAL_BODY(::Type{T}) where {T<:AbstractFloat} =
    SVector{3,T}(zero(T), zero(T), one(T))


"""Density table for the 1976 US Standard Atmosphere."""
const _DENSITY_TABLE = [
    (0.0,     6.819976,    1.225e+00),
    (100.0,   8.748483,    5.25e-07),
    (150.0,   25.366807,   1.73e-09),
    (200.0,   35.830353,   2.41e-10),
    (250.0,   43.073427,   5.97e-11),
    (300.0,   49.418148,   1.87e-11),
    (350.0,   54.709106,   6.66e-12),
    (400.0,   60.321849,   2.62e-12),
    (450.0,   64.782849,   1.09e-12),
    (500.0,   70.157534,   4.76e-13),
    (550.0,   76.602647,   2.14e-13),
    (600.0,   82.687189,   9.89e-14),
    (650.0,   93.220408,   4.73e-14),
    (700.0,   106.360222,  2.36e-14),
    (750.0,   123.183253,  1.24e-14),
    (800.0,   141.999899,  6.95e-15),
    (850.0,   160.324532,  4.22e-15),
    (900.0,   174.164745,  2.78e-15),
    (950.0,   197.153120,  1.98e-15),
    (1000.0,  253.937111,  1.49e-15),
    (1250.0,  325.664496,  5.70e-16),
    (1500.0,  442.070195,  2.79e-16),
    (2000.0,  541.423094,  9.09e-17),
    (2500.0,  629.634097,  4.23e-17),
    (3000.0,  737.651160,  2.54e-17),
    (3500.0,  873.379787,  1.77e-17),
    (4000.0,  1012.963270, 1.34e-17),
    (4500.0,  1230.230861, 1.06e-17),
    (5000.0,  1467.347729, 8.62e-18),
    (6000.0,  1609.982217, 6.09e-18),
    (7000.0,  1865.398802, 4.56e-18),
    (8000.0,  2112.074786, 3.56e-18),
    (9000.0,  2430.633291, 2.87e-18),
    (10000.0, 3197.970034, 2.37e-18),
    (15000.0, 4702.351347, 1.21e-18),
    (20000.0, 6147.703093, 7.92e-19),
    (25000.0, 7606.199567, 5.95e-19),
    (30000.0, 9072.470056, 4.83e-19),
    (35000.0, 12269.421028,4.13e-19),
    (35786.0, 12269.421028,4.04e-19),  # last point duplicates H from previous
]

# Return as Vector of SVector{3,T} (fast, GPU-friendly small structs)
@inline _CONVERT_DENSITY_TABLE(::Type{T}) where {T<:AbstractFloat} =
    [SVector{3,T}(T(h), T(H), T(ρ)) for (h,H,ρ) in _DENSITY_TABLE]

@inline DENSITY_TABLE(::Type{T}) where {T<:AbstractFloat} = _CONVERT_DENSITY_TABLE(T)

end # module