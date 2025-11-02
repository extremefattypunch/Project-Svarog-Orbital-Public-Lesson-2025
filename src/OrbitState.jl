module OrbitState

using LinearAlgebra
using StaticArrays
using ..Attitude

export StateVector, coe2rv, rv2coe, quaternion_to_rotation_matrix

"""
    StateVector

Represents the state of a spacecraft in Cartesian coordinates, including quaternion and angular velocity.

# Fields
- `r::SVector{3,Float64}`: Position vector [m]
- `v::SVector{3,Float64}`: Velocity vector [m/s]
- `q::SVector{4,Float64}`: Quaternion (attitude)
- `ω::SVector{3,Float64}`: Angular velocity [rad/s]
"""
struct StateVector{T<:AbstractFloat}
    r :: SVector{3,T}
    v :: SVector{3,T}
    q :: SVector{4,T}
    ω :: SVector{3,T}
    function StateVector{T}(r::SVector{3,T}, v::SVector{3,T}, q::SVector{4,T}, ω::SVector{3,T}) where {T<:AbstractFloat}
        new{T}(r, v, q, ω)
    end
end

# Parametric outer constructor
StateVector{T}(r::AbstractVector, v::AbstractVector,
               q::AbstractVector, ω::AbstractVector) where {T<:AbstractFloat} =
    StateVector{T}( SVector{3,T}(r...),
                    SVector{3,T}(v...),
                    normalize(SVector{4,T}(q...)),
                    SVector{3,T}(ω...) )

# Infer T from inputs
StateVector(r::AbstractVector, v::AbstractVector, q::AbstractVector, ω::AbstractVector) =
    StateVector{promote_type(eltype(r), eltype(v), eltype(q), eltype(ω))}(r, v, q, ω)

"""
    StateVector(a, e, i, Ω, ω, ν, μ)

Construct a StateVector from orbital elements with default attitude (identity quaternion and zero angular velocity).
"""
function StateVector(a::T, e::T, i::T, Ω::T, ω::T, ν::T, μ::T) where {T<:AbstractFloat}
    r, v = coe2rv(a, e, i, Ω, ω, ν, μ)
    q = SVector{4,T}(T(1.0), T(0.0), T(0.0), T(0.0))
    ω = SVector{3,T}(T(0.0), T(0.0), T(0.0))
    return StateVector(r, v, q, ω)
end

function coe2rv(a::T, e::T, i::T, Ω::T, ω::T, ν::T, μ::T) where {T<:AbstractFloat}
    i_rad = deg2rad(i)
    Ω_rad = deg2rad(Ω)
    ω_rad = deg2rad(ω)
    ν_rad = deg2rad(ν)

    p = a * (T(1) - e^2)
    r_pf = (p / (1 + e * cos(ν_rad))) * @SVector [cos(ν_rad), sin(ν_rad), T(0.0)]
    v_pf = sqrt(μ / p) * @SVector [-sin(ν_rad), e + cos(ν_rad), T(0.0)]

    function R1(θ)
        @SMatrix [
            T(1.0) T(0.0) T(0.0);
            T(0.0) cos(θ) -sin(θ);
            T(0.0) sin(θ)  cos(θ)
        ]
    end

    function R3(θ)
        @SMatrix [
            cos(θ)  sin(θ) T(0.0);
           -sin(θ)  cos(θ) T(0.0);
            T(0.0)     T(0.0) T(1.0)
        ]
    end

    Q = R3(-Ω_rad) * R1(-i_rad) * R3(-ω_rad)
    return Q * r_pf, Q * v_pf
end

function rv2coe(r::AbstractVector, v::AbstractVector, μ::T) where {T<:AbstractFloat}
    r_vec = SVector{3,T}(r...)
    v_vec = SVector{3,T}(v...)

    r_mag = norm(r_vec)
    v_mag = norm(v_vec)

    h_vec = cross(r_vec, v_vec)
    h_mag = norm(h_vec)

    n_vec = cross(SVector{3,T}(T(0.0), T(0.0), T(1.0)), h_vec)
    n_mag = norm(n_vec)

    e_vec = ((v_mag^2 - μ / r_mag) * r_vec - dot(r_vec, v_vec) * v_vec) / μ
    e = norm(e_vec)

    a = h_mag^2 / (μ * (1 - e^2))
    i = acos(clamp(h_vec[3] / h_mag, T(-1.0), T(1.0)))

    Ω = n_mag < T(1e-10) ? T(0.0) : acos(clamp(n_vec[1] / n_mag, T(-1.0), T(1.0)))
    if n_vec[2] < 0; Ω = 2π - Ω; end

    if n_mag < T(1e-10)
        ω = atan(e_vec[2], e_vec[1])
    else
        ω = acos(clamp(dot(n_vec, e_vec) / (n_mag * e), T(-1.0), T(1.0)))
        if e_vec[3] < 0; ω = 2π - ω; end
    end

    ν = acos(clamp(dot(e_vec, r_vec) / (e * r_mag), T(-1.0), T(1.0)))
    if dot(r_vec, v_vec) < 0; ν = 2π - ν; end

    return a, e, rad2deg(i), rad2deg(Ω), rad2deg(ω), rad2deg(ν)
end

# Keep public API name but delegate to Attitude
@inline function quaternion_to_rotation_matrix(q::SVector{4,T}) where {T<:AbstractFloat}
    return quat_to_rotmat(q)
end

end # module
