module Attitude

using StaticArrays, LinearAlgebra

export quat_normalize, quat_to_rotmat, rotmat_to_quat, quat_derivative

# Normalize quaternion; if near-zero, return identity
@inline function quat_normalize(q::SVector{4,T}) where {T<:AbstractFloat}
    n = norm(q)
    if isapprox(n, zero(T); atol=T(1e-12))
        return SVector{4,T}(one(T), zero(T), zero(T), zero(T))
    end
    return q / n
end

# Quaternion derivative given angular velocity ω (body rates)
# q̇ = 0.5 * [  0   -ωᵀ; ω  -[ω×] ] q
@inline function quat_derivative(q::SVector{4,T}, ω::SVector{3,T}) where {T<:AbstractFloat}
    w, x, y, z = q
    wx, wy, wz = ω
    return SVector{4,T}(
        T(0.5) * (-wx*x - wy*y - wz*z),
        T(0.5) * ( wx*w + wz*y - wy*z),
        T(0.5) * ( wy*w - wz*x + wx*z),
        T(0.5) * ( wz*w + wy*x - wx*y),
    )
end

# Quaternion (w,x,y,z) to rotation matrix (body -> inertial)
@inline function quat_to_rotmat(q::SVector{4,T}) where {T<:AbstractFloat}
    w, x, y, z = q
    xx = x*x; yy = y*y; zz = z*z
    wx = w*x; wy = w*y; wz = w*z
    xy = x*y; xz = x*z; yz = y*z
    return @SMatrix [
        T(1) - T(2)*(yy + zz)    T(2)*(xy - wz)          T(2)*(xz + wy);
        T(2)*(xy + wz)           T(1) - T(2)*(xx + zz)   T(2)*(yz - wx);
        T(2)*(xz - wy)           T(2)*(yz + wx)          T(1) - T(2)*(xx + yy)
    ]
end

# Rotation matrix to quaternion (w,x,y,z)
@inline function rotmat_to_quat(R::SMatrix{3,3,T}) where {T<:AbstractFloat}
    tr = R[1,1] + R[2,2] + R[3,3]
    if tr > T(0)
        S = sqrt(tr + T(1)) * T(2)
        qw = T(0.25) * S
        qx = (R[3,2] - R[2,3]) / S
        qy = (R[1,3] - R[3,1]) / S
        qz = (R[2,1] - R[1,2]) / S
    elseif (R[1,1] > R[2,2]) && (R[1,1] > R[3,3])
        S = sqrt(T(1) + R[1,1] - R[2,2] - R[3,3]) * T(2)
        qw = (R[3,2] - R[2,3]) / S
        qx = T(0.25) * S
        qy = (R[1,2] + R[2,1]) / S
        qz = (R[1,3] + R[3,1]) / S
    elseif R[2,2] > R[3,3]
        S = sqrt(T(1) + R[2,2] - R[1,1] - R[3,3]) * T(2)
        qw = (R[1,3] - R[3,1]) / S
        qx = (R[1,2] + R[2,1]) / S
        qy = T(0.25) * S
        qz = (R[2,3] + R[3,2]) / S
    else
        S = sqrt(T(1) + R[3,3] - R[1,1] - R[2,2]) * T(2)
        qw = (R[2,1] - R[1,2]) / S
        qx = (R[1,3] + R[3,1]) / S
        qy = (R[2,3] + R[3,2]) / S
        qz = T(0.25) * S
    end
    return quat_normalize(SVector{4,T}(qw, qx, qy, qz))
end

end # module
