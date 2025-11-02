module Analysis

# CPU-only analysis and postprocessing utilities.
# These functions operate on recorded trajectories and may use CPU-only
# helpers (e.g., SPICE via SpiceUtils) for offline computations.

using LinearAlgebra, StaticArrays, Dates, Statistics
using ..OrbitState
using ..Gravity
using ..SpiceUtils
using ..Constants

export find_periapsis_indices, srp_work_components_per_orbit, srp_transverse_work_over_orbit

"""
    find_periapsis_indices(states, times, mu; min_time_between=5000.0, wrap_threshold=15.0)

Detect periapsis passages by looking for true anomaly wrap-around ν ≈ 360 -> 0.
Returns a vector of indices into `states`/`times`.
"""
function find_periapsis_indices(states::Vector{OrbitState.StateVector{T}},
                                times::Vector{T}, mu::T;
                                min_time_between::Real=5000.0,
                                wrap_threshold::Real=15.0) where {T<:AbstractFloat}
    νs_deg = [mod(OrbitState.rv2coe(s.r, s.v, mu)[6], 360) for s in states]
    peri_indices = Int[]
    for i in 2:length(νs_deg)
        if νs_deg[i-1] > (360 - wrap_threshold) && νs_deg[i] < wrap_threshold
            if isempty(peri_indices) || (Float64(times[i] - times[peri_indices[end]]) > Float64(min_time_between))
                push!(peri_indices, i)
            end
        end
    end
    return peri_indices
end

"""
    srp_work_components_per_orbit(times, states, prop)

Compute per-orbit SRP work components (Transverse, Radial, Normal) integrated over each orbit,
using periapsis-to-periapsis segments.

Returns (t_orbit_hr, W_t, W_r, W_h).
"""
function srp_work_components_per_orbit(times::Vector{T},
                                       states::Vector{OrbitState.StateVector{T}},
                                       prop) where {T<:AbstractFloat}
    m  = only(filter(m -> m isa Gravity.SRPModel{T}, prop.models))
    μ  = prop.central_body.mu

    peri = find_periapsis_indices(states, times, μ;
                                  min_time_between = 4000.0,
                                  wrap_threshold   = 10.0)
    length(peri) < 2 && error("Not enough periapsis passages detected")

    Δt = mean(diff(times))
    N = length(times) - 1
    dW_r  = zeros(T, N);  dW_t = zeros(T, N);  dW_h = zeros(T, N)

    @inbounds for i in 1:N
        s       = states[i];  r, v, q = s.r, s.v, s.q
        F_srp   = Gravity.acceleration(m, r, v, q, times[i]) * m.mass
        r̂       = normalize(r)
        ĥ       = normalize(cross(r, v))
        t̂       = normalize(cross(ĥ, r̂))

        dW_r[i] = dot(F_srp, r̂) * dot(v, r̂) * Δt
        dW_t[i] = dot(F_srp, t̂) * dot(v, t̂) * Δt
        dW_h[i] = dot(F_srp, ĥ) * dot(v, ĥ) * Δt
    end

    sum_per_orbit(dW) = [sum(dW[peri[k] : peri[k+1]-1]) for k in 1:length(peri)-1]

    W_r  = sum_per_orbit(dW_r)
    W_t  = sum_per_orbit(dW_t)
    W_h  = sum_per_orbit(dW_h)

    t_orbit_hr = (times[peri[2:end]] .- times[peri[1]]) ./ 3600
    return t_orbit_hr, W_t, W_r, W_h
end

"""
    srp_transverse_work_over_orbit(times, states, prop, t)

Compute cumulative SRP transverse work components over the single orbit that brackets time `t`.
Returns (t_hr, inbound_work, outbound_work, net_work), each a Vector{T}.

This function is CPU-only and may use SPICE (via SpiceUtils) when the SRP model
has no cache; callers should pass an SRPModel with a cache for reproducibility.
"""
function srp_transverse_work_over_orbit(times::Vector{T},
                                        states::Vector{OrbitState.StateVector{T}},
                                        prop,
                                        t::T) where {T<:AbstractFloat}
    m = only(filter(m -> m isa Gravity.SRPModel{T}, prop.models))
    sun_cache = m.cache
    μ = prop.central_body.mu

    # Periapsis detection
    peri_idxs = find_periapsis_indices(states, times, μ; min_time_between=4000.0)
    length(peri_idxs) < 2 && error("Not enough periapsis indices.")

    # Bracket t by peri-to-peri segment
    closest_idx = argmin(abs.(times .- t))
    t1, t2 = nothing, nothing
    for i in 1:length(peri_idxs)-1
        if peri_idxs[i] <= closest_idx < peri_idxs[i+1]
            t1, t2 = peri_idxs[i], peri_idxs[i+1]
            break
        end
    end
    if t1 === nothing || t2 === nothing
        @warn "Could not bracket t=$(t). Falling back to first full orbit."
        t1, t2 = peri_idxs[1], peri_idxs[2]
    end

    # Slice exactly one orbit
    times1  = times[t1:t2]
    states1 = states[t1:t2]
    dt = diff(times1); push!(dt, dt[end])

    inbound_work = zeros(T, length(times1))
    outbound_work = zeros(T, length(times1))
    net_work = zeros(T, length(times1))
    W_in, W_out = zero(T), zero(T)

    @inbounds for i in eachindex(states1)
        s = states1[i]
        r, v, q = s.r, s.v, s.q
        F_srp = Gravity.acceleration(m, r, v, q, times1[i]) * m.mass
        r̂ = normalize(r)
        v_trans = v - dot(v, r̂) * r̂
        dW = dot(F_srp, v_trans) * dt[i]

        sun_r = sun_cache === nothing ?
            SVector{3,T}(SpiceUtils.get_body_pos("SUN", Dates.format(Dates.unix2datetime(Float64(times1[i])), dateformat"yyyy-mm-ddTHH:MM:SS")) .* T(1e3)) :
            SpiceUtils.sun_pos(sun_cache, times1[i])
        sun_dir = normalize(sun_r - r)
        v_radial_to_sun = dot(v, sun_dir)

        if v_radial_to_sun > zero(T)
            W_out += dW
        elseif v_radial_to_sun < zero(T)
            W_in += dW
        end
        inbound_work[i]  = W_in
        outbound_work[i] = W_out
        net_work[i]      = W_in + W_out
    end

    t_hr = (times1 .- times1[1]) ./ 3600
    return t_hr, inbound_work, outbound_work, net_work
end

end # module
