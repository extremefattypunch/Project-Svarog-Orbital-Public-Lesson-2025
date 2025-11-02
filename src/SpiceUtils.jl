module SpiceUtils

# CPU-ONLY PRECOMPUTE MODULE
#
# This module is for preparing ephemeris and environment data on the CPU.
# Do NOT call any functions in this module from inside ODE dynamics! or
# any GPU kernel. SPICE, Dates, file I/O, and heap allocations live here.
#
# Runtime (CPU/GPU) code should only see numeric, GPU-safe structs and
# should never call SPICE or do file I/O. Prepare caches here, then pass
# typed, immutable values (e.g., SVector{3,T} fixed sun_pos, or a cache
# you index on the CPU path only) into the integrator params.
#
# Sun ephemeris cache overview:
# - SunCache is parametric: SunCache{T<:AbstractFloat}
# - sun_pos(cache::SunCache{T}, t) returns SVector{3,T} by index (no SPICE)
# - build_cache! stores meters in SVector{3,T}
# - load_cache attempts to deserialize old caches; on failure, returns
#   nothing so callers can regenerate.

using SPICE
using Downloads
using Serialization
using JLD2
using Dates
using StaticArrays

export download_de440, get_body_pos, SunCache, save_cache, load_cache, build_cache!, sun_pos, convert_cache

# ------------------------------
# Kernel management (unchanged)
# ------------------------------
const KERNEL_DIR = joinpath(@__DIR__, "..", "kernels")
const KERNELS = [
    ("de440.bsp",
     "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"),
    ("naif0012.tls",
     "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"),
]

const _LOADED = Ref(false)  # private flag
const _ERRCFG = Ref(false)  # CSPICE error handling configured

"""
    _configure_spice_errors_once()

Set CSPICE error action to RETURN and silence error device. This prevents
process-aborting behavior (ABORT) on SPICE errors and turns them into
call-level failures that Julia can catch.
"""
function _configure_spice_errors_once()
    if _ERRCFG[]
        return
    end
    # Only configure if SPICE exposes these (some builds may not)
    if hasproperty(SPICE, :erract) && hasproperty(SPICE, :errdev)
        try
            # Equivalent to erract_c("SET", 0, "RETURN") and errdev_c("SET", 0, "NULL")
            SPICE.erract("SET", 0, "RETURN")
            SPICE.errdev("SET", 0, "NULL")
            _ERRCFG[] = true
        catch
            # Silently ignore; default behavior will apply
        end
    end
end

function download_de440()
    if _LOADED[]
        return nothing  # fast no-op after first call
    end
    _configure_spice_errors_once()
    isdir(KERNEL_DIR) || mkpath(KERNEL_DIR)

    for (fname, url) in KERNELS
        path = joinpath(KERNEL_DIR, fname)
        if !isfile(path)
            @info "Fetching $fname …"
            Downloads.download(url, path)
        end
    SPICE.furnsh(path)
    end

    _LOADED[] = true
    return nothing
end

# ------------------------------
# Typed cache
# ------------------------------
mutable struct SunCache{T<:AbstractFloat}
    step::T                      # [s]
    t0::T                        # unix epoch seconds
    data::Vector{SVector{3,T}}   # Sun position in meters (J2000), Earth-centered

    # Inner constructors
    function SunCache{T}(step::T) where {T<:AbstractFloat}
        new{T}(step, zero(T), SVector{3,T}[])
    end
    function SunCache{T}(step::T, t0::T, data::Vector{SVector{3,T}}) where {T<:AbstractFloat}
        new{T}(step, t0, data)
    end
end

# Convenience outer constructor
SunCache(step::Real; T::Type{<:AbstractFloat}=Float64) = SunCache{T}(T(step))

# Convert between precisions
function convert_cache(c::SunCache{U}, ::Type{T}) where {U<:AbstractFloat, T<:AbstractFloat}
    U === T && return c
    dataT = Vector{SVector{3,T}}(undef, length(c.data))
    @inbounds for i in eachindex(c.data)
        v = c.data[i]
        dataT[i] = SVector{3,T}(T(v[1]), T(v[2]), T(v[3]))
    end
    return SunCache{T}(T(c.step), T(c.t0), dataT)
end

# Build cache over [t_start, t_end] with spacing `cache.step`.
# Stores positions in METERS (converted from SPICE km output).
function build_cache!(cache::SunCache{T}, t_start::Real, t_end::Real) where {T<:AbstractFloat}
    cache.t0 = T(t_start)
    n = Int(floor(((T(t_end) - cache.t0) / cache.step))) + 1
    n = max(n, 1)
    resize!(cache.data, n)

    t = cache.t0
    @inbounds for i in 1:n
        # Dates.unix2datetime expects Float64; this is CPU-side only.
        utc = Dates.format(Dates.unix2datetime(Float64(t)), dateformat"yyyy-mm-ddTHH:MM:SS")
        pos_km = get_body_pos("SUN", utc)                # Vector{Float64}, J2000, km
        # Convert to typed meters
        cache.data[i] = SVector{3,T}(T(pos_km[1]), T(pos_km[2]), T(pos_km[3])) * T(1e3)
        t += cache.step
    end
    return cache
end

# Save/load (with legacy-aware load that falls back cleanly)
function save_cache(cache::SunCache, path::AbstractString)
    if endswith(lowercase(path), ".jld2")
        JLD2.jldsave(path; cache)
    else
        open(path, "w") do io
            serialize(io, cache)
        end
    end
    return nothing
end

"""
    load_cache(path; T=Float64, allow_convert=true) -> Union{SunCache{T},Nothing}

Attempts to deserialize a cache from `path`. If it is a SunCache with a different
precision and `allow_convert=true`, converts it to `T`. If deserialization fails
(e.g., legacy format mismatch), returns `nothing` so callers can regenerate.
"""
function load_cache(path::AbstractString; T::Type{<:AbstractFloat}=Float64, allow_convert::Bool=true)
    if !isfile(path)
        return nothing
    end
    c = nothing
    try
        if endswith(lowercase(path), ".jld2")
            dict = JLD2.load(path)
            c = get(dict, "cache", nothing)
        else
            c = open(path) do io
                deserialize(io)
            end
        end
    catch e
        @warn "Failed to load cache. Will need to rebuild." path exception=(e, catch_backtrace())
        return nothing
    end

    if c isa SunCache{T}
        return c
    elseif c isa SunCache
        return allow_convert ? convert_cache(c, T) : c
    else
        # Duck-typed migration attempt (best-effort)
        try
            step = getfield(c, :step)
            t0   = getfield(c, :t0)
            data = getfield(c, :data)
            if step isa Real && t0 isa Real && data isa AbstractVector
                dataT = Vector{SVector{3,T}}(undef, length(data))
                @inbounds for i in eachindex(data)
                    v = data[i]
                    # handle Vector or SVector of Float64
                    dataT[i] = SVector{3,T}(T(v[1]), T(v[2]), T(v[3]))
                end
                return SunCache{T}(T(step), T(t0), dataT)
            end
        catch
            # ignore and fall through
        end
    @warn "Unrecognized cache contents; will need to rebuild." path
        return nothing
    end
end

# Typed, index-stable accessor (returns SVector{3,T})
@inline function sun_pos(cache::SunCache{T}, t::Real) where {T<:AbstractFloat}
    tT  = T(t)
    idx = round(Int, (tT - cache.t0) / cache.step) + 1
    idx = clamp(idx, 1, length(cache.data))
    return cache.data[idx]
end

# ------------------------------
# SPICE wrapper (unchanged API)
# ------------------------------

"Return position of `target` [km] relative to Earth at `utc` (J2000 frame)."
function get_body_pos(target::AbstractString, utc::AbstractString)
    _configure_spice_errors_once()
    download_de440()                            # ensure kernels are loaded
    try
        et        = SPICE.str2et(utc)
        pos_km, _ = SPICE.spkpos(target, et, "J2000", "NONE", "EARTH")
        return pos_km
    catch e
        # Re-throw as a standard error so callers can handle it.
        error("SPICE get_body_pos failed for $(target) @ $(utc): $(e)")
    end
end

end # module
