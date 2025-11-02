#!/usr/bin/env julia
# generate_sma_heatmaps_ltan.jl  —  scan trajectory files from the coarse ensemble
#                               and plot two heat-maps:
#    1) Δa (gain in semi-major axis, km)
#    2) tₘₐₓ (time to reach that gain, weeks)
#                       both on an LTAN × pitch grid
#
# Each trajectory produced by **earth_ensemble_gpu.jl** is written into
#     trajectories_ensemble_updated_gpu/600km_100dt/
# with the filename pattern
#     <RAAN>_<pitch>_traj.jld2
#
# Inside each file: `traj :: TrajectoryData` with .states :: Vector{StateVector}.
# For each file we compute:
#   • Δa   = max(a) - a₀  (in km)
#   • tₘₐₓ = time (weeks) at which max(a) occurs
#
# This version replaces RAAN with LTAN, where:
#   LTAN = ((RAAN - Sun_RA) mod 360) / 15.0   [in hours]
#
# ————————————————————————————————————————————————————————————————

using JLD2, Glob, Printf, Statistics
using SolarSailPropagator          # rv2coe + EarthEnv
using Plots, Measures              # VS Code plot pane
using Base.Filesystem              # for mkpath

# ─────────────────────── User Settings ────────────────────────────────
data_dir     = "examples/trajectories_ensemble_updated_gpu/600km_100dt"
out_dir      = "examples/trajectories_ensemble_updated/600km_100dt"
outer_dt_s   = 100.0
outfile_dA   = joinpath(out_dir, "delta_a_heatmap_ltan.png")
outfile_tmax = joinpath(out_dir, "tmax_heatmap_ltan.png")
verbose      = true
margin_kwargs = (left_margin=10mm, right_margin=10mm, top_margin=10mm, bottom_margin=10mm)
# Set this to Sun RA at epoch t₀ [deg]
sun_ra_deg = 281.70

# Ensure output directory exists
mkpath(out_dir)

# ─────────────────────── Load & Prep ────────────────────────────────
files = Glob.glob("*_traj.jld2", data_dir)
@info "Scanning $(length(files)) trajectory files …"

earth = EarthEnv()
sec_per_week = 7 * 24 * 3600.0

ltan_vals  = Float64[]
pitch_vals = Float64[]
Δa_map     = Dict{Tuple{Float64,Float64},Float64}()
tmax_map   = Dict{Tuple{Float64,Float64},Float64}()

# ─────────────────────── Main Loop ────────────────────────────────
for path in files
    base  = basename(path)
    parts = split(base, "_")
    raan  = parse(Float64, parts[1])
    pitch = parse(Float64, replace(parts[2], "_traj.jld2" => ""))

    # Convert RAAN → LTAN (in hours)
    ltan = mod(raan - sun_ra_deg, 360.0) / 15.0

    push!(ltan_vals,  ltan)
    push!(pitch_vals, pitch)

    a0     = 0.0
    amax   = -Inf
    tmax_w = 0.0

    jldopen(path, "r") do f
        traj = f["traj"]
        states = traj.states
        a0     = rv2coe(states[1].r, states[1].v, earth.mu)[1]
        amax   = a0
        idxmax = 1

        for idx in 1:length(states)
            a = rv2coe(states[idx].r, states[idx].v, earth.mu)[1]
            if a > amax
                amax = a
                idxmax = idx
            end
        end
        tmax_w = (idxmax - 1) * outer_dt_s / sec_per_week
    end

    Δa_km = max(0.0, (amax - a0) / 1e3)

    key = (ltan, pitch)
    Δa_map[key]   = Δa_km
    tmax_map[key] = tmax_w

    if verbose
        @printf("%15s  ▸  LTAN = %5.2f hr, pitch = %6.2f°, Δa = %7.1f km, tₘₐₓ = %6.1f wk\n",
                base, ltan, pitch, Δa_km, tmax_w)
    end
end

# ─────────────────────── Grid Build ────────────────────────────────
unique_ltan  = sort(unique(ltan_vals))
unique_pitch = sort(unique(pitch_vals))

Z_da   = fill(NaN, length(unique_pitch), length(unique_ltan))
Z_tmax = fill(NaN, length(unique_pitch), length(unique_ltan))

for (i, pitch) in enumerate(unique_pitch)
    for (j, ltan) in enumerate(unique_ltan)
        key = (ltan, pitch)
        if haskey(Δa_map, key)
            Z_da[i, j]   = Δa_map[key]
            Z_tmax[i, j] = tmax_map[key]
        end
    end
end

common_fonts = (guidefont = font(16),
                tickfont = font(16),
                legendfont = font(16),
                titlefont = font(18))

plt_da = heatmap(unique_ltan, unique_pitch, Z_da;
    xlabel = "LTAN [hours]",
    ylabel = "Pitch [deg]",
    colorbar_title = "Δa [km]",
    colorbar_titlefont = font(16),
    # title = "Gain in Semi-major Axis (max − initial)",  # remove plot title
    xlims = (minimum(unique_ltan), maximum(unique_ltan)),
    ylims = (minimum(unique_pitch), maximum(unique_pitch)),
    size = (1200, 1000),
    margin_kwargs...,
    common_fonts...)

plt_tm = heatmap(unique_ltan, unique_pitch, Z_tmax;
    xlabel = "LTAN [hours]",
    ylabel = "Pitch [deg]",
    colorbar_title = "Weeks to Δaₘₐₓ",
    colorbar_titlefont = font(16),
    # title = "Time to Reach Max Δa",  # remove plot title
    xlims = (minimum(unique_ltan), maximum(unique_ltan)),
    ylims = (minimum(unique_pitch), maximum(unique_pitch)),
    size = (1200, 1000),
    margin_kwargs...,
    common_fonts...)

savefig(plt_da, outfile_dA)
savefig(plt_tm, outfile_tmax)

display(plt_da)
display(plt_tm)

println("\nLTAN heatmaps saved to $(outfile_dA) and $(outfile_tmax).")
