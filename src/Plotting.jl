module Plotting

using Plots, JLD2, Measures, LinearAlgebra, StaticArrays, Dates, Statistics
using ..OrbitState, ..CentralBody, ..Constants, ..Gravity, ..Torque, ..SpiceUtils, ..Propagator, ..Analysis

export plot_orbit_3d, plot_altitude, plot_orbital_elements, plot_attitude,
       plot_inclination_per_orbit, plot_raan_per_periapsis,
       plot_force_contributions, plot_torque_contributions, plot_srp_contribution,
       plot_sail_plane_angle, plot_srp_transverse_work, plot_srp_transverse_work_per_orbit, plot_srp_radial_work_per_orbit,
       plot_srp_normal_work_per_orbit

const _MAXPTS = 150_000
const _FONT_KWARGS = (
    guidefont = font(14),   # axis labels
    tickfont  = font(12),   # tick labels
    legendfont = font(12)   # legend entries
)
const _MARGIN_KWARGS = (left_margin=10mm, right_margin=10mm, top_margin=10mm, bottom_margin=10mm)

function plot_with_margin(args...; kwargs...)
    plot(title="\u00a0\n",args...; kwargs..., _MARGIN_KWARGS..., _FONT_KWARGS...)
end

function _thin_idxs(n; maxpts=_MAXPTS)
    n ≤ maxpts && return 1:n
    step = ceil(Int, n / maxpts)
    idxs = collect(1:step:n)
    idxs[end] == n || push!(idxs, n)
    idxs
end

function _load_traj(traj::TrajectoryData; maxpts=_MAXPTS)
    times = traj.times
    states = traj.states
    n = length(times)
    keep = _thin_idxs(n; maxpts=maxpts)
    times = times[keep]
    states = states[keep]
    return times, states
end

function plot_orbit_3d(file::TrajectoryData, body; show_body=true, maxpts=_MAXPTS)
    times, states = _load_traj(file; maxpts=maxpts)
    x = [s.r[1] for s in states]
    y = [s.r[2] for s in states]
    z = [s.r[3] for s in states]

    tmin, tmax = minimum(times), maximum(times)
    tnorm = (times .- tmin) ./ (tmax - tmin)

    plt = plot_with_margin(size=(900, 750),
                           xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]",
                           title="Spacecraft Trajectory", legend=false)

    if show_body
        r = body.radius
        u, v = range(0, 2π; length=30), range(0, π; length=30)
        xs = r .* [cos(ui)*sin(vj) for ui in u, vj in v]
        ys = r .* [sin(ui)*sin(vj) for ui in u, vj in v]
        zs = r .* [cos(vj)         for ui in u, vj in v]
        plot!(plt, xs, ys, zs, seriestype=:surface, color=:lightblue, alpha=1)
    end

    plot!(plt, x, y, z, line_z=tnorm, linewidth=3, c=:viridis, clim=(0.0, 1.0))

    return plt
end

function plot_altitude(file, body; maxpts=_MAXPTS)
    times, states = _load_traj(file; maxpts=maxpts)
    alt = [norm(s.r) - body.radius for s in states]
    t = times ./ 3600
    plot_with_margin(t, alt ./ 1000,
        xlabel="Time [hours]", ylabel="Altitude [km]",
        title="Spacecraft Altitude vs Time", linewidth=2, grid=true, size=(1200, 1000))
end

function plot_orbital_elements(file, body; maxpts=_MAXPTS)
    times, states = _load_traj(file; maxpts=maxpts)
    a, e, i, Ω, ω, ν = Float64[], Float64[], Float64[], Float64[], Float64[], Float64[]
    for s in states
        A, E, I, O, W, N = rv2coe(s.r, s.v, body.mu)
        push!(a, A); push!(e, E); push!(i, I); push!(Ω, O); push!(ω, W); push!(ν, N)
    end
    t = times ./ 3600  # time in hours

    # Individual subplots (no xlabel in them — only 1 global xlabel at the bottom)
    p1 = plot_with_margin(t, a ./ 1000, ylabel="a [km]", title="Semi-major axis", legend=false)
    p2 = plot_with_margin(t, e,         ylabel="e",      title="Eccentricity",    legend=false)
    p3 = plot_with_margin(t, i,         ylabel="i [deg]",title="Inclination",     legend=false)
    p4 = plot_with_margin(t, Ω,         ylabel="Ω [deg]",title="RAAN",            legend=false)

    # Combine plots into 2x2 layout, shared xlabel only
    return plot(p1, p2, p3, p4;
                layout = (2, 2),
                size = (1400, 1000),
                xlabel = "Time [hours]",
                _MARGIN_KWARGS...)
end

quat_conj(q) = ( q[1], -q[2], -q[3], -q[4] )

function quat_mul(a, b)
    a0,a1,a2,a3 = a; b0,b1,b2,b3 = b
    (
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0
    )
end

function plot_attitude(file; maxpts=_MAXPTS)
    times, states = _load_traj(file; maxpts=maxpts)
    q_ref = states[1].q
    q_ref_conj = quat_conj(q_ref)

    yaw = Float64[]; pitch = Float64[]; roll = Float64[]
    ωx = Float64[]; ωy = Float64[]; ωz = Float64[]

    for s in states
        q_rel = quat_mul(s.q, q_ref_conj)
        q0,q1,q2,q3 = q_rel
        ψ = atan(2(q0*q3 + q1*q2), 1 - 2(q2^2 + q3^2))
        θ = asin(clamp(2(q0*q2 - q3*q1), -1, 1))
        φ = atan(2(q0*q1 + q2*q3), 1 - 2(q1^2 + q2^2))
        push!(yaw, rad2deg(ψ)); push!(pitch, rad2deg(θ)); push!(roll, rad2deg(φ))
        ω = s.ω
        push!(ωx, rad2deg(ω[1])); push!(ωy, rad2deg(ω[2])); push!(ωz, rad2deg(ω[3]))
    end

    t = times ./ 3600
    p1 = plot_with_margin(t, yaw, ylabel="ΔYaw [deg]", title="Yaw (Δ from t₀)", legend=false)
    p2 = plot_with_margin(t, pitch, ylabel="ΔPitch [deg]", title="Pitch (Δ from t₀)", legend=false)
    p3 = plot_with_margin(t, roll, ylabel="ΔRoll [deg]", title="Roll (Δ from t₀)", legend=false)
    p4 = plot_with_margin(t, ωx, label="ω_x", ylabel="Angular Velocity [deg/s]", title="Body Rates")
    plot!(p4, t, ωy, label="ω_y")
    plot!(p4, t, ωz, label="ω_z")
    plot(p1, p2, p3, p4; layout=(4,1), size=(1200,1200), xlabel="Time [hours]", _MARGIN_KWARGS...)
end

function _plot_contrib(file, prop, items, f, ylab; maxpts=_MAXPTS, include_gravity=false, logscale=false)
    times, states = _load_traj(file; maxpts=maxpts)
    t = times ./ 3600
    yscale = logscale ? :log10 : :linear
    plt = plot_with_margin(xlabel="Time [hours]", ylabel=ylab, yscale=yscale, size=(1200, 800))

    for m in items
        values = [norm(f(m, s, τ)) for (s, τ) in zip(states, times)]
        plot!(plt, t, values, label=string(typeof(m)))
    end

    if include_gravity
        gravity_model = GravityModel(prop.central_body)
        values = [norm(acceleration(gravity_model, s.r, s.v, s.q, τ)) for (s, τ) in zip(states, times)]
        plot!(plt, t, values, label="GravityModel")
    end

    return plt
end

function plot_force_contributions(file, prop; maxpts=_MAXPTS)
    _plot_contrib(file, prop, prop.models,
        (m, s, τ) -> acceleration(m, s.r, s.v, s.q, τ),
        "‖a‖ [m/s²]";
        maxpts=maxpts,
        include_gravity=true,
        logscale=true)
end

function plot_torque_contributions(file, prop; maxpts=_MAXPTS)
    _plot_contrib(file, prop, prop.torques,
        (m, s, τ) -> torque(m, s.r, s.v, s.q, s.ω, τ),
        "‖τ‖ [N·m]";
        maxpts=maxpts,
        logscale=false)
end

function plot_srp_contribution(file, prop; maxpts=_MAXPTS)
    times, states = _load_traj(file; maxpts=maxpts)
    t = times ./ 3600
    plt = plot_with_margin(xlabel="Time [hours]", ylabel="‖aₛᵣₚ‖ [m/s²]", yscale=:linear,
                           title="SRP Force Contribution", size=(1200, 800), legend=false)

    for m in prop.models
        if m isa SRPModel
            values = [norm(acceleration(m, s.r, s.v, s.q, τ)) for (s, τ) in zip(states, times)]
            plot!(plt, t, values, label="SRPModel")
        end
    end

    return plt
end

function _peri_plot(file, body, getval, ylabel, title; maxpts=_MAXPTS)
    times, states = _load_traj(file; maxpts=maxpts)
    peri = Analysis.find_periapsis_indices(states, times, body.mu)
    tim = (times[peri] .- times[peri[1]]) ./ 86400
    val = [getval(states[i]) for i in peri]
    plot_with_margin(tim, val, marker=:circle, xlabel="Time [days]", ylabel=ylabel, title=title, legend=false)
end

plot_inclination_per_orbit(file, body; maxpts=_MAXPTS) =
    _peri_plot(file, body, s -> rv2coe(s.r, s.v, body.mu)[3], "Inclination [deg]", "Inclination at Periapsis (ν≈0)"; maxpts)

plot_raan_per_periapsis(file, body; maxpts=_MAXPTS) =
    _peri_plot(file, body, s -> rv2coe(s.r, s.v, body.mu)[4], "RAAN [deg]", "RAAN at Periapsis (ν≈0)"; maxpts)

function plot_sail_plane_angle(file; maxpts=_MAXPTS)
    times, states = _load_traj(file; maxpts=maxpts)
    angles = Float64[]
    for s in states
        ĥ = normalize(cross(s.r, s.v))
        R = quaternion_to_rotation_matrix(s.q)
        ẑ = R[:, 3]
        cosθ = clamp(dot(ẑ, ĥ), -1.0, 1.0)
        push!(angles, rad2deg(acos(cosθ)))
    end
    t = times ./ 3600
    plot_with_margin(t, angles,
        xlabel="Time [hours]", ylabel="Angle [deg]",
        title="Sail Normal vs Orbital Plane Normal", grid=true,
        size=(1200, 800))
end

function plot_srp_transverse_work(file, prop, t; maxpts=_MAXPTS)
    times, states = _load_traj(file; maxpts=maxpts)
    t_hr, inbound_work, outbound_work, net_work = Analysis.srp_transverse_work_over_orbit(times, states, prop, t)

    plt = plot_with_margin(xlabel="Time [hours]",
                           ylabel="Cumulative Transverse SRP Work [J]",
                           title="SRP Work near t = $(round(t / 2_592_000, digits=2)) months",
                           size=(1200, 800),
                           grid=true)
    plot!(plt, t_hr, inbound_work, label="Sun-Inbound", color=:red, linewidth=2)
    plot!(plt, t_hr, outbound_work, label="Sun-Outbound", color=:green, linewidth=2)
    plot!(plt, t_hr, net_work, label="Net Work", color=:blue, linewidth=2)
    return plt
end

# ──────────────────────────────────────────────────────────────────────────────
#  Internal helper –- NO filtering
# ──────────────────────────────────────────────────────────────────────────────
function _srp_work_components_per_orbit(file, prop; maxpts=_MAXPTS)
    times, states = _load_traj(file; maxpts=maxpts)
    return Analysis.srp_work_components_per_orbit(times, states, prop)
end


# ──────────────────────────────────────────────────────────────────────────────
#  Public plotting wrappers – each with its own Wmax filter
# ──────────────────────────────────────────────────────────────────────────────
function plot_srp_transverse_work_per_orbit(file, prop; maxpts=_MAXPTS,
                                            Wmax=5000.0)
    t, W_t, _, _ = _srp_work_components_per_orbit(file, prop; maxpts)
    keep = findall(abs.(W_t) .<= Wmax)

    plt = plot_with_margin(xlabel="Time since first periapsis [h]",
                           ylabel="Net transverse SRP work per orbit [J]",
                           title ="Transverse (in-plane) SRP Work",
                           size=(1200,800), grid=true)

    plot!(plt, t[keep], W_t[keep];
          seriestype=:steppost, linewidth=2, color=:green, label="Transverse")
    return plt
end


function plot_srp_radial_work_per_orbit(file, prop; maxpts=_MAXPTS,
                                        Wmax=5.0)
    t, _, W_r, _ = _srp_work_components_per_orbit(file, prop; maxpts)
    keep = findall(abs.(W_r) .<= Wmax)

    plt = plot_with_margin(xlabel="Time since first periapsis [h]",
                           ylabel="Net radial SRP work per orbit [J]",
                           title ="Radial SRP Work",
                           size=(1200,800), grid=true)

    plot!(plt, t[keep], W_r[keep];
          seriestype=:steppost, linewidth=2, color=:blue, label="Radial")
    return plt
end


function plot_srp_normal_work_per_orbit(file, prop; maxpts=_MAXPTS,
                                        Wmax=5000.0)
    t, _, _, W_h = _srp_work_components_per_orbit(file, prop; maxpts)
    keep = findall(abs.(W_h) .<= Wmax)

    plt = plot_with_margin(xlabel="Time since first periapsis [h]",
                           ylabel="Net normal-to-plane SRP work per orbit [J]",
                           title ="Normal (out-of-plane) SRP Work",
                           size=(1200,800), grid=true)

    plot!(plt, t[keep], W_h[keep];
          seriestype=:steppost, linewidth=2, color=:red, label="Normal")
    return plt
end


end # module Plotting