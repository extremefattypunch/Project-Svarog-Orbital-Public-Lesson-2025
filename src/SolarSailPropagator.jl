module SolarSailPropagator

# ── Sub-modules ───────────────────────────────────────────────────
include("Constants.jl")
include("CentralBody.jl")
include("Attitude.jl")
include("Illumination.jl")
include("SpiceUtils.jl")
include("OrbitState.jl")
include("Gravity.jl")
include("Torque.jl")
include("Control.jl")
include("Analysis.jl")
include("Propagator.jl")       # contains PropagatorType / propagate!
include("Plotting.jl")

# ── Bring sub-modules into the umbrella namespace ────────────────
using .Constants
using .CentralBody
using .Attitude
using .Illumination
using .SpiceUtils
using .OrbitState
using .Gravity
using .Torque
using .Control
using .Analysis
using .Propagator
using .Plotting

# ── Export only the high-level, user-facing API ──────────────────
export
    # constants (minimal set most scripts need)
    EARTH_MU, EARTH_RADIUS, EARTH_J2, AU, PLATE_NORMAL_BODY,

    # central-body types
    AbstractCentralBody, EarthEnv, SunEnv,

    #SpiceUtils
    download_de440, get_body_pos, SunCache, save_cache, load_cache, build_cache!, sun_pos,

    # state and conversions
    StateVector, coe2rv, rv2coe, quaternion_to_rotation_matrix,
    rotmat_to_quat,

    # Gravity
    AbstractForceModel, GravityModel, SRPModel, acceleration, AtmosphericDragModel, nrlmsise_density, AlbedoModel, gravity_acceleration, srp_acceleration,

    # Illumination
    illumination_factor, shadow_factor,

    # Torque
    AbstractTorqueModel, torque, SRPTorqueModel, AtmosphericDragTorqueModel, GravityGradientTorqueModel, AlbedoTorqueModel,

    # Control
    AbstractControlLaw, apply_control!, SunFeatherLaw, fixed_in_plane_attitude, angular_velocity_along_body_z,

    # inertia helpers
    plate_inertia, rod_inertia, parallel_axis,

    # Propagator
    PropagatorType, propagate, add_force!, TrajectoryData, add_torque!, dynamics!, make_params, DynamicsParams,

    # Plotting
    plot_orbit_3d, plot_altitude, plot_orbital_elements, plot_attitude, 
    plot_inclination_per_orbit, plot_raan_per_periapsis, plot_torque_contributions, 
    plot_force_contributions, plot_srp_contribution, plot_sail_plane_angle,
    plot_srp_transverse_work, plot_srp_transverse_work_per_orbit, plot_srp_radial_work_per_orbit,
    plot_srp_normal_work_per_orbit

end # module
