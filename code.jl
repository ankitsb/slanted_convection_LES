using Oceananigans
using Oceananigans.Units
using Statistics

using CUDA
using Random
using Printf
using NCDatasets
using SpecialFunctions

using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, VerticalScalarDiffusivity, HorizontalDivergenceScalarDiffusivity
using Oceananigans.Coriolis: SphericalCoriolis, NonhydrostaticFormulation
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Grids: ynodes, znodes, xspacings

Random.seed!(314159) # for reproducible results

arch = GPU()
const R_E = 251e3           # Radius of Enceladus
dx = 4
Nx = 140                   # number of longitude points
Ny = 300                   # number of latitude points (avoiding poles)
Nz = 300                    # number of vertical levels
x = (0, Nx*dx)       # longitude range
y = (0, Ny*dx)     # latitude range (avoiding poles for lat-lon 
H = Nz*dx                    # domain depth [m]
size = (Nx, Ny, Nz)
halo = (4, 4, 4)            # halo size for higher-order advection schemes

z = (-H, 0)                 # vertical extent

recti_grid = RectilinearGrid(arch; size, halo, x, y, z,
                             topology = (Periodic, Periodic, Bounded))

# @inline ice_shell(x, y, z) = @inbounds 10e3 * cos(pi * (y - (2π * R_E * 90 / 360)) / (2π * R_E * 90 / 360)) - 10e3
# grid = ImmersedBoundaryGrid(recti_grid, GridFittedBoundary((x, y, z)-> z > ice_shell(x, y, z)))

grid = recti_grid
@show grid

const S_init = 20
const T_init = 0

Theta = 60
rotation = 1e-4
alpha = 4e-5
g = 0.1
B0 = 4e-14
const TF = B0 / (g * alpha)

tracer_advection = WENO(order=5)
coriolis = NonTraditionalBetaPlane(rotation_rate=rotation, radius=R_E, latitude=Theta)
leos = LinearEquationOfState(thermal_expansion=alpha, haline_contraction=0e-0)
buoyancy =  SeawaterBuoyancy(equation_of_state=leos,
            constant_salinity=S_init, gravitational_acceleration=g) #

@inline T_flux_top(x, y, t) = @inbounds TF
@inline T_flux_bot(x, y, t) = @inbounds TF

T_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(T_flux_top),
                                bottom=FluxBoundaryCondition(T_flux_bot),
)

u_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0.0),
                            bottom=ValueBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0.0),
                            bottom=ValueBoundaryCondition(0.0),)
# w_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(0.0),)

boundary_conditions = (; u=u_bcs, v=v_bcs, T=T_bcs, ) #, S=S_bcs, w=w_bcs, 

# closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=1e-2, κ=1e-3)
closure = DynamicSmagorinsky()#, ScalarDiffusivity(ν=1e-3, κ=1e-4)),

@info "Conditions required for model are done"
model = NonhydrostaticModel(grid; buoyancy,
                            tracers = (:T, ), #:S
                            coriolis = coriolis,
                            advection = tracer_advection,
                            closure = closure,
                            boundary_conditions = boundary_conditions
)
@show model

# Initial conditions
a = 0.8
c = 0.5
s = 0.1

Bell_curve_modified_b(z) = ( 1 - 1 / 2.5a * (erf((-z / model.grid.Lz - c + a/2 + 1.0) / s) - 
                                             erf((-z / model.grid.Lz - c - a/2 - 0.1) / s)) )

# Temperature initial condition: a stable density gradient with random noise superposed.
Tᵢ(x, y, z) = T_init + 1e-4 * (randn()) #* Bell_curve_modified_b(z)

# Velocity initial condition: random noise scaled by the friction velocity.
uᵢ(x, y, z) = 1e-3 * (randn()) #* Bell_curve_modified_b(z) #1e-4 * randn() #+ Ξ(z) )

set!(model, u=uᵢ, v=uᵢ, T=Tᵢ) #, S=S_init, w=uᵢ
@info "Model is all set & initialization is done"

# Simulation runner
wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=15minutes)

stop_time=1500days
save_interval=5days
simulation = Simulation(model; Δt=5, stop_time)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(5))

wall_clock = [time_ns()]

# Progress callback
function progress(sim)
    T = sim.model.tracers.T
    u, v, w = sim.model.velocities

    msg = @sprintf("iter % 4d: % 10s, wall time: %s, Δt: %s, max|u,w|: (%.2e, %.2e) m/s",
                   iteration(sim), prettytime(sim), prettytime(1e-9 * (time_ns() - wall_clock[1])),
                   prettytime(sim.Δt), maximum(abs, u), maximum(abs, w))

    msg *= @sprintf(", T ∈ (%.3f, %.3f)", minimum(T), maximum(T))
    @info msg
    
    # wall_clock[1] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(500))

# Set up output: save vorticity and temperature at the surface
u, v, w = model.velocities
T = model.tracers.T
ζ = ∂x(v) - ∂y(u)
fields = (;u, v, w, ζ, T, )
filename = "slanted_convective_instability"

# simulation.output_writers[:bottom] = NetCDFWriter(model, fields; indices=(:, :, 1), 
#                                     filename=filename * "_bott.nc",
#                                     schedule = TimeInterval(save_interval),
#                                     # overwrite_existing = true
# )
# simulation.output_writers[:top] = NetCDFWriter(model, fields; indices=(:, :, Nz-2), 
#                                     filename=filename * "_top.nc",
#                                     schedule = TimeInterval(save_interval),
#                                     # overwrite_existing = true
# )
simulation.output_writers[:mid] = NetCDFWriter(model, fields; indices=(:, :, convert(Int, Nz/2)), 
                                    filename=filename * "_mid.nc",
                                    schedule = TimeInterval(save_interval),
                                    # overwrite_existing = true
)
simulation.output_writers[:vPlaneLatPole] = NetCDFWriter(model, fields; indices=(:, 1, :), 
                                    filename=filename * "_vPlaneLatPole.nc",
                                    schedule = TimeInterval(save_interval),
                                    # overwrite_existing = true
)
simulation.output_writers[:vPlaneLon] = NetCDFWriter(model, fields; indices=(1, :, :), 
                                    filename=filename * "_vPlaneLon.nc",
                                    schedule = TimeInterval(save_interval),
                                    # overwrite_existing = true
)


simulation.output_writers[:checkpointer] = Checkpointer(model;
                        schedule = TimeInterval(20days),
                        prefix = "checkpoint",
                        overwrite_existing = true, cleanup=true )

@info "Initial output files are written"
run!(simulation; pickup=true)
# run!(simulation)
