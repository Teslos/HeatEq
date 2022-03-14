# The 3D PDE is given by:
# uₜ = uₓₓ + u_yy + u_zz
# For boundary conditions we have Dirichlet conditions:
# u(t,x=0,y,z) = u(t,x=1,y,z)   = 1    x - direction
# u(t,x,y=0,z) = u(t,x,y=1,z)   = 0    y - direction
# u_z(t,x,y,z=0) + u(t,x,y,z=0) = 1    z - direction
# u_z(t,x,y,z=1) + u(t,x,y,z=1) = 1

using OrdinaryDiffEq, ModelingToolkit, DiffEqOperators
using ModelingToolkit: Differential, infimum,supremum,Interval

anal_sol_funct(t,x,y,z) = 1.0

@parameters t x y z
@variables  u(..)

Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dzz = Differential(z)^2
Dy  = Differential(y)
Dz  = Differential(z)
Dt  = Differential(t)

# Time interval
t₀ = 0.0
t₁ = 1.0
x_max = 1.0
x_min = 0.0
y_max = 1.0
y_min = 0.0
z_max = 1.0
z_min = 0.0

# 3D equation
eq = Dt(u(t,x,y,z)) ~ Dxx(u(t,x,y,z)) + Dyy(u(t,x,y,z)) + Dzz(u(t,x,y,z))
# Initial and boundary conditions
bcs = [u(t₀, x_min, y, z) ~ 1.0, u(t₁,x_max,y,z) ~ 1.0,    # pseudo condition
       u(t,x_min,y,z) ~ 1.0, u(t,x_max,y,z) ~ 1.0,         # x direction
       Dy(u(t,x,y_min,z)) ~ 0.0, Dy(u(t,x,y_max,z)) ~ 0.0, # y direction
       Dz(u(t,x,y,z_min)) + u(t,x,y,z_min) ~ 1.0,          # z direction
       Dz(u(t,x,y,z_max)) + u(t,x,y,z_max) ~ 1.0 ]
domains = [x ∈ Interval(x_min, x_max),
           y ∈ Interval(y_min, y_max),
           z ∈ Interval(z_min, z_max),
           t ∈ Interval(t₀, t₁)]
@named pdesys = PDESystem([eq], bcs, domains, [t, x, y, z], [u(t, x, y, z)])

dx = 0.1
dy = 0.1
dz = 0.1

order = 2
discretization =
    MOLFiniteDifference([x => dx, y => dy, z => dz], t)

prob = discretize(pdesys, discretization)
sol  = solve(prob, Tsit5())
