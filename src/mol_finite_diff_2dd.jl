
using ModelingToolkit, MethodOfLines, LinearAlgebra, Test, OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential


# Variables, parameters, and derivatives
@parameters t x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)
t_min = 0.0
t_max = 2.0
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 2.0
dx = 0.1
dy = 0.2
order = 4

# Analytic solution
analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)

# Equation
eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

# Initial and boundary conditions
bcs = [u(t_min, x, y) ~ analytic_sol_func(t_min, x, y),
    u(t, x_min, y) ~ analytic_sol_func(t, x_min, y),
    u(t, x_max, y) ~ analytic_sol_func(t, x_max, y),
    u(t, x, y_min) ~ analytic_sol_func(t, x, y_min),
    u(t, x, y_max) ~ analytic_sol_func(t, x, y_max)]

# Space and time domains
domains = [t ∈ Interval(t_min, t_max),
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max)]

# Space and time domains
@named pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])


# Test against exact solution
Nx = floor(Int64, (x_max - x_min) / dx) + 1
Ny = floor(Int64, (y_max - y_min) / dy) + 1

# Method of lines discretization
discretization = MOLFiniteDifference([x => dx, y => dy], t; approx_order=order)
prob = ModelingToolkit.discretize(pdesys, discretization)
# Solution of the ODE system
sol = solve(prob, Tsit5())
r_space_x = sol[x]
r_space_y = sol[y]
asf = [analytic_sol_func(t_max, X, Y) for X in r_space_x, Y in r_space_y]
asf[1, 1] = asf[1, end] = asf[end, 1] = asf[end, end] = 0.0

# Test against exact solution
sol′ = sol[u(t, x, y)]
@test asf ≈ sol′[end, :, :] atol = 0.4

#Plot
using Plots
heatmap(sol′)
savefig("MOL_Linear_Diffusion_2D_Test00.png")