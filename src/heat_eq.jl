using OrdinaryDiffEq, ModelingToolkit, DiffEqOperators
using ModelingToolkit: Differential, infimum,supremum

@parameters t x y
@variables u(..)

Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)

t_min = 0.0
t_max = 1.0
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 2.0

eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))
anal_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4 * t)
# initial and boundary conditions
bcs = [
        u(t_min, x, y) ~ anal_sol_func(t_min, x, y),
        u(t, x_max, y) ~ anal_sol_func(t, x_max, y),
        u(t, x_min, y) ~ anal_sol_func(t, x_min, y),
        u(t, x, y_max) ~ anal_sol_func(t, x, y_max),
        u(t, x, y_min) ~ anal_sol_func(t, x, y_min),
]

domains = [
        t ∈ IntervalDomain(t_min, t_max),
        x ∈ IntervalDomain(x_min, x_max),
        y ∈ IntervalDomain(y_min, y_max),
]

pdesys = PDESystem([eq], bcs, domains, [t,x,y], [u(t,x,y)])

# Method of the lines
dx = 0.1; dy = 0.2;
order = 2;
discretization = MOLFiniteDifference([x=>dx,y=>dy],t;centered_order = order)

prob = ModelingToolkit.discretize(pdesys, discretization)
sol  = solve(prob, Tsit5())

using Plots
Nx = floor(Int64, (x_max - x_min) / dx) + 1
Ny = floor(Int64, (y_max - y_min) / dy) + 1
sol1 = reshape([sol.u[end][(j-1)*(Nx-2) + i] for j in 1:Ny-2 for i in 1:Nx-2],(Nx-2,Ny-2) )
heatmap(sol1)

r_x = x_min:dx:x_max
r_y = y_min:dy:y_max
asf = reshape([anal_sol_func(t_max, r_x[i], r_y[j])
              for j in 1:Ny for i in 1:Nx],(Nx,Ny) )
heatmap(asf[1:19,1:9])

plot(asf[:,10])
plot!(sol1[:,9])
