# 2D diffusion problem

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq, DomainSets
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

analytic_sol_func(t, x, y) = exp(x + y) * cos(x + y + 4t)
# Analytic solution

# Equation
eq = Dt(u(t, x, y)) ~ Dxx(u(t, x, y)) + Dyy(u(t, x, y))

# Initial and boundary conditions
bcs = [
    u(t_min, x, y) ~ analytic_sol_func(t_min, x, y),
    u(t, x_min, y) ~ analytic_sol_func(t, x_min, y),
    u(t, x_max, y) ~ analytic_sol_func(t, x_max, y),
    u(t, x, y_min) ~ analytic_sol_func(t, x, y_min),
    u(t, x, y_max) ~ analytic_sol_func(t, x, y_max),
]

# Space and time domains
domains = [
        t ∈ Interval(t_min, t_max),
        x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max),
]

# Space and time domains
pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)])

dx = 0.1;
dy = 0.2;


# Test against exact solution
Nx = floor(Int64, (x_max - x_min) / dx) + 1
Ny = floor(Int64, (y_max - y_min) / dy) + 1
@variables u[1:Nx, 1:Ny](t)
r_space_x = x_min:dx:x_max
r_space_y = y_min:dy:y_max
asf = reshape(
    [
        analytic_sol_func(t_max, r_space_x[i], r_space_y[j]) for j = 1:Ny
        for i = 1:Nx
    ],
    (Nx, Ny),
)
# Method of lines discretization
order = 2
discretization =
    MOLFiniteDifference([x => dx, y => dy], t; centered_order = order)
prob = ModelingToolkit.discretize(pdesys, discretization)

# Solution of the ODE system
sol = solve(prob, Tsit5())

# Test against exact solution
sol1 = reshape([sol[u[(i-1)*Ny+j]][end] for i = 1:Nx for j = 1:Ny], (Nx, Ny))
sol1 = zeros((Nx-2,Ny-2))
using Printf
for i = 1:(Nx-2)
    for j = 1:(Ny-2)
        k = (j-1)*(Nx-2) + i
        @printf("k:%d, i:%d, j:%d\n",k,i,j)
        sol1[i,j] = sol.u[end][k]
    end
end


#Plot
using Plots
plot(sol1[:,4])
    #savefig("MOL_Linear_Diffusion_2D_Test00.png")
plot!(asf[:,5])
plot(analytic_sol_func.(2.,0.,r_space_y))
heatmap(sol1)
heatmap(asf[1:19,1:10])
