# 2D heat problem with laser heating
# Problem to solve:
# uₜ = Dk*uₓₓ + Dk*u_yy + S

# Packages and inclusions
using ModelingToolkit,DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential
ρ = 7860; Cₚ = 624; k = 30.1
γ = 2.5e+4; P = 1e+2; r₀ = 0.025
S(t,x,y) = 2P/(π*r₀^2)*exp(-2/(r₀^2)*((x-t*0.02)^2 + (y-y_max/2)^2))
delta_funct(x,y,a,b) = exp(-(x/a)^2-(y/b)^2)/(a*b*sqrt(π))
# Variables, parameters, and derivatives
@parameters t x y
@parameters Dk
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dy  = Differential(y)
Dt  = Differential(t)
t_min = 0.0
t_max = 5.0
x_min = 0.0
x_max = 0.1
y_min = 0.0
y_max = 0.2

# Equation
eq = Dt(u(t, x, y)) ~ Dk*Dxx(u(t, x, y)) + Dk*Dyy(u(t, x, y)) + S(t,x,y)
T₀ = 300.0
# Initial and boundary conditions
bcs = [
    #u(t_min, x, y) ~ delta_funct(x-1,y-1,0.01,0.01),  # initial temperature T0 = 0
    u(t_min, x, y) ~ 0.0,
    u(t, x_min, y) ~ T₀,  # left side is set to temp T₀
    u(t, x_max, y) ~ T₀,  # right side is set to temp T₀
    u(t, x, y_min) ~ T₀,  # bottom side is set to T₀
    #Dy(u(t, x, y_max)) ~ 1.0  # top side is set to laser heat flux
    #u(t, x, y_max) ~ 1.0
]

# Space and time domains
domains = [
        t ∈ Interval(t_min, t_max),
        x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max),
]

# Space and time domains
pdesys = PDESystem([eq], bcs, domains, [t, x, y], [u(t, x, y)], [Dk=>k/(ρ*Cₚ)])

dx = (x_max-x_min)/50;
dy = (y_max-y_min)/50;

# Method of lines discretization
order = 2
discretization =
    MOLFiniteDifference([x => dx, y => dy], t; centered_order = order)
prob = ModelingToolkit.discretize(pdesys, discretization)

# Solution of the ODE system
sol = solve(prob, KenCarp4())
# Test against exact solution
Nx = floor(Int64, (x_max - x_min) / dx) + 1
Ny = floor(Int64, (y_max - y_min) / dy) + 1
# Test against exact solution
sol1 = zeros((Nx-2,Ny-2))
using Printf
for i = 1:(Ny-2)
    for j = 1:(Nx-2)
        k = (j-1)*(Ny-2) + i
        sol1[j,i] = sol.u[end][k]
    end
end


#Plot
using Plots

#savefig("MOL_Linear_Diffusion_2D_Test00.png")
_,xs,_ = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
times = sol.t
anim = @animate for i = 1 : length(times)
    global u_sol
    u_sol = reshape(sol.u[i], (length(xs)-2,length(xs)-1))
    heatmap(u_sol)
end

gif(anim,"laser_heating.gif",fps=20)
plot(u_sol[:,21])
minimum(u_sol[:,21])
