# The PDE is the one dimensional (1D) diffusion equation
# uₜ = D uₓₓ
# with initial conditions
# u(x,t=0) = δ(x)
using OrdinaryDiffEq, ModelingToolkit, DiffEqOperators
using ModelingToolkit: Differential, infimum,supremum

anal_sol_funct(τ,x) = 1.0/(2*√(π*Dc*τ)) * exp(-x^2/(4Dc*τ))
@parameters τ x y
@parameters Dc
@variables u(..)

Dxx = Differential(x)^2
Dτ = Differential(τ)

delta_funct(x,a) = exp(-(x/a)^2)/(a*sqrt(π)) # approx. to delta function
npoints = 101
xl = -10.0
xu = 10.0
dx = (xu - xl)/(npoints - 1)
eq = Dτ(u(τ,x)) ~ Dc*Dxx(u(τ,x))
# time interval
τ₀ = 0.0
τ₁ = 2.0
Dc = 2.0

# initial and boundary conditions
bcs = [u(τ₀,x) ~ delta_funct(x,0.1),
       u(τ,xl) ~ 0.0,
       u(τ,xu) ~ 0.0 ]
domains = [
        τ ∈ Interval(τ₀, τ₁),
        x ∈ Interval(xl, xu)
]

pdesys = PDESystem([eq], bcs, domains, [τ,x], [u(τ,x)],[Dc=>2.0,])

discretization = MOLFiniteDifference([x=>0.1],τ)
prob = discretize(pdesys, discretization)
sol  = solve(prob, Tsit5())
x = (-10:0.1:10)[2:end-1]
t = sol.t
using Plots
anim = @animate for i in 1:length(t)
        p1 = plot(x,sol.u[i][1:199],label="u, t=$(t[i])";legend=false,xlabel="x",ylabel="u",ylim=[0,5])
end
gif(anim,"plot.gif",fps=20)

using Test
@test anal_sol_funct.(τ₁,x) ≈ sol[end] rtol=1e-3
