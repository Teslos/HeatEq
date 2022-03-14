# This the one dimensional Burgers' equation
# that is nonlinear approx to Navier-Stokes equation.
# the results are compared with the known analytic
# solution.
using OrdinaryDiffEq, ModelingToolkit, DiffEqOperators
using ModelingToolkit: Differential, infimum,supremum

anal_solution(t,x) = (0.1*exp(-A(t,x))+0.5*exp(-B(t,x)) + exp(-C(t,x))) /
                     (exp(-A(t,x))+exp(-B(t,x))+exp(-C(t,x)))
A(t,x) = 0.05/ν*(x-0.5+4.95t)
B(t,x) = 0.25/ν*(x-0.5+0.75t)
C(t,x) = 0.5/ν*(x-0.375)

@parameters t x
@parameters ν
@variables v(..)

Dxx = Differential(x)^2
Dx  = Differential(x)
Dt  = Differential(t)
# Burgers equation
eq = Dt(v(t,x)) ~ -v(t,x) * Dx(v(t,x)) + ν*Dxx(v(t,x))

npoints = 201
xl = 0.
xu = 1.
#ν  = 0.003
dx = (xu-xl)/(npoints-1)
# Time interval
t₀ = 0.0
t₁ = 1.0

# initial and boundary conditions
bcs = [v(t₀,x) ~ anal_solution(t₀,x),
       v(t₀,x) ~ anal_solution(t₀,x),
       v(t,xl) ~ anal_solution(t,xl),
       v(t,xu) ~ anal_solution(t,xu)]


domains = [t ∈ Interval(t₀,t₁),
           x ∈ Interval(xl,xu)]

@named pdesys = PDESystem([eq], bcs, domains, [t,x], [v(t,x)],[ν=>0.003])
discretization = MOLFiniteDifference([x=>dx],t)

prob = discretize(pdesys, discretization)
sol  = solve(prob, Tsit5())

x = (xl:dx:xu)[2:end-1]
t = sol.t
using Plots
anim = @animate for i in 1:length(t)
        p1 = plot(x,sol.u[i][:],label="u, t=$(t[i])";legend=false,xlabel="x",ylabel="u",ylim=[0,1])
end
gif(anim,"plot.gif",fps=20)

using Test
@test anal_solution.(t₁,x) ≈ sol[end] rtol=1e-3
