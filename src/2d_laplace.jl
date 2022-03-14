# This is the Laplace equation using the MOL discretization:
# u_xx + u_yy = 0

using ModelingToolkit,DiffEqOperators,LinearAlgebra,DomainSets
using ModelingToolkit: Differential
using DifferentialEquations

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

eq = Dxx(u(x, y)) + Dyy(u(x, y))~ 0
dx = 0.1
dy = 0.1

bcs = [u(0,y) ~ 0.0,
       u(1,y) ~ y,
       u(x,0) ~ 0.0,
       u(x,2) ~ x]

# Space and time domains
domains = [x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,2.0)]

pdesys = PDESystem([eq],bcs,domains,[x,y],[u(x,y)])

# Note that we pass in `nothing` for the time variable `t` here since we
# are creating a stationary problem without a dependence on time, only space.
discretization = MOLFiniteDifference([x=>dx,y=>dy], nothing)

prob = discretize(pdesys,discretization)
sol = solve(prob)

using Plots
xs,ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
u_sol = reshape(sol.u, (length(xs),length(ys)))
plot(xs, ys, u_sol, linetype=:contourf,title = "solution")
