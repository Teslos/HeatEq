using ModelingToolkit,DiffEqOperators,LinearAlgebra,DomainSets
using ModelingToolkit: Differential

@parameters x y z
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dzz = Differential(z)^2

eq = Dxx(u(x, y, z)) + Dyy(u(x, y, z)) + Dzz(u(x, y, z)) ~ 0
dx = 0.1
dy = 0.1
dz = 0.1

bcs = [u(0,y,z) ~ 0.0,
       u(1,y,z) ~ 1.0,
       u(x,0,z) ~ 0.0,
       u(x,1,z) ~ 1.0,
       u(x,y,0) ~ 0.0,
       u(x,y,1) ~ 1.0]

# Space and time domains
domains = [x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,1.0),
           z ∈ Interval(0.0,1.0)]

pdesys = PDESystem([eq],bcs,domains,[x,y,z],[u(x,y,z)])

# Note that we pass in `nothing` for the time variable `t` here since we
# are creating a stationary problem without a dependence on time, only space.
discretization = MOLFiniteDifference([x=>dx,y=>dy,z=>dz], nothing)

prob = discretize(pdesys,discretization)
sol = solve(prob)

using Plots
xs,ys,zs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
u_sol = reshape(sol.u, (length(xs),length(ys),length(zs)))
# plot(xs, ys, u_sol, linetype=:contourf,title = "solution")
