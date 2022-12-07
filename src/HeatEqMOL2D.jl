using MethodOfLines, ModelingToolkit, DomainSets

@parameters t, x, y
@variables u(..)

Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Dx^2
Dyy = Dy^2

α = 1.1

step(x, y) = x * y > 5 ? 1.0 : 0.0

@register_symbolic step(x, y)

eq = Dt(u(t,x,y)) ~ α * (Dxx(u(t,x,y)) + Dyy(u(t,x,y))) + Dx(u(t,x,y)) - Dy(u(t,x,y))

domain = [x ∈ Interval(0.0, 10.0),
          y ∈ Interval(0.0, 10.0),
          t ∈ Interval(0.0, 5.0) ]

ic_bc = [u(0.0, x, y) ~ step(x, y),
         u(t,0.0,y) ~ u(t,10.0,y),
         u(t,x, 0.0) ~ u(t,x,10.0)]

@named sys = PDESystem(eq, ic_bc, domain, [t,x,y], [u(t,x,y)])

dx = 0.25
dy = 0.25
discretization  = MOLFiniteDifference([x=>dx, y=>dy], t; approx_order = 2)
prob = discretize(sys, discretization)

using OrdinaryDiffEq

sol = solve(prob, Tsit5(), saveat=0.04)

g = get_discrete(sys, discretization)
using Plots
anim = @animate for (i, t_disc) in enumerate(sol[t])
    heatmap(sol[u(t,x,y)][i,:,:], title="t = $t_disc")
end



gif(anim, "heat_rod2D.gif", fps = 10)


