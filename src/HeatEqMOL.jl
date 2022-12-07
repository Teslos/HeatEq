using MethodOfLines, ModelingToolkit, DomainSets

@parameters t, x
@variables u(..)

Dt = Differential(t)
Dx = Differential(x)
Dxx = Dx^2

α = 1.1

step(x) = x > 5 ? 1.0 : 0.0

@register_symbolic step(x)

eq = Dt(u(t,x)) ~ α * Dxx(u(t,x))

domain = [x ∈ Interval(0.0, 10.0),
        t ∈ Interval(0.0, 1.0) ]

ic_bc = [u(0.0, x) ~ step(x),
    u(t,0.0) ~ 0.0, u(t, 10.0) ~ 1.0]

@named sys = PDESystem(eq, ic_bc, domain, [t,x], [u(t,x)])

dx = 0.1
discretization  = MOLFiniteDifference([x=>dx], t; approx_order = 2)
prob = ModelingToolkit.discretize(sys, discretization)

using OrdinaryDiffEq

sol = solve(prob, Tsit5(), saveat=0.04)

g = get_discrete(sys, discretization)
using Plots
anim = @animate for (i, t_disc) in enumerate(sol[t])
    plot(g[x], sol[u(t,x)][i,:], ylim = (0,1), label = "u", title="t = $t_disc")
end

gif(anim, "heat_rod.gif", fps = 10)


