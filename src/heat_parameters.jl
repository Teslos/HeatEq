# This is system with multiple dependent variables and parameters as follows
using ModelingToolkit, MethodOfLines, DiffEqOperators,LinearAlgebra,Test,OrdinaryDiffEq, DomainSets
using ModelingToolkit: Differential

@parameters t x
@parameters Dn, Dρ
@variables u(..) v(..)
Dt  = Differential(t)
Dxx = Differential(x)^2
Dx  = Differential(x)

eqs = [Dt(u(t,x)) ~ Dxx(u(t,x)) + u(t,x)*v(t,x),
       Dt(v(t,x)) ~ Dxx(v(t,x)) - u(t,x)*v(t,x)]

bcs = [u(0,x) ~ sin(pi*x/2),
       v(0,x) ~ sin(pi*x/2),
       u(t,0) ~ 0.0, Dx(u(t,1)) ~ 0,
       v(t,0) ~ 0.0, Dx(v(t,1)) ~ 0]

domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(0.0,1.0)
]

@named pdesys = PDESystem(eqs, bcs, domains, [t,x], [u(t,x),v(t,x)],[Dn=>0.5,Dρ=>2])
discretization = MOLFiniteDifference([x=>0.1],t)

prob = discretize(pdesys, discretization)
sol  = solve(prob, Tsit5())

x = (0:0.1:1)[2:end-1]
t = sol.t
using Plots
anim = @animate for i in 1:length(t)
      p1 = plot(x,sol.u[i][1:9],label="u, t=$(t[i])";legend=false,xlabel="x",ylabel="u",ylim=[0,1])
      p2 = plot(x,sol.u[i][10:end],label="v, t=$(t[i])";legend=false,xlabel="x",ylabel="v",ylim=[0,1])
      plot(p1,p2)
end
gif(anim,"plot.gif",fps=20)
