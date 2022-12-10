using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum

@parameters t, x
@variables u1(..), u2(..), u3(..)
Dt = Differential(t)
Dtt = Differential(t)^2
Dx = Differential(x)
Dxx = Differential(x)^2

eqs = [Dtt(u1(t,x)) ~ Dxx(u1(t,x)) + u3(t,x)*sin(pi*x),
       Dtt(u2(t,x)) ~ Dxx(u2(t,x)) + u3(t,x)*cos(pi*x),
       0. ~ u1(t,x)*sin(pi*x) + u2(t,x)*cos(pi*x) - exp(-t)]

bcs = [u1(0,x) ~ sin(pi*x),
       u2(0,x) ~ cos(pi*x),
       Dt(u1(0,x)) ~ -sin(pi*x),
       Dt(u2(0,x)) ~ -cos(pi*x),
       u1(t,0) ~ 0.,
       u2(t,0) ~ exp(-t),
       u1(t,1) ~ 0.,
       u2(t,1) ~ -exp(-t)]


# Space and time domains
domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(0.0,1.0)]

# Neural network
input_ = length(domains)
n = 15
chain =[Lux.Chain(Dense(input_,n,Lux.σ),Dense(n,n,Lux.σ),Dense(n,1)) for _ in 1:3]

strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, strategy)

@named pdesystem = PDESystem(eqs,bcs,domains,[t,x],[u1(t, x),u2(t, x),u3(t, x)])
prob = discretize(pdesystem,discretization)
sym_prob = symbolic_discretize(pdesystem,discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

callback = function (p, l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    return false
end

res = Optimization.solve(prob,BFGS(); callback = callback, maxiters=5000)

phi = discretization.phi

using Plots
gr()
xs,ts = [infimum(d.domain):dx:supremum(d.domain) for (d,dx) in zip(domains,[dx,dt])]
minimizers_ = [res.u.depvar[sym_prob.depvars[i]] for i in 1:3]
u_predict = [[phi[i]([t,x],minimizers_[i])[1] for t in ts for x in xs] for i in 1:3]

for i in 1:3
    p2 = plot(ts, xs, u_predict[i],linetype=:contourf,title = "predict");
    plot(p2)
    savefig("sol_u$i")
end
