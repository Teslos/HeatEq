using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum

@parameters t, x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)

f(x) = (1/(2*√π))*exp((-1/2)*(x-3)^2)
  #2D PDE
eq  = Dt(u(t,x)) + u(t,x)*Dx(u(t,x)) ~ 0

# Initial and boundary conditions
bcs = [u(0,x) ~ f(x),
       u(0,x) - f(x) ~ 0]

  # Space and time domains
domains = [t ∈ Interval(0.0,20.0),
           x ∈ Interval(0.0,10.0)]

# Neural network
input_ = length(domains)
n = 15
chain =[Lux.Chain(Dense(input_,n,Lux.σ),Dense(n,n,Lux.σ),Dense(n,1)) for _ in 1:3]


  # Discretization
strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, strategy)
#discretization = PhysicsInformedNN(chain,GridTraining(dx))

@named pdesystem = PDESystem([eq],bcs,domains,[t,x],[u(t, x)])
prob = discretize(pdesystem,discretization)

cb = function (p,l)
      println("Current loss is: $l")
      return false
end

opt = OptimizationOptimJL.BFGS()
res = Optimization.solve(prob,opt; cb = cb, maxiters=5000)
phi = discretization.phi

using Plots

xs,ts = [infimum(d.domain):dx:supremum(d.domain) for (d,dx) in zip(domains,[dx,dt])]
u_predict = reshape([Array(phi([x, t], res.minimizer))[1] for x in xs for t in ts], length(xs), length(ts))

plot(xs,ts,u_predict',st=:surface, title="Burguer equation, PINN", xlabel="X", ylabel="Time", zlabel="U")
