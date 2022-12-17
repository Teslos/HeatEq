using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum
using MethodOfLines
@parameters t, x
@variables u1(..) u2(..)
Dt = Differential(t)
Dx = Differential(x)

# Damping value
δ = 4.02; γ = 10.0; ω = 90.8; α = 10.3; β = 2.5;
eqs = [Dt(u1(t,x)) ~ u2(t,x),
      Dt(u2(t,x)) ~ -δ * Dx(u2(t,x)) - α * u1(t,x) - β * u1(t,x)^3  - γ * cos(ω * t)]

# Initial and boundary conditions
bcs = [u1(0,x) ~ 1.0,
       u2(0,x) ~ -1.0]

# Space and time domains
domains = [t ∈ Interval(0.0,20.0),
           x ∈ Interval(0.0,10.0)]
 # Neural network
input_ = length(domains)
n = 15
chain =[Lux.Chain(Dense(input_,n,Lux.σ),Dense(n,n,Lux.σ),Dense(n,1)) for _ in 1:3]
strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, strategy)
@named pdesystem = PDESystem(eqs,bcs,domains,[t,x],[u1(t, x),u2(t,x)])
# Method of lines discretization
order = 2
dx = 0.1
# discretization = MOLFiniteDifference([x => dx], t; approx_order=order)
prob = discretize(pdesystem, discretization)
#prob = discretize(pdesystem, discretization)
# Solution of the ODE system
sol = solve(prob, Tsit5())

#prob = discretize(pdesystem,discretization)
