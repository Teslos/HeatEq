using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval
using SpecialFunctions

@parameters t x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)
t_min= 0.
t_max = 2.0
x_min = 0.
x_max = 2.
y_min = 0.
y_max = 2.
q = 10000; κ = 9; V = 0.1; k = 20.0; T0=298; Q=q
# 2D PDE
eq  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y)) # + Q*exp(-2*(x^2+y^2))
# analytical solution of Rosenthal
analytic_sol_func(q,κ,k,V,x,y) = T0+q/(2π*k*1)*exp(-V*x/(2*κ))*besselk(0,(V*sqrt(x^2+y^2)/(2κ)))
# Initial and boundary conditions
bcs = [u(t_min,x,y) ~ 0.0,
       u(t,x_min,y) ~ analytic_sol_func(q,κ,k,V,x_min,y),
       u(t,x_max,y) ~ analytic_sol_func(q,κ,k,V,x_max,y),
       u(t,x,y_min) ~ analytic_sol_func(q,κ,k,V,x,y_min),
       u(t,x,y_max) ~ analytic_sol_func(q,κ,k,V,x,y_max)]

# Space and time domains
domains = [t ∈ Interval(t_min,t_max),
           x ∈ Interval(x_min,x_max),
           y ∈ Interval(y_min,y_max)]

# Neural network
# Neural network
dim = 2 # number of dimensions
chain = Lux.Chain(Dense(dim,16,Lux.σ),Dense(16,16,Lux.σ),Dense(16,1))

# Discretization
dx = 0.05; dy = 0.05
dt = 0.2
discretization = PhysicsInformedNN(chain,GridTraining([dt,dx,dy]))

# Method of lines discretization
discretization = MOLFiniteDifference([x => dx, y => dy], t; approx_order=order)
prob = ModelingToolkit.discretize(pdesys, discretization)

#@named pde_system = PDESystem(eq,bcs,domains,[t,x,y],[u(t, x, y)])
#prob = discretize(pde_system,discretization)

#Optimizer
opt = OptimizationOptimJL.BFGS()

#Callback function
callback = function (p,l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, opt; callback = callback, maxiters=1000)


phi = discretization.phi


using Plots

xs,ys = [infimum(d.domain):dx/10:supremum(d.domain) for d in domains]
u_predict = reshape([first(phi([x,y],res.u)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(q,κ,k,V,x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)
