using GalacticOptim, Optim
rosenbrock(x,p) = (p[1]-x[1])^2+p[2]*(x[2]-x[1]^2)^2
x0 = zeros(2)

p = [1.0,100.0]
prob = OptimizationProblem(rosenbrock,x0,p)
sol = solve(prob,NelderMead())

using BlackBoxOptim
prob = OptimizationProblem(rosenbrock,x0,p,lb=[-1.0,-1.0],ub=[1.0,1.0])
sol = solve(prob,BBO_adaptive_de_rand_1_bin_radiuslimited())

f = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(f, x0, p)
sol = solve(prob,BFGS())
