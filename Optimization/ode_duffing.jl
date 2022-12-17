using OrdinaryDiffEq
using Plots
using LinearAlgebra

function duffing(du,u,p,t)
    du[1] = du[2]
    du[2] = p[1]* u[1]- p[2] * u[1]^3 - p[3] * u[2] + p[4]*cos(p[5] * t)
end

γ = 0.1
d = 10.0
ω = 90.8
α = 10.3
β = 2.5
u0 = [1.0; -1.0]
tspan = (0.0, 50.0)
Δt = 1e-2
T = tspan[1]:Δt:tspan[end]
θ = [α,β,γ,d,ω]
prob = ODEProblem(duffing, u0, tspan, θ)
sol = solve(prob, Tsit5(), adaptive=false, dt=Δt)
integrator = init(prob, Tsit5(), adaptive=false, dt=Δt, save_everystep = false)

plot(sol, vars=(1))

using Distributions

θ = [Uniform(0.5,1.5),Beta(5,1),Normal(3,0.5),Gamma(5,2), Normal(3,0.5)]

_θ = rand.(θ)
prob1 = ODEProblem(duffing,u0,tspan,_θ)
sol = solve(prob1,Tsit5())
plot(sol)
# phase space solution 
plot([sol[i][1] for i=1:length(sol.t)], [sol.u[i][2] for i = 1:length(sol.t)])
prob_func = function(prob,i,repeat)
    remake(prob,p=rand.(θ))
end

ensemble_prob = EnsembleProblem(ODEProblem(duffing,u0,tspan,θ),
    prob_func=prob_func)
sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(),trajectories=1000)
using DiffEqBase.EnsembleAnalysis
plot(EnsembleSummary(sol))
