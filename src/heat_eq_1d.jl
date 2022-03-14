# This is the code for heat equation
#
# This example demonstrate how to combine 'OrdinaryDiffEq' with
# 'DiffEqOperators' to solve a time-dependent PDE.
# We consider the heat equation on the unit interval, with Dirichlet
# boundary conditions.

# ∂ₜu = ∇u
# u(x=0,t) = a
# u(x=1,t) = b
# u(x,t=0) = u₀(x)

# For a=b=0 and u₀(x) = sin(2πx) a solution is given by:
u_anal(x,t) = sin(2π*x)*exp(-t*(2π)^2)
# we want to reproduce it numerically
using DiffEqOperators, OrdinaryDiffEq

№points = 100
h = 1/(№points+1)
points = range(h, step=h, length=№points)
ord_∂ = 2
order_approx = 2

const Δ = CenteredDifference(ord_∂, order_approx, h, №points)
const bc = Dirichlet0BC(Float64)

t₀=0.
t₁=0.03
u₀=u_anal.(points,t₀)

Base.step(u,p,t) = Δ*bc*u
prob = ODEProblem(step,u₀,(t₀,t₁))
sol  = solve(prob, KenCarp4())

using Test
@test u_anal.(points,t₁) ≈ sol[end] rtol=1e-3
