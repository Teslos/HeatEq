# 2D nonlinear explicit diffusion solver
using Plots, Printf, LinearAlgebra

# enable plotting by default
if !@isdefined do_visu; do_visu = true end

# finite-difference support functions
@views av_xi(A) = 0.5*(A[1:end-1,2:end-1].+A[2:end,2:end-1])
@views av_yi(A) = 0.5*(A[2:end-1,1:end-1].+A[2:end-1,2:end])
@views   inn(A) = A[2:end-1,2:end-1]

@views function laserheating_2D_expl(; do_visu=true, save_fig=false)
    # Physics
    lx, ly = 10.0, 10.0   # domain size
    npow   = 3            # power-law exponent
    ttot   = 10.0          # total simulation time
    r₀ = 1                # laser beam size
    v  = 0.7              # laser speed
    P  = 1.e4;
    kx, ky = 1.0, 1.0     # conductivity of the material
    # Numerics
    nx, ny = 128, 128     # number of grid points
    # Derived numerics
    dx, dy = lx/nx, ly/ny # grid size
    xc, yc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
    # Array allocation
    qUx    = zeros(nx-1, ny-2) # on staggered grid
    qUy    = zeros(nx-2, ny-1) # on staggered grid
    dUdt   = zeros(nx-2, ny-2) # normal grid, without boundary points
    # Initial condition
    U      = exp.(.-(xc.-lx/2).^2 .-(yc.-ly/2)'.^2)
    S(t)   = 2*P/(π * r₀).*exp.( -2/1^2 .* ((xc.-v*t).^2 .+ (yc.-ly/2)'.^2) )
    dt     = minimum(min(dx, dy)^2 ./inn(U)./4.1)  # time step (obeys ~CFL condition)
    t = 0.0; it = 0
    # Physical time loop
    while t<ttot
        qUx    .= .-kx.*diff(U[:,2:end-1], dims=1)/dx  # flux
        qUy    .= .-ky.*diff(U[2:end-1,:], dims=2)/dy  # flux
        dUdt   .= .-diff(qUx, dims=1)/dx .-diff(qUy, dims=2)/dy .+ S(t)[2:end-1,2:end-1]   # rate of change
        #@printf("length qUx:%i, length S: %i",length(diff(qUx,dims=1)/dx),
        #    length(S(t)[2:end-1,2:end-1]))
        U[2:end-1,2:end-1] .= inn(U) .+ dt.*dUdt      # update rule, sets the BC as H[1]=H[end]=0
        t += dt; it += 1
    end
    @printf("Total time = %1.2f, time steps = %d \n", round(ttot, sigdigits=2), it)
    # Visualize
    if do_visu
        fontsize = 12
        opts = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
                ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", colorbar_title="",
                xlabel="Lx", ylabel="Ly", xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), clims=(0.,maximum(U)))
        display(heatmap(xc, yc, U'; c=:davos, title="explicit diffusion (nt=$it)", opts...))
        if save_fig savefig("laserheating2D_expl.png") end
    end
    @printf("Max temp: %lf", maximum(U))
    return xc, yc, U
end

laserheating_2D_expl(; do_visu=do_visu);
