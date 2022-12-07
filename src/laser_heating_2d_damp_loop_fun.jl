# 2D nonlinear  implicit diffusion solver with acceleration
using Plots, Printf, LinearAlgebra


# enable plotting by default
if !@isdefined do_visu; do_visu = true end

# finite-difference support functions
@views av_xi(A) = 0.5*(A[1:end-1,2:end-1].+A[2:end,2:end-1])
@views av_yi(A) = 0.5*(A[2:end-1,1:end-1].+A[2:end-1,2:end])
@views   inn(A) = A[2:end-1,2:end-1]

# macros to avoid array allocation
macro qUx(ix,iy)  esc(:(-kx*(U[$ix+1,$iy+1]-U[$ix,$iy+1])*_dx )) end
macro qUy(ix,iy)  esc(:(-ky*(U[$ix+1,$iy+1]-U[$ix+1,$iy])*_dy )) end
macro dtau(ix,iy) esc(:( (1/ (min_dxy2 /kx /4.1) + 1.0 *_dt)^-1)) end

function compute_update!(source, U2, dUdtau, U, Uold, _dt, damp, kx, ky, min_dxy2, _dx, _dy)
    Threads.@threads for iy=1:size(dUdtau,2)
        for ix=1:size(dUdtau,1)
            dUdtau[ix,iy] = -(U[ix+1,iy+1] - Uold[ix+1,iy+1])*_dt +
            (-(@qUx(ix+1,iy)-@qUx(ix,iy))*_dx -(@qUy(ix,iy+1)-@qUy(ix,iy))*_dy) +
            damp*dUdtau[ix,iy] + source[ix,iy]  # rate of change
            U2[ix+1,iy+1] = U[ix+1,iy+1] + @dtau(ix,iy).*dUdtau[ix,iy] # update rule, sets the BC as U[1]=U[end]=0
        end
    end
    return
end
function compute_residual!(source, ResU, U, Uold, kx, ky, _dt, _dx, _dy)
    Threads.@threads for iy = 1:size(ResU,2)
        for ix = 1:size(ResU,1)
            ResU[ix,iy]   = -(U[ix+1,iy+1] - Uold[ix+1,iy+1])*_dt +
            (-(@qUx(ix+1,iy)-@qUx(ix,iy))*_dx
            -(@qUy(ix,iy+1)-@qUy(ix,iy))*_dy) +
            source[ix,iy]
        end
    end
    return
end

function assign!(Uold, U)
    Threads.@threads for iy=1:size(U,2)
    # for iy=1:size(H,2)
        for ix=1:size(U,1)
            Uold[ix,iy] = U[ix,iy]
        end
    end
    return
end

@views function laserheating_2D_damp_loop_fun(; do_visu=true, save_fig=false)
    # Physics
    lx, ly = 10.0, 10.0   # domain size
    npow   = 3            # power-law exponent
    ttot   = 12.0         # total simulation time
    dt     = 0.5          # physical time step
    r₀ = 1                # laser beam size
    v  = 0.7              # laser speed
    P  = 1.e4;
    kx, ky = 2.0, 2 .0     # conductivity of the material
    # Numerics
    nx, ny = 256, 256     # number of grid points
    tol    = 1e-6         # tolerance
    nout   = 100
    itMax  = 1e5          # max number of iterations
    damp   = 1-19/nx      # damping (this is tuning parameter, dependent on e.g. grid resolution)
    # Derived numerics
    dx, dy = lx/nx, ly/ny # grid size
    xc, yc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
    # Array allocation
    dUdtau = zeros(nx-2, ny-2) # normal grid, without boundary points
    ResU   = zeros(nx-2, ny-2) # normal grid, without boundary points
    source = zeros(nx-2, ny-2)
    # Initial condition
    U      = exp.(.-(xc.-lx/2).^2 .-(yc.-ly/2)'.^2)
    S(t)   = 2*P/(π * r₀^2).*exp.( -2/1^2 .* ((xc.-v*t).^2 .+ (yc.-ly/2)'.^2) )
    Uold   = copy(U)
    U2     = copy(U)
    _dx, _dy, _dt = 1.0/dx, 1.0/dy, 1.0/dt
    min_dxy2 = min(dx,dy)^2
    #dt     = minimum(min(dx, dy)^2 ./inn(U)./4.1)  # time step (obeys ~CFL condition)
    t = 0.0; it = 0; ittot = 0; t_tic = 0.0; niter = 0
    # Physical time loop
    anim = @animate while t<ttot
        iter = 0; err = 2*tol
        source = S(t)
        # Picard-type iteration
        while err>tol && iter<itMax
            if (it==1 && iter==0) t_tic = Base.time() end
            compute_update!(source, U2, dUdtau, U, Uold, _dt, damp, kx, ky, min_dxy2, _dx, _dy)
            U, U2 = U2, U
            if iter % nout == 0
                compute_residual!(source, ResU, U, Uold, kx, ky, _dt, _dx, _dy)
                err = norm(ResU)/length(ResU)
            end

            iter += 1; niter += 1
        end
        t += dt; ittot += iter; it +=1
        assign!(Uold,U)
        # Visualize
        if do_visu
            fontsize = 12
            #opts = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
            #        ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", colorbar_title="",
            #        xlabel="Lx", ylabel="Ly", xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), clims=(0.,maximum(U)))
            opts = (aspect_ratio=1, ticks=nothing, framestyle=:box,xlabel="Lx", ylabel="Ly", xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), clims=(0.,maximum(U)))
            #display(heatmap(xc, yc, U'; c=:davos, title="implicit diffusion (nt=$it)", opts...))
            heatmap(xc, yc, U'; c=:davos, title="implicit diffusion (nt=$it)", opts...)
            #if save_fig savefig("laserheating2D_damp2.png") end
        end
    end
    t_toc = Base.time() - t_tic
    A_eff = (2*2+1)/1e9*nx*ny*sizeof(Float64)
    t_it  = t_toc/ittot
    T_eff = A_eff/t_it
    @printf("Time = %1.3f sec, T_eff=%1.2f GB/s (niter=%d)\n",t_toc,round(ttot, sigdigits=2), ittot)


    @printf("Max temp: %lf", maximum(U))
    gif(anim, "anim_laser_heating_2d.gif",fps=12)
    return xc, yc, U
end

laserheating_2D_damp_loop_fun(; do_visu=do_visu);
