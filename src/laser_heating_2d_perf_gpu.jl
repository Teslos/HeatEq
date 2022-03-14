# 2D nonlinear diffusion GPU implicit solver
# with the acceleration
using CUDA, Plots, Printf, LinearAlgebra
# enable plotting by default
if !@isdefined do_visu; do_visu = true end

# macros to avoid array allocation
macro qUx(ix,iy)  esc(:(-kx*(U[$ix+1,$iy+1]-U[$ix,$iy+1])*_dx )) end
macro qUy(ix,iy)  esc(:(-ky*(U[$ix+1,$iy+1]-U[$ix+1,$iy])*_dy )) end
macro dtau(ix,iy) esc(:( (1.0 / (min_dxy2 /kx /4.1) + 1.0 *_dt)^-1)) end

function compute_update!(source, U2, dUdtau, U, Uold, _dt, damp, kx, ky, min_dxy2, _dx, _dy)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if (ix <= size(dUdtau,1) && iy <= size(dUdtau,2))
        dUdtau[ix,iy] = -(U[ix+1,iy+1] - Uold[ix+1,iy+1])*_dt +
        (-(@qUx(ix+1,iy)-@qUx(ix,iy))*_dx -(@qUy(ix,iy+1)-@qUy(ix,iy))*_dy) +
        damp*dUdtau[ix,iy] + source[ix,iy]  # rate of change
        U2[ix+1,iy+1] = U[ix+1,iy+1] + @dtau(ix,iy) * dUdtau[ix,iy]
    end
    return
end

function compute_residual!(source, ResU, U, Uold, kx, ky, _dt, _dx, _dy)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if (ix <= size(ResU,1) && iy <= size(ResU,2))
        ResU[ix,iy]   = -(U[ix+1,iy+1] - Uold[ix+1,iy+1])*_dt +
        (-(@qUx(ix+1,iy)-@qUx(ix,iy))*_dx
        -(@qUy(ix,iy+1)-@qUy(ix,iy))*_dy) +
        source[ix,iy]
    end
    return
end

function assign!(Uold, U)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (ix<=size(U,1) && iy<=size(U,2)) Uold[ix,iy] = U[ix,iy] end
    return
end

function laser!(xc, yc, S, Q, v, t, ly)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (ix<=size(S,1) && iy<=size(S,2))
        S[ix,iy] = Q*exp( -2/1^2 * ((xc[ix]-v*t)^2 + (yc[ix]-ly/2)^2) )
    end
    return
end

@views function laserheating_2D_damp_gpu(; do_visu=true, save_fig=false)
    # Physics
    lx, ly = 10.0, 10.0   # domain size
    ttot   = 4.0          # total simulation time
    dt = 0.2              # physical time step
    r₀ = 1                # laser beam size
    v  = 0.7              # laser speed
    P  = 1.e4;
    kx, ky = 1.0, 1.0     # conductivity of the material
    # Numerics
    BLOCKX = 32
    BLOCKY = 8
    GRIDX  = 16*16
    GRIDY  = 64*16

    nx, ny = BLOCKX*GRIDX, BLOCKY*GRIDY    # number of grid points
    tol    = 1e-4         # tolerance
    itMax  = 1e5          # max number of iterations
    damp   = 1-19/nx      # damping (this is tuning parameter, dependent on e.g. grid resolution)
    # Derived numerics
    dx, dy = lx/nx, ly/ny # grid size
    xc, yc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
    # Array allocation
    dUdtau = CUDA.zeros(Float32, nx-2, ny-2) # normal grid, without boundary points
    ResU   = CUDA.zeros(Float32, nx-2, ny-2) # normal grid, without boundary points
    S      = CUDA.zeros(Float32, nx-2, ny-2)
    # Initial condition
    U      = CuArray(exp.(.-(xc.-lx/2).^2 .-(yc.-ly/2)'.^2))
    # S(t)   = CuArray(2*P/(π * r₀).*exp.( -2/1^2 .* ((xc.-v*t).^2 .+ (yc.-ly/2)'.^2) ))
    #S      = CuArray(2*P/(π * r₀).*exp.( -2/1^2 .* ((xc.-lx/2).^2 .+ (yc.-ly/2)'.^2) ))
    #S       = CUDA.zeros(Float32, nx-2, ny-2)
    Uold   = copy(U)
    U2     = copy(U)
    cuthreads = (BLOCKX,BLOCKY,1)
    cublocks  = (GRIDX, GRIDY, 1)
    _dx, _dy, _dt = 1.0/dx, 1.0/dy, 1.0/dt
    min_dxy2 = min(dx, dy)^2
    t = 0.0; it = 0; ittot = 0; t_tic = 0.0; niter = 0; nout = 10
    Q = 2*P/(π * r₀)
    # Physical time loop
    while t<ttot
        iter = 0; err = 2*tol
        # calculate the source term
        @cuda blocks = cublocks threads = cuthreads laser!(xc, yc, S, Q, v, t, ly)
        synchronize()
        # Picard-type iteration
        while err>tol && iter<itMax
            if (it==1 && iter==0) t_tic = Base.time() end
            @cuda blocks = cublocks threads=cuthreads compute_update!(
                S,
                U2,
                dUdtau,
                U,
                Uold,
                _dt,
                damp,
                kx,
                ky,
                min_dxy2,
                _dx,
                _dy)
            synchronize()
            U, U2 = U2, U
            if iter % nout == 0
                @cuda blocks = cublocks threads=cuthreads compute_residual!(
                S, ResU, U, Uold, kx,ky, _dt, _dx, _dy)
                synchronize()
                err = norm(ResU)/length(ResU)
            end
            iter += 1; niter +=1
        end
        t += dt; ittot += niter; it +=1
        @cuda blocks = cublocks threads=cuthreads assign!(Uold,U)
        synchronize()
    end

    t_toc = Base.time() - t_tic
    A_eff = (2*3+1)/1e9*nx*ny*sizeof(Float64)
    t_it  = t_toc/ittot
    T_eff = A_eff/t_it
    @printf("Time = %1.3f sec, T_eff=%1.2f GB/s (niter=%d)\n",t_toc,round(ttot, sigdigits=2), ittot)

    # Visualize
    if do_visu
        fontsize = 12
        opts = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
                ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", colorbar_title="",
                xlabel="Lx", ylabel="Ly", xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), clims=(0.,maximum(U)))
        display(heatmap(xc, yc, Array(U)'; c=:davos, title="implicit diffusion (nt=$it)", opts...))
        if save_fig savefig("laserheating2D_damp_gpu04.png") end
    end
    @printf("Max temp: %lf", maximum(U))
    #return U
end

laserheating_2D_damp_gpu(; do_visu=do_visu)
