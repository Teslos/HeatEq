# 3D nonlinear  implicit heat diffusion solver with acceleration
using Plots, Printf, LinearAlgebra
using WriteVTK

# enable plotting by default
if !@isdefined do_visu; do_visu = true end

# finite-difference support functions
@views av_xi(A) = 0.5*(A[1:end-1,2:end-1].+A[2:end,2:end-1])
@views av_yi(A) = 0.5*(A[2:end-1,1:end-1].+A[2:end-1,2:end])
@views   inn(A) = A[2:end-1,2:end-1]

# macros to avoid array allocation
macro qUx(ix,iy,iz)  esc(:(-kx*(U[$ix+1,$iy+1,$iz+1]-U[$ix,$iy+1,$iz+1])*_dx )) end
macro qUy(ix,iy,iz)  esc(:(-ky*(U[$ix+1,$iy+1,$iz+1]-U[$ix+1,$iy,$iz+1])*_dy )) end
macro qUz(ix,iy,iz)  esc(:(-kz*(U[$ix+1,$iy+1,$iz+1]-U[$ix+1,$iy+1,$iz])*_dz )) end

macro dtau(ix,iy,iz) esc(:( (1/ (min_dxyz2 /kx /8.1) + 1.0 *_dt)^-1)) end

function compute_update!(source, U2, dUdtau, U, Uold, _dt, damp, ρCp, kx, ky, kz, min_dxyz2, _dx, _dy, _dz)
    Threads.@threads for iz=1:size(dUdtau,3)
    for iy=1:size(dUdtau,2)
        for ix=1:size(dUdtau,1)
            dUdtau[ix,iy,iz] = -ρCp*(U[ix+1,iy+1,iz+1] - Uold[ix+1,iy+1,iz+1])*_dt +
            (-(@qUx(ix+1,iy,iz)-@qUx(ix,iy,iz))*_dx
             -(@qUy(ix,iy+1,iz)-@qUy(ix,iy,iz))*_dy
             -(@qUz(ix,iy,iz+1)-@qUz(ix,iy,iz))*_dz) +
            damp*dUdtau[ix,iy,iz] + source[ix,iy,iz]  # rate of change
            U2[ix+1,iy+1,iz+1] = U[ix+1,iy+1,iz+1] + @dtau(ix,iy,iz)*dUdtau[ix,iy,iz] # update rule, sets the BC as U[1]=U[end]=0
        end
    end
    end
    return
end
function compute_residual!(source, ResU, U, Uold, ρCp, kx, ky, kz, _dt, _dx, _dy, _dz)
    Threads.@threads for iz = 1:size(ResU,3)
    for iy = 1:size(ResU,2)
        for ix = 1:size(ResU,1)
            ResU[ix,iy,iz]   = -ρCp*(U[ix+1,iy+1,iz+1] - Uold[ix+1,iy+1,iz+1])*_dt +
            (-(@qUx(ix+1,iy,iz)-@qUx(ix,iy,iz))*_dx
             -(@qUy(ix,iy+1,iz)-@qUy(ix,iy,iz))*_dy
             -(@qUz(ix,iy,iz+1)-@qUz(ix,iy,iz))*_dz) +
            source[ix,iy,iz]
        end
    end
    end
    return
end

function assign!(Uold, U)
    Threads.@threads for iz=1:size(U,3)
    for iy=1:size(U,2)
        for ix=1:size(U,1)
            Uold[ix,iy,iz] = U[ix,iy,iz]
        end
    end
    end
    return
end

using Printf
@views function laserheating_3D_damp_loop_fun(; do_visu=true, save_fig=false, do_vtk=false)
    # Physics
    lx, ly, lz = 2e-3, 1e-3, 1e-4  # domain size
    ttot   = 0.2           # total simulation time
    dt     = 0.05          # physical time step
    a,b,c = 2e-4,1e-4,5.0e-5       # laser parameters
    r₀ = 1                 # laser beam size
    v  = 10e-3             # laser speed

    ρ = 7860; Cₚ = 624; k = 30.1
    ρCp = ρ*Cₚ
    QLaser  = 5.0;
    # mathematica notebook
    power = 5e+6
    γ = 5e+4

    κ = k/(ρ*Cₚ)
    kx, ky, kz = k, k, k     # conductivity of the material
    # Numerics
    nx, ny, nz = 128, 128, 10     # number of grid points
    tol    = 1e-4         # tolerance
    nout   = 100
    itMax  = 1e5          # max number of iterations
    damp   = 1-19/nx      # damping (this is tuning parameter, dependent on e.g. grid resolution)
    # Derived numerics
    dx, dy, dz = lx/nx, ly/ny, lz/nz # grid size
    xc, yc, zc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny), LinRange(dz/2, lz-dz/2, nz)
    # Array allocation
    dUdtau = zeros(nx-2, ny-2, nz-2) # normal grid, without boundary points
    ResU   = zeros(nx-2, ny-2, nz-2) # normal grid, without boundary points
    source = zeros(nx-2, ny-2, nz-2) # normal grid, without boundary points
    # Initial condition
    U = 20*ones(nx,ny,nz)
    #U   = 3*sqrt(3)*P/(π*sqrt(π)*a*b*c)* [exp(-   (xc[i]-lx/2)^2  -(yc[j]-ly/2)^2 - (zc[k]-lz/2)^2) for i=1:size(U,1), j=1:size(U,2), k=1:size(U,3)]
    S(t) = 6*sqrt(3)*QLaser/(π*sqrt(π)*a*b*c).*[exp(-3*( ((xc[i]-v*t)/a)^2+((yc[j]-ly/2)/b)^2+((zc[k]-lz/2)/c)^2) )
             for i=1:size(U,1),j=1:size(U,2),k=1:size(U,3)]
    @printf("Power: %lf",maximum(S(0.0)))
    # this is source from mathematica notebook
    newsource(t) = [ρ*power*exp(-γ*(xc[i]-v*t)^2 + (yc[j]-ly/2)^2 + (zc[k]-lz)^2) for i=1:size(U,1), j=1:size(U,2), k=1:size(U,3)]
    Uold   = copy(U)
    U2     = copy(U)
    _dx, _dy, _dz, _dt = 1.0/dx, 1.0/dy, 1.0/dz, 1.0/dt
    min_dxyz2 = min(dx,dy,dz)^2
    #dt     = minimum(min(dx, dy)^2 ./inn(U)./4.1)  # time step (obeys ~CFL condition)
    t = 0.0; it = 0; ittot = 0; t_tic = 0.0; niter = 0
    # save files
    pvd = paraview_collection("transient"; append=true)
    # Physical time loop
    while t<ttot
        iter = 0; err = 2*tol
        source = S(t)
        # Picard-type iteration
        while err>tol && iter<itMax
            if (it==1 && iter==0) t_tic = Base.time() end
            compute_update!(source, U2, dUdtau, U, Uold, _dt, damp, ρCp, kx, ky, kz, min_dxyz2, _dx, _dy, _dz)
            U, U2 = U2, U
            if iter % nout == 0
                compute_residual!(source, ResU, U, Uold, ρCp, kx, ky, kz, _dt, _dx, _dy, _dz)
                err = norm(ResU)/length(ResU)
            end

            iter += 1; niter += 1
        end
        t += dt; ittot += iter; it +=1
        assign!(Uold,U)

        if do_vtk
            vtk_grid("fields_$it", xc, yc, zc) do vtk
                vtk["Temperature"] = U
                pvd[t] = vtk
            end
        end
    end
    vtk_save(pvd) # save all the files

    t_toc = Base.time() - t_tic
    A_eff = (2*2+1)/1e9*nx*ny*nz*sizeof(Float64)
    t_it  = t_toc/ittot
    T_eff = A_eff/t_it
    @printf("Time = %1.3f sec, T_eff=%1.2f GB/s (niter=%d)\n",t_toc,round(ttot, sigdigits=2), ittot)

    # Visualize
    if do_visu
        fontsize = 12
        opts = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
                ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", colorbar_title="",
                xlabel="Lx", ylabel="Ly", xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), clims=(0.,maximum(U)))
        display(heatmap(xc, yc, U[:,:,18]; c=:davos, title="implicit laser heating (nt=$it)", opts...))
        if save_fig savefig("laserheating3D_damp4.png") end
    end


    @printf("Max temp: %lf", maximum(U))
    #return xc, yc, U
end

laserheating_3D_damp_loop_fun(; do_visu=false, do_vtk=true);
