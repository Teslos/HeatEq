const USE_GPU = true
using ImplicitGlobalGrid
using ParallelStencil
using WriteVTK
using Printf

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

# find max depth of the melt pool
function find_max_depth(T, Tm, σ, dx,dy, dz, max_depth, x0, y0, z0)

    # find all cells with temperature below Tm+σ
    solid_cells = findall(x -> x < Tm+σ, T[:,:,:])

    # find the maximum depth of the melt pool
    im = maximum([(Tuple(e))[3] for e in solid_cells])
    max_depth = dz*im
    print("Max. depth: ", max_depth)
    return
end

@parallel_indices (ix,iy,iz) function updatecp!(Ci,c0,T,Lf,σ,Tm)
        if (ix>1 && ix<size(Ci,1) && iy>1 && iy<size(Ci,2) && iz>1 && iz<size(Ci,3))
            c = (c0+Lf*exp(-(T[ix,iy,iz]-Tm)^2/(2*σ^2)))
            Ci[ix,iy,iz] = 1/c
        end
        return
end

@parallel_indices (ix,iy,iz) function diffusion3D_step!(T2, T, Q, Lam, σ, c0, Lf, Tm, dt, _dx, _dy, _dz, dx, dy, dz, x0, y0, z0, a,b,c)
        if (ix>1 && ix<size(T2,1) && iy>1 && iy<size(T2,2) && iz>1 && iz<size(T2,3))
            source = Q*exp(-3*(( (ix*dx-x0)/a)^2+((iy*dy-y0)/b)^2+((iz*dz-z0)/c)^2))
            ci     = c0+Lf/(σ*sqrt(2π))*exp(-(T[ix,iy,iz]-Tm)^2/(2*σ^2))
            T2[ix,iy,iz] = T[ix,iy,iz] + dt*(1/ci * (
                            - ((-Lam[ix,iy,iz]*(T[ix+1,iy,iz] - T[ix,iy,iz])*_dx) - (-Lam[ix,iy,iz]*(T[ix,iy,iz] - T[ix-1,iy,iz])*_dx))*_dx
                            - ((-Lam[ix,iy,iz]*(T[ix,iy+1,iz] - T[ix,iy,iz])*_dy) - (-Lam[ix,iy,iz]*(T[ix,iy,iz] - T[ix,iy-1,iz])*_dy))*_dy
                            - ((-Lam[ix,iy,iz]*(T[ix,iy,iz+1] - T[ix,iy,iz])*_dz) - (-Lam[ix,iy,iz]*(T[ix,iy,iz] - T[ix,iy,iz-1])*_dz))*_dz + source)
                            );
        end
    return
end

function diffusion3D(;do_vtk=true)
# Physics
lam        = 30.1;                                        # Thermal conductivity
c0         = 7860*624;                                    # Heat capacity
lx, ly, lz = 2e-3, 1e-3, 1e-4                             # Length of computational domain in dimension x, y and z
Lf         = 292e+4*7860                                  # Latent heat
Tm         = 1933
σ          = 0.001
max_depth  = 0
a,  b,  c  = 0.5e-4, 0.5e-4, 5.0e-5                       # laser parameters
v  = 1.5                                                  # laser speed
P  = 200                                                  # power laser
α  = 0.4                                                  # absorption coefficient
σ  = 5
time_stedy_state = 0.025                                    # Time for steady state
# Numerics
nx, ny, nz = 256, 256, 64;                               # Number of gridpoints in dimensions x, y and z
nt         = 1000;                                       # Number of time steps

if ImplicitGlobalGrid.grid_is_initialized()
    me, dims   = init_global_grid(nx, ny, nz, init_MPI=false)
else
    me, dims   = init_global_grid(nx, ny, nz)
end

dx         = lx/(nx_g()-1);                              # Space step in x-dimension
dy         = ly/(ny_g()-1);                              # Space step in y-dimension
dz         = lz/(nz_g()-1);                              # Space step in z-dimension
_dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz;
xc, yc, zc = Array(LinRange(dx/2, lx-dx/2, nx)), Array(LinRange(dy/2, ly-dy/2, ny)), Array(LinRange(dz/2, lz-dz/2, nz))

# Array initializations
T   = @zeros(nx, ny, nz)
T2  = @zeros(nx, ny, nz)
Lam = @zeros(nx, ny, nz)
Ci  = @zeros(nx, ny, nz)

# Initial conditions
Lam .= Data.Array([if (iy > ny/2) lam else lam/10 end for ix=1:size(Lam,1), iy=1:size(Lam,2),iz=1:size(Lam,3)])                                              # conductivity

T   .= 1.7;
T2  .= T;                                                 # Assign also T2 to get correct boundary conditions.
Q = 6.0*sqrt(3)*α*P/(π*sqrt(π)*a*b*c)                     # Laser flux for Goldak's heat source
# Time loop
dt   = min(dx^2,dy^2,dz^2)/lam/maximum(1/c0)/8.1;          # Time step for 3D Heat diffusion
@printf("Choosen timestep: %g\n",dt)
y0 = ly/2; z0 = lz-dz;
ns = Int(ceil(time_stedy_state/dt))                     # Number of time steps for steady state
for it = 1:nt
    if (it == 11) tic(); end                             # Start measuring time.
    #Sl = CUDA.CuArray(source(it*dt))
    x0 = v*it*dt

    @hide_communication (16, 2, 2) begin
        @parallel diffusion3D_step!(T2, T,  Q, Lam, σ, c0, Lf, Tm, dt, _dx, _dy, _dz, dx,
                                    dy, dz, x0, y0, z0, a, b, c)
        #@parallel find_max_depth(T, Tm, σ, dz, max_depth)
        update_halo!(T2);
    end
    T, T2 = T2, T;
end
time_s = toc()
x0 = v*nt*dt
max_depth = lz-dz
print("Max. depth at start: ", max_depth)
find_max_depth(T, Tm, σ, dx, dy, dz, max_depth, x0, y0, z0)
println("Max. depth: ", max_depth)

# Performance
A_eff = (2*1+1)*1/1e9*nx*ny*nz*sizeof(Data.Number);      # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
t_it  = time_s/(nt-10);                                  # Execution time per iteration [s]
T_eff = A_eff/t_it;                                      # Effective memory throughput [GB/s]
if (me==0) println("time_s=$time_s T_eff=$T_eff"); end

finalize_global_grid();


if do_vtk
    vtk_grid("fields", xc, yc, zc) do vtk
        vtk["Temperature"] = Array(T)
    end
end
end

diffusion3D()
