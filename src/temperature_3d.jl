const USE_GPU = true
using ImplicitGlobalGrid
using ParallelStencil
using WriteVTK
using Printf
using JSON3
using StructTypes

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

mutable struct MyParameters
    lam::Float64
    c0::Float64
    Lf::Float64
    Tm::Float64
    σ::Float64
    a::Float64
    b::Float64
    c::Float64
    v::Float64
    P::Float64
    α::Float64
end

StructTypes.StructType(::Type{MyParameters}) = StructTypes.OrderedStruct()


@parallel_indices (ix,iy,iz) function diffusion3D_step!(T2, T, Q, Ci, lam, dt, _dx, _dy, _dz, dx, dy, dz, x0, y0, z0, a,b,c)
        if (ix>1 && ix<size(T2,1) && iy>1 && iy<size(T2,2) && iz>1 && iz<size(T2,3))
            source = Q*exp(-3*(( (ix*dx-x0)/a)^2+((iy*dy-y0)/b)^2+((iz*dz-z0)/c)^2))
            T2[ix,iy,iz] = T[ix,iy,iz] + dt*(Ci[ix,iy,iz]*(
                            - ((-lam*(T[ix+1,iy,iz] - T[ix,iy,iz])*_dx) - (-lam*(T[ix,iy,iz] - T[ix-1,iy,iz])*_dx))*_dx
                            - ((-lam*(T[ix,iy+1,iz] - T[ix,iy,iz])*_dy) - (-lam*(T[ix,iy,iz] - T[ix,iy-1,iz])*_dy))*_dy
                            - ((-lam*(T[ix,iy,iz+1] - T[ix,iy,iz])*_dz) - (-lam*(T[ix,iy,iz] - T[ix,iy,iz-1])*_dz))*_dz + source)
                            );
        end
    return
end

function diffusion3D(;do_vtk=true)

# read the parameters from the input file
file = open("./input_3d.json", "r")
json_data = read(file, String)
close(file)
# parse the parameters from JSON data
parameters = JSON3.read(json_data, MyParameters)
print(parameters)

# Physics
lam        = 30.1;                                        # Thermal conductivity
c0         = 7860*624;                                    # Heat capacity

lx, ly, lz = 2e-3, 1e-3, 1e-4                             # Length of computational domain in dimension x, y and z
a,  b,  c  = 0.5e-4, 0.5e-4, 5.0e-5                       # laser parameters
v  = 0.01                                                 # laser speed
P  = 200                                                  # power laser
α  = 0.4
σ  = 1e-4                                                 # absorption coefficient
# Numerics
nx, ny, nz = 256, 256, 64;                               # Number of gridpoints in dimensions x, y and z
nt         = 10000;                                      # Number of time steps
me, dims   = init_global_grid(nx, ny, nz);
dx         = lx/(nx_g()-1);                              # Space step in x-dimension
dy         = ly/(ny_g()-1);                              # Space step in y-dimension
dz         = lz/(nz_g()-1);                              # Space step in z-dimension
_dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz;
xc, yc, zc = Array(LinRange(dx/2, lx-dx/2, nx)), Array(LinRange(dy/2, ly-dy/2, ny)), Array(LinRange(dz/2, lz-dz/2, nz))

# Array initializations
T   = @zeros(nx, ny, nz);
T2  = @zeros(nx, ny, nz);
Ci  = @zeros(nx, ny, nz);
Sl  = @zeros(nx, ny, nz);

# Initial conditions
Ci .= 1/c0;                                              # 1/Heat capacity
# Li  = 1/Lf;
T  .= 1.7;
T2 .= T;                                                 # Assign also T2 to get correct boundary conditions.
Q = 6.0*sqrt(3)*α*P/(π*sqrt(π)*a*b*c)                      # Laser flux for Goldak's heat source
# Time loop
dt   = min(dx^2,dy^2,dz^2)/lam/maximum(Ci)/8.1;          # Time step for 3D Heat diffusion
@printf("Choosen timestep: %g\n",dt)
x0 = lx/2; y0 = ly/2; z0 = lz-dz;
for it = 1:nt
    if (it == 11) tic(); end                             # Start measuring time.
    #Sl = CUDA.CuArray(source(it*dt))
    x0 += v*dt



    @hide_communication (16, 2, 2) begin
        @parallel diffusion3D_step!(T2, T,  Q, Ci, lam,  dt, _dx, _dy, _dz, dx,
                                    dy, dz, x0, y0, z0, a, b, c)
        update_halo!(T2);
    end
    T, T2 = T2, T;
end
time_s = toc()

# Performance
A_eff = (2*1+1)*1/1e9*nx*ny*nz*sizeof(Data.Number);      # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
t_it  = time_s/(nt-10);                                  # Execution time per iteration [s]
T_eff = A_eff/t_it;                                      # Effective memory throughput [GB/s]
if (me==0) println("time_s=$time_s T_eff=$T_eff"); end

finalize_global_grid();

if do_vtk
    vtk_grid("fields_3d", xc, yc, zc) do vtk
        vtk["Temperature"] = Array(T)
    end
end
end

diffusion3D()
