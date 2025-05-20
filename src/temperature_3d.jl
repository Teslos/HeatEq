const USE_GPU = true
using ImplicitGlobalGrid
using ParallelStencil
using WriteVTK
using Printf
using JSON3
using StructTypes
using HDF5
using CSV
using Parquet
using DataFrames
using MeshGrid
using NPZ

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end
function create_new_input(template_path::AbstractString;
    P::Number, 
    v::Number,
    output_path::AbstractString="new_input.json")
# Load the template JSON
data = JSON3.read(read(template_path, String))

# Create a mutable copy of the data
mutable_data = copy(data)

# Swap P and v values (place v in P field, and P in v field)
mutable_data[:P] = P
mutable_data[:v] = v

# Write to new file
open(output_path, "w") do io
JSON3.write(io, mutable_data)
end

return output_path
end
# parameters struct to pass to simulation
mutable struct MyParameters
    lam::Float64
    c0::Float64
    lx::Float64
    ly::Float64
    lz::Float64
    Lf::Float64
    Tm::Float64
    σ::Float64
    a::Float64
    b::Float64
    c::Float64
    v::Float64
    P::Float64
    α::Float64
    nx::Int64
    ny::Int64
    nz::Int64
    nt::Int64
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

function diffusion3D(;do_vtk=true, do_hdf5=true, do_csv=true, do_npz=true)
# parse the parameters from JSON data
parameters = JSON3.read("input_3d.json")
print(parameters)

# Physics
lam        = parameters.lam                               # Thermal conductivity
c0         = parameters.c0                                # Heat capacity

lx, ly, lz = parameters.lx, parameters.ly, parameters.lz  # Length of computational domain in dimension x, y and z
a,  b,  c  = parameters.a, parameters.b, parameters.c     # laser parameters
v  = parameters.v                                         # laser speed
P  = parameters.P                                         # power laser
α  = parameters.α                                         # absorption coefficient
σ  = parameters.σ                                         # absorption coefficient
# Numerics
nx, ny, nz = parameters.nx, parameters.ny, parameters.nz; # Number of gridpoints in dimensions x, y and z
nt         = parameters.nt;                               # Number of time steps
# initialize only first call
me, dims   = init_global_grid(nx, ny, nz);
dx         = lx/(nx_g()-1);                              # Space step in x-dimension
dy         = ly/(ny_g()-1);                              # Space step in y-dimension
dz         = lz/(nz_g()-1);                              # Space step in z-dimension
_dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz;
xc, yc, zc = Array(LinRange(dx/2, lx-dx/2, nx)), Array(LinRange(dy/2, ly-dy/2, ny)), Array(LinRange(dz/2, lz-dz/2, nz))

# Array initializations
T   = CUDA.zeros(nx, ny, nz);
T2  = CUDA.zeros(nx, ny, nz);
Ci  = CUDA.zeros(nx, ny, nz);
Sl  = CUDA.zeros(nx, ny, nz);

# intialize the array of parameters
params = [
    (P=100.0, v=0.5),
    (P=150.0, v=0.6),
    (P=200.0, v=0.7),
    (P=250.0, v=0.8),
    (P=300.0, v=0.9)
]

# run the simulation for each set of parameters
for (i, param) in enumerate(params)
    P = param.P
    v = param.v

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

    if do_vtk
        vtk_grid("fields_3d_$(P)_$(v).vtr", xc, yc, zc) do vtk
            vtk["Temperature"] = Array(T)
            vtk["P"] = P
            vtk["v"] = v
            vtk["LaserX"] = x0
            vtk["LaserY"] = y0
            vtk["LaserZ"] = z0
        end
    end


    if do_npz
        filename = "fields_3d_$(P)_$(v).npz"
        npzwrite(filename, Dict(
            "Temperature" => vec(Array(T)),
            "P"           => P,
            "v"           => v,
            "x"           => vec(xc),
            "y"           => vec(yc),
            "z"           => vec(zc),
            "LaserX"      => x0,
            "LaserY"      => y0,
            "LaserZ"      => z0
        ))
    end

    # write hdf5 file
    if do_hdf5
        h5open("fields_3d_$(P)_$(v).h5", "w") do file
        # Create a group for coordinate
        g = create_group(file, "coordinates")
        dset = create_dataset(g, "x", Float64, (nx,))
        write(dset, xc)
        dset = create_dataset(g, "y", Float64, (ny,))
        write(dset, yc)
        dset = create_dataset(g, "z", Float64, (nz,))
        write(dset, zc)
        # create group for temperature
        g = create_group(file, "temperature")
        # Create a dataset for temperature data
        g["temperature"] = Array{Float64}(T)
        end
    end

    # write csv file (parquet) that is gzip
    if do_csv
        X,Y,Z = meshgrid(yc, xc, zc)
        # Save coordinates and temperature data in a DataFrame in column-major order
        df = DataFrame(
            X = vec(X),
            Y = vec(Y),
            Z = vec(Z),
            P = fill(P, length(X)),
            v = fill(v, length(X)),
            LaserX = fill(x0, length(X)),
            LaserY = fill(y0, length(X)),
            LaserZ = fill(z0, length(X)),
            Temperature = vec(Array(T)),
        )

        # Write the DataFrame to a CSV file
        Parquet.write_parquet("fields_3d_$(P)_$(v).parquet", df, compression_codec="gzip")
    end
        
end
finalize_global_grid();
end

diffusion3D(;do_csv=true)

