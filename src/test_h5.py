import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_hdf5_and_create_vtk(hdf5_filename, vtk_filename=None):
    """
    Read HDF5 file and create VTK file for ParaView visualization
    """
    # Read HDF5 file
    with h5py.File(hdf5_filename, 'r') as f:
        # Read coordinates
        x = f['coordinates/x'][:]
        y = f['coordinates/y'][:]
        z = f['coordinates/z'][:]
        
        # Read temperature data
        T = f['temperature/temperature'][:]
        
        # Read parameters
        params = {}
        for key in f['parameters'].attrs.keys():
            params[key] = f['parameters'].attrs[key]
    
    # Create meshgrid for structured grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # If VTK filename not provided, create one
    if vtk_filename is None:
        vtk_filename = hdf5_filename.replace('.h5', '.vtk').replace('.hdf5', '.vtk')
    
    # Write VTK file for ParaView
    write_vtk_structured_grid(vtk_filename, X, Y, Z, T, params)
    
    return X, Y, Z, T, params

def write_vtk_structured_grid(filename, X, Y, Z, T, params):
    """
    Write VTK structured grid file for ParaView
    """
    nx, ny, nz = X.shape
    
    with open(filename, 'w') as f:
        # VTK header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Temperature field from HDF5\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"POINTS {nx*ny*nz} float\n")
        
        # Write coordinates
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{X[i,j,k]} {Y[i,j,k]} {Z[i,j,k]}\n")
        
        # Write temperature data
        f.write(f"\nPOINT_DATA {nx*ny*nz}\n")
        f.write("SCALARS Temperature float 1\n")
        f.write("LOOKUP_TABLE default\n")
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{T[i,j,k]}\n")
        
        # Add parameters as field data
        f.write(f"\nFIELD FieldData {len(params)}\n")
        for key, value in params.items():
            #f.write(f"    {key} 1 1 float\n {value} \n")
            pass

# Example usage (create a sample HDF5 file for demonstration)
def create_sample_hdf5():
    """Create a sample HDF5 file with the same structure"""
    filename = "sample_data.h5"
    
    # Sample data
    nx, ny, nz = 10, 8, 6
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 0.8, ny)
    z = np.linspace(0, 0.6, nz)
    
    # Create sample temperature field
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    T = 300 + 100 * np.exp(-((X-0.5)**2 + (Y-0.4)**2 + (Z-0.3)**2) / 0.1)
    
    # Sample parameters
    params = {
        'P': 1000.0, 'v': 0.1, 'nx': nx, 'ny': ny, 'nz': nz,
        'LaserX': 0.5, 'LaserY': 0.4, 'LaserZ': 0.3,
        'lam': 1.0, 'c0': 1.0, 'lx': 1.0, 'ly': 0.8, 'lz': 0.6,
        'Lf': 100.0, 'Tm': 300.0, 'a': 1.0, 'b': 1.0, 'c': 1.0,
        'σ': 0.1, 'α': 0.1, 'nt': 100, 'dt': 0.01
    }
    
    # Write HDF5 file
    with h5py.File(filename, 'w') as f:
        # Coordinates group
        coords_group = f.create_group('coordinates')
        coords_group.create_dataset('x', data=x)
        coords_group.create_dataset('y', data=y)
        coords_group.create_dataset('z', data=z)
        
        # Temperature group
        temp_group = f.create_group('temperature')
        temp_group.create_dataset('temperature', data=T)
        
        # Parameters group
        params_group = f.create_group('parameters')
        for key, value in params.items():
            params_group.attrs[key] = value
    
    print(f"Sample HDF5 file created: {filename}")
    return filename

# Create sample data and convert to VTK
sample_file = create_sample_hdf5()
X, Y, Z, T, params = read_hdf5_and_create_vtk(sample_file)

print("HDF5 file successfully read and VTK file created!")
print(f"Data shape: {T.shape}")
print(f"Temperature range: {T.min():.2f} - {T.max():.2f}")
print("Parameters:", list(params.keys()))

# Create a quick matplotlib visualization
fig = plt.figure(figsize=(12, 4))

# 2D slice at middle z
ax1 = fig.add_subplot(131)
mid_z = T.shape[2] // 2
im1 = ax1.imshow(T[:, :, mid_z].T, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
ax1.set_title(f'Temperature at z={Z[0,0,mid_z]:.2f}')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.colorbar(im1, ax=ax1)

# 2D slice at middle y
ax2 = fig.add_subplot(132)
mid_y = T.shape[1] // 2
im2 = ax2.imshow(T[:, mid_y, :].T, origin='lower', extent=[X.min(), X.max(), Z.min(), Z.max()])
ax2.set_title(f'Temperature at y={Y[0,mid_y,0]:.2f}')
ax2.set_xlabel('X')
ax2.set_ylabel('Z')
plt.colorbar(im2, ax=ax2)

# 3D scatter plot of high temperature points
ax3 = fig.add_subplot(133, projection='3d')
high_temp_mask = T > T.mean() + T.std()
high_temp_indices = np.where(high_temp_mask)
ax3.scatter(X[high_temp_indices], Y[high_temp_indices], Z[high_temp_indices], 
           c=T[high_temp_indices], cmap='hot', s=20)
ax3.set_title('High Temperature Regions')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

plt.tight_layout()
plt.show()

print("\nFiles created:")
print("- sample_data.h5 (HDF5 input file)")
print("- sample_data.vtk (VTK file for ParaView)")