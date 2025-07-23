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
            f.write(f"{key} 1 1 float\n{value}\n")

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
    T = 300 + 100 * np.exp(-((X-0.5)**2 + (Y-0.4)**2 + (Z-0.3)