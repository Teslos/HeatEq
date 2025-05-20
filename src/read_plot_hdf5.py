import xml.etree.ElementTree as ET
import h5py
import numpy as np
import matplotlib.pyplot as plt

def read_temperature_from_xmf(xmf_filename):
    """
    Read temperature data referenced in an XMF file.

    Args:
        xmf_filename (str): Path to the XMF file

    Returns:
        tuple: (temperature_data, dimensions) where temperature_data is a numpy array
               and dimensions is a tuple of the data dimensions
    """
    try:
        # Parse the XMF file
        tree = ET.parse(xmf_filename)
        root = tree.getroot()

        # Find the Temperature attribute
        temp_attribute = root.find(".//Attribute[@Name='Temperature']")
        if temp_attribute is None:
            raise ValueError("Temperature attribute not found in the XMF file")

        # Get the DataItem element that contains the reference to the HDF5 file
        data_item = temp_attribute.find("DataItem")
        if data_item is None:
            raise ValueError("DataItem not found for Temperature attribute")

        # Extract the HDF5 file path and dataset name
        data_ref = data_item.text.strip()
        h5_file_path, dataset_path = data_ref.split(':')

        # Get dimensions from the DataItem
        dimensions_str = data_item.get('Dimensions', '')
        dimensions = tuple(int(dim) for dim in dimensions_str.split())

        # Read the data from the HDF5 file
        with h5py.File(h5_file_path, 'r') as h5_file:
            temperature_data = h5_file[dataset_path][:]

        return temperature_data, dimensions

    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        return None, None
    except Exception as e:
        print(f"Error reading temperature data: {e}")
        return None, None

def visualize_temperature(temperature_data, dimensions):
    """
    Visualize the temperature data with some basic plots

    Args:
        temperature_data (numpy.ndarray): The temperature data
        dimensions (tuple): The dimensions of the data
    """
    if temperature_data is None:
        print("No data to visualize")
        return

    # Reshape the data if needed
    if temperature_data.shape != dimensions:
        temperature_data = temperature_data.reshape(dimensions)

    # Basic statistics
    print(f"Temperature Data Shape: {temperature_data.shape}")
    print(f"Min Temperature: {np.min(temperature_data)}")
    print(f"Max Temperature: {np.max(temperature_data)}")
    print(f"Mean Temperature: {np.mean(temperature_data)}")

    # Create some visualizations
    plt.figure(figsize=(15, 10))

    # Middle slice in Z direction
    z_mid = dimensions[2] // 2
    plt.subplot(2, 2, 1)
    plt.imshow(temperature_data[:, :, z_mid], cmap='hot')
    plt.colorbar(label='Temperature')
    plt.title(f'Temperature at Z={z_mid}')

    # Middle slice in Y direction
    y_mid = dimensions[1] // 2
    plt.subplot(2, 2, 2)
    plt.imshow(temperature_data[:, y_mid, :], cmap='hot')
    plt.colorbar(label='Temperature')
    plt.title(f'Temperature at Y={y_mid}')

    # Middle slice in X direction
    x_mid = dimensions[0] // 2
    plt.subplot(2, 2, 3)
    plt.imshow(temperature_data[x_mid, :, :], cmap='hot')
    plt.colorbar(label='Temperature')
    plt.title(f'Temperature at X={x_mid}')

    # Histogram of temperature values
    plt.subplot(2, 2, 4)
    plt.hist(temperature_data.flatten(), bins=50)
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')
    plt.title('Temperature Distribution')

    plt.tight_layout()
    plt.savefig('temperature_visualization.png')
    plt.show()

def save_temperature_data(temperature_data, output_file='temperature_data.npy'):
    """
    Save the temperature data to a NumPy file

    Args:
        temperature_data (numpy.ndarray): The temperature data
        output_file (str): Output file name
    """
    if temperature_data is not None:
        np.save(output_file, temperature_data)
        print(f"Temperature data saved to {output_file}")

if __name__ == "__main__":
    xmf_filename = "fields_3d_100.0_0.5.xmf"

    # Read the temperature data
    temperature_data, dimensions = read_temperature_from_xmf(xmf_filename)

    if temperature_data is not None:
        print(f"Successfully read temperature data with dimensions {dimensions}")

        # Save the data
        save_temperature_data(temperature_data)

        # Visualize the data
        visualize_temperature(temperature_data, dimensions)
    else:
        print("Failed to read temperature data")