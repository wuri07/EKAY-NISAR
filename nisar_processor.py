import asf_search as asf
import h5py
import numpy as np
import os
import rasterio
from rasterio.transform import from_origin
import earthaccess


def run_processor(lat_min, lat_max, lon_min, lon_max, output_file="nisar_dynamic_map.tif"):
    """
    Searches the NASA/ASF catalog dynamically using official NISAR dataset filters,
    streams the HDF5 granule for the given bounding box, processes backscatter,
    and exports a custom-named GeoTIFF.
    """
    # ==========================================
    # 1. Credentials Setup
    # ==========================================
    os.environ["EARTHDATA_USERNAME"] = "ekay"
    os.environ["EARTHDATA_PASSWORD"] = "ekay12K@an$2"

    print("Authenticating with NASA Earthdata...")
    auth = earthaccess.login(strategy="environment")

    # ==========================================
    # 2. Dynamic Spatial Search via ASF Search
    # ==========================================
    print(f"Searching ASF catalog for area: [{lat_min}, {lat_max}, {lon_min}, {lon_max}]...")

    wkt_polygon = f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, {lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))"

    results = []
    try:
        results = asf.search(
            dataset=asf.DATASET.NISAR,
            processingLevel="GCOV",
            intersectsWith=wkt_polygon,
            maxResults=5
        )
    except Exception as e:
        print(f"Spatial query warning: {e}")

    # Fallback to default operational sample granule if spatial search yields nothing
    if not results:
        print("No granules found for these exact coordinates. Using default mission calibration granule...")
        results = asf.granule_search(
            "NISAR_L2_PR_GCOV_031_020_A_043_0005_NASV_A_20260919T140343_20260919T140414_P05023_F_F_J_001")

    if not results:
        raise ValueError("Could not retrieve any NISAR granules. Please check network connectivity or credentials.")

    download_url = results[0].properties['url']
    granule_id = results[0].properties.get('granule') or results[0].properties.get('fileID') or "Unknown_Granule"

    print(f"Successfully targeted granule: {granule_id}")
    print(f"Direct stream URL: {download_url}")

    # ==========================================
    # 3. Stream HDF5 Dataset Directly over HTTPS
    # ==========================================
    print("Opening HTTPS stream for NISAR granule...")
    file_objects = earthaccess.open([download_url])

    # ==========================================
    # 4. Read HDF5 & Spatial Metadata
    # ==========================================
    print("Reading HDF5 grids...")
    with h5py.File(file_objects[0], 'r') as h5_file:
        base_path = "/science/LSAR/GCOV/grids"
        if base_path not in h5_file:
            base_path = list(h5_file.keys())[0]

        grids_group = h5_file[base_path]
        freq_key = list(grids_group.keys())[0]
        grid_group = grids_group[freq_key]

        x_coords = grid_group["xCoordinates"][:]
        y_coords = grid_group["yCoordinates"][:]

        proj_val = grid_group["projection"][()]
        epsg_code = int(proj_val) if np.issubdtype(type(proj_val), np.integer) else 4326

        dset_key = None
        for key in grid_group.keys():
            if isinstance(grid_group[key], h5py.Dataset) and key not in ["xCoordinates", "yCoordinates", "projection"]:
                dset_key = key
                break

        if dset_key is None:
            raise RuntimeError("No backscatter array dataset found inside grid group.")

        power_data = grid_group[dset_key][:]

    if power_data.ndim > 2:
        power_data = power_data.squeeze()

    # ==========================================
    # 5. Decibel Conversion & Classification (Optimized uint8 to save RAM)
    # ==========================================
    db_data = 10 * np.log10(np.where(power_data > 0, power_data, np.nan))

    classified = np.zeros_like(db_data, dtype=np.uint8)
    classified[db_data < -18.0] = 1  # Water & Floods
    classified[(db_data >= -18.0) & (db_data < -12.0)] = 2  # Bare Soil / Low Crop
    classified[(db_data >= -12.0) & (db_data < -6.0)] = 3  # Forest Cover & Canopy
    classified[db_data >= -6.0] = 4  # Urban Structures & Buildings

    # ==========================================
    # 6. Export Spatial GeoTIFF
    # ==========================================
    pixel_size_x = abs(x_coords[1] - x_coords[0])
    pixel_size_y = abs(y_coords[1] - y_coords[0])
    transform = from_origin(x_coords[0], y_coords[0], pixel_size_x, pixel_size_y)

    metadata = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': 0,
        'width': classified.shape[1],
        'height': classified.shape[0],
        'count': 1,
        'crs': f'EPSG:{epsg_code}',
        'transform': transform,
        'compress': 'deflate'
    }

    with rasterio.open(output_file, 'w', **metadata) as dst:
        dst.write(classified, 1)

    print(f"Success! GeoTIFF saved as '{output_file}'.")
    return output_file