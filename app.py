import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

from nisar_processor import run_processor

st.set_page_config(page_title="Global NISAR Radar Monitor", page_icon="🌍", layout="wide")

st.title("🌍 Global NISAR Radar Landform & Disaster Monitor")
st.markdown(
    "Stream and analyze active satellite radar streams across up to 4 countries simultaneously, tracking flood waters through storms and clouds.")

# ==========================================
# 1. Sidebar Configuration: Multi-Country & Category
# ==========================================
st.sidebar.header("🌍 Region & Country Selector")

# Dictionary mapping countries to bounding boxes [lat_min, lat_max, lon_min, lon_max]
country_bboxes = {
    "United States": (37.5, 38.5, -121.5, -120.5),  # California Central Valley
    "Brazil": (-4.0, -3.0, -60.0, -59.0),  # Amazon Rainforest
    "India": (25.0, 26.0, 82.0, 83.0),  # Gangetic Plain
    "Italy": (44.5, 45.5, 10.0, 11.0),  # Po Valley
    "Egypt": (30.0, 31.0, 31.0, 32.0),  # Nile Delta
    "Australia": (-34.0, -33.0, 145.0, 146.0),  # Murray-Darling Basin
    "Japan": (35.5, 36.0, 139.5, 140.0),  # Tokyo Metropolitan
    "Germany": (50.0, 51.0, 7.0, 8.0)  # Rhine River Basin
}

selected_countries = st.sidebar.multiselect(
    "Choose 1 to 4 countries to analyze:",
    options=list(country_bboxes.keys()),
    default=["United States"],
    max_selections=4
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Visualization Mode")
monitoring_mode = st.sidebar.selectbox(
    "Select Target Category:",
    [
        "🌊 Surface Water Mask Inspection (VV Band)",
        "🌊 Flood & Standing Water Extent (< -18 dB)",
        "🌾 Bare Soil & Low Crops (-18 to -12 dB)",
        "🌲 Forest Cover & Canopy (-12 to -6 dB)",
        "🏙️ Urban Structures (≥ -6 dB)",
        "🗺️ Full Multi-Class Map View"
    ]
)

fetch_btn = st.sidebar.button("🔄 Fetch & Process Selected Countries")

# ==========================================
# 2. Execution & Multi-Country Rendering Loop
# ==========================================
if fetch_btn:
    if not selected_countries:
        st.warning("⚠️ Please select at least one country from the sidebar.")
    else:
        st.info(f"Processing data for: {', '.join(selected_countries)}...")

        # Create side-by-side columns based on the number of selected countries
        cols = st.columns(len(selected_countries))

        for idx, country in enumerate(selected_countries):
            lat_min, lat_max, lon_min, lon_max = country_bboxes[country]
            output_filename = f"nisar_{country.lower().replace(' ', '_')}.tif"

            with cols[idx]:
                st.subheader(f"📍 {country}")
                with st.spinner(f"Streaming HDF5 & processing {country}..."):
                    try:
                        # Calls your backend processor
                        run_processor(lat_min, lat_max, lon_min, lon_max, output_file=output_filename)

                        # Read generated GeoTIFF for rendering
                        with rasterio.open(output_filename) as src:
                            data = src.read(1)
                            bounds = src.bounds

                        # Render Matplotlib Figure based on Category Mode
                        fig, ax = plt.subplots(figsize=(5, 4), facecolor="white")
                        ax.set_facecolor("#f0f0f0")

                        if "Surface Water Mask Inspection" in monitoring_mode:
                            masked_data = np.where(data == 1, 1.0, np.nan)
                            im = ax.imshow(masked_data, cmap="Blues", vmin=0, vmax=1,
                                           extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
                            ax.set_title("Surface Water Detection (VV)", fontsize=10, fontweight='bold')
                            ax.axis('off')
                            st.pyplot(fig)

                        elif "Full Multi-Class Map View" in monitoring_mode:
                            colors = ['white', '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
                            cmap = ListedColormap(colors)
                            bounds_norm = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
                            norm = BoundaryNorm(bounds_norm, cmap.N)

                            im = ax.imshow(data, cmap=cmap, norm=norm,
                                           extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
                            ax.set_title("Multi-Landform Classification", fontsize=10, fontweight='bold')
                            ax.axis('off')

                            # Add professional colorbar legend
                            cbar = fig.colorbar(im, ticks=[0, 1, 2, 3, 4], orientation='horizontal', pad=0.2,
                                                shrink=0.9)
                            cbar.set_ticklabels(['No Data', 'Water', 'Soil', 'Forest', 'Urban'])
                            cbar.ax.tick_params(labelsize=7)
                            st.pyplot(fig)

                        else:
                            if "Flood" in monitoring_mode:
                                display_mask = np.where(data == 1, 1, 0)
                                cmap_name, title_str = 'Blues', "Flood Extent"
                            elif "Soil" in monitoring_mode:
                                display_mask = np.where(data == 2, 1, 0)
                                cmap_name, title_str = 'YlOrBr', "Bare Soil"
                            elif "Forest" in monitoring_mode:
                                display_mask = np.where(data == 3, 1, 0)
                                cmap_name, title_str = 'Forest Canopy'
                            else:
                                display_mask = np.where(data == 4, 1, 0)
                                cmap_name, title_str = 'Reds', "Urban Structures"

                            im = ax.imshow(display_mask, cmap=cmap_name,
                                           extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
                            ax.set_title(title_str, fontsize=10, fontweight='bold')
                            ax.axis('off')
                            st.pyplot(fig)

                        # Display metrics
                        water_count = np.sum(data == 1)
                        total = data.size
                        pct = (water_count / total) * 100
                        st.metric("Detected Water Coverage", f"{water_count} px", f"{pct:.2f}% of area")

                    except Exception as e:
                        st.error(f"Error processing {country}: {e}")
else:
    st.info(
        "👈 Select up to 4 countries and your visualization mode from the sidebar, then click **'Fetch & Process Selected Countries'** to start live analysis!")