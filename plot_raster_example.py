import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import earthpy.plot as ep

# Define raster file and where to save figures
canopy_temp_file = 'Results/20180810T192556/out_value_list_kernel_3_moving_1/canopy_temperature.tif'
Results_Figures = 'Results/Figures'

#opens raster file
canopy_temp = rxr.open_rasterio(canopy_temp_file, masked=True)

#convert temps to C from K
canopy_temp = canopy_temp - 273.0

# Set colorbar min/max to 5th and 95th percentiles
bar_max = np.nanpercentile(canopy_temp.data, 95)
bar_min = np.nanpercentile(canopy_temp.data, 5)

#plot raster
fig, ax = plt.subplots(figsize=(10, 6))
ep.plot_bands(canopy_temp,
              ax=ax,
              cmap='magma',
              vmin=bar_min,
              vmax=bar_max,
              title="Downscaled Canopy Temperature")
ax.set_facecolor('gainsboro')
plt.savefig('%s/map_canopy_temp.pdf' % Results_Figures, dpi=300)
plt.savefig('%s/PNG_Files/map_canopy_temp' % Results_Figures, dpi=300)
plt.show()