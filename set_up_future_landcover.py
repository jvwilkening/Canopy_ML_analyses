import numpy as np
import pandas as pd
import os
from utility_functions import (import_env_data, import_gis_data, format_env_data_all_cells, create_cover_df,
                               import_env_cell_data, format_env_data_cell_basis)

def return_visible_files(directory): #ignores system files and only returns actual files
    visible_files = [file for file in os.listdir(directory) if not file.startswith('.')]
    return visible_files
def set_up_future_landcover(ML_data_folder, downscaled_data_folder, cover_frac_increase = 0.05):
    image_files = return_visible_files(downscaled_data_folder)

    combined_impervious_data_df = pd.DataFrame()
    combined_grass_data_df = pd.DataFrame()

    for image_num in range(len(image_files)):
        image_code = image_files[image_num]
        env_data_folder = "GIS_Processed_Data/" + image_code + "/env_variables"
        cell_env_data_folder = "GIS_Processed_Data/" + image_code + "/env_variables/Cell_Basis"
        gis_data_folder = "GIS_Processed_Data/" + image_code

        image_files_list = return_visible_files("GIS_Processed_Data")

        canopy_density_5m, canopy_density_10m, canopy_density_25m, canopy_large_water_dist, canopy_major_road_dist, \
            canopy_park_dist, canopy_road_dist, canopy_traffic_vol_20m, canopy_traffic_vol_100m, canopy_traffic_vol_250m, \
            canopy_traffic_vol_500m, canopy_water_dist, downtown_zone_dist, green_frac, impervious_frac, pop_density_1km, \
            pop_density, TWI = import_env_data(env_data_folder)

        env_variables_df = format_env_data_all_cells(canopy_density_5m, canopy_density_10m, canopy_density_25m,
                                           canopy_large_water_dist, canopy_major_road_dist,
                                           canopy_park_dist, canopy_road_dist, canopy_traffic_vol_20m,
                                           canopy_traffic_vol_100m, canopy_traffic_vol_250m,
                                           canopy_traffic_vol_500m, canopy_water_dist, downtown_zone_dist, green_frac,
                                           impervious_frac, pop_density_1km,
                                           pop_density, TWI)

        _, _, agriculture_frac_df, buildings_frac_df, coniferous_frac_df, deciduous_frac_df, emerg_wetland_frac_df, \
            extraction_frac_df, forest_wetland_frac_df, grass_frac_df, lakes_frac_df, river_frac_df, roads_frac_df, \
            soil_frac_df = import_gis_data(gis_data_folder)

        cover_frac_df = create_cover_df(agriculture_frac_df, buildings_frac_df, coniferous_frac_df, deciduous_frac_df,
                                        emerg_wetland_frac_df, extraction_frac_df, forest_wetland_frac_df, grass_frac_df,
                                        lakes_frac_df, river_frac_df, roads_frac_df, soil_frac_df)


        cells_large_water_dist, cells_park_dist, cells_road_dist, cells_traffic_vol_500m = import_env_cell_data(env_data_folder)

        cell_basis_env_df = format_env_data_cell_basis(cells_large_water_dist, cells_park_dist, cells_road_dist, cells_traffic_vol_500m)


        all_env_variables_df = pd.merge(env_variables_df, cell_basis_env_df, on='Fishnet_ID')

        # If no prior canopy was in cell, use feature values based on cell center/mean, otherwise assume value remains same
        # Applies to water distance, park distance, and traffic volume
        all_env_variables_df['canopy_large_water_dist'] = all_env_variables_df.apply(
            lambda row: row['cells_large_water_dist'] if np.isnan(row['canopy_large_water_dist']) else row['canopy_large_water_dist'],
            axis=1)

        all_env_variables_df['canopy_park_dist'] = all_env_variables_df.apply(
            lambda row: row['cells_park_dist'] if np.isnan(row['canopy_park_dist']) else row['canopy_park_dist'],
            axis=1)

        all_env_variables_df['canopy_road_dist'] = all_env_variables_df.apply(
            lambda row: row['cells_road_dist'] if np.isnan(row['canopy_road_dist']) else row['canopy_road_dist'],
            axis=1)

        all_env_variables_df['canopy_traffic_vol_500m'] = all_env_variables_df.apply(
            lambda row: row['cells_traffic_vol_500m'] if np.isnan(row['canopy_traffic_vol_500m']) else row['canopy_traffic_vol_500m'],
            axis=1)

        # create new cover datasets for increased tree cover, either over impervious surface or over grass
        new_imp_df = pd.merge(cover_frac_df, all_env_variables_df, on='Fishnet_ID')
        new_grass_df = pd.merge(cover_frac_df, all_env_variables_df, on='Fishnet_ID')

        new_imp_df['image_code'] = image_code
        new_imp_df[image_code] = 1

        new_grass_df['image_code'] = image_code
        new_grass_df[image_code] = 1

        def adjust_roads(row):
            if row['roads_frac'] > row['buildings_frac'] and row['roads_frac'] > cover_frac_increase:  # roads above threshold and greater than buildings take all from roads
                val = row['roads_frac'] - cover_frac_increase
            elif row['roads_frac'] > row['buildings_frac'] and row['impervious_frac'] >= cover_frac_increase:  # both under threshold but sum over and roads greater, take first all from road
                val = 0.0
            elif row['roads_frac'] <= row['buildings_frac'] and row['impervious_frac'] >= cover_frac_increase and row['buildings_frac'] < cover_frac_increase:  # both under threshold but sum over and roads less than, take first from buildings
                val = row['roads_frac'] - (cover_frac_increase - row['buildings_frac'])
            else:  # if sum not over threshold or if buildings greater
                val = row['roads_frac']
            return val

        # ID pixels with at least 0.05 grass
        # Add to tree frac
        new_grass_df['tree_frac'] = new_grass_df.apply(
            lambda x: x['tree_frac'] + cover_frac_increase if x['grass_frac'] >= cover_frac_increase else x['tree_frac'],
            axis=1)
        # Add change flag
        new_grass_df['change_flag'] = new_grass_df.apply(lambda x: 1.0 if x['grass_frac'] >= cover_frac_increase else 0.0,
                                                         axis=1)
        # Subtract grass
        new_grass_df['grass_frac'] = new_grass_df.apply(
            lambda x: x['grass_frac'] - cover_frac_increase if x['grass_frac'] >= cover_frac_increase else x['grass_frac'],
            axis=1)

        # ID pixels with at least 0.05 impervious
        # Add to tree frac
        new_imp_df['tree_frac'] = new_imp_df.apply(
            lambda x: x['tree_frac'] + cover_frac_increase if x['impervious_frac'] >= cover_frac_increase else x[
                'tree_frac'], axis=1)
        # Add change flag
        new_imp_df['change_flag'] = new_imp_df.apply(lambda x: 1.0 if x['impervious_frac'] >= cover_frac_increase else 0.0,
                                                     axis=1)
        # Add to green frac
        new_imp_df['green_frac'] = new_imp_df.apply(
            lambda x: x['green_frac'] + cover_frac_increase if x['impervious_frac'] >= cover_frac_increase else x[
                'green_frac'], axis=1)
        # Subtract from whichever is greater out of road and building frac
        new_imp_df['roads_frac'] = new_imp_df.apply(adjust_roads, axis=1)
        # Subtract impervious
        new_imp_df['impervious_frac'] = new_imp_df.apply(
            lambda x: x['impervious_frac'] - cover_frac_increase if x['impervious_frac'] >= cover_frac_increase else x[
                'impervious_frac'], axis=1)
        # Update buildings frac based on roads and impervious frac
        new_imp_df['buildings_frac'] = new_imp_df['impervious_frac'] - new_imp_df['roads_frac']

        combined_impervious_data_df = pd.concat([combined_impervious_data_df, new_imp_df], axis=0, ignore_index=True)
        combined_grass_data_df = pd.concat([combined_grass_data_df, new_grass_df], axis=0, ignore_index=True)

    combined_impervious_data_df = combined_impervious_data_df.fillna(0)
    combined_grass_data_df = combined_grass_data_df.fillna(0)
    combined_impervious_data_df.to_pickle('%s/canopy_increase_over_impervious_full.pkl' % ML_data_folder)
    combined_impervious_data_df.to_csv('%s/csv_Files/canopy_increase_over_impervious_full.csv' % ML_data_folder)

    combined_grass_data_df.to_pickle('%s/canopy_increase_over_grass_full.pkl' % ML_data_folder)
    combined_grass_data_df.to_csv('%s/csv_Files/canopy_increase_over_grass_full.csv' % ML_data_folder)


if __name__ == '__main__':
    set_up_future_landcover(ML_data_folder= 'ML_data_files', downscaled_data_folder= 'Results', cover_frac_increase = 0.05)



