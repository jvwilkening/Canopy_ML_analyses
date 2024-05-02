from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from utility_functions import import_env_data, import_gis_data, format_env_data, create_cover_df, import_all_csvs
from cover_dict import land_cover_class_short

def return_visible_files(directory): #ignores system files and only returns actual files
    visible_files = [file for file in os.listdir(directory) if not file.startswith('.')]
    return visible_files

def import_all_data(downscaled_data_folder, ML_data_folder, env_input_data_folder, test_split=0.2):
    image_files = return_visible_files(downscaled_data_folder)
    cover_key = land_cover_class_short

    combined_image_data_df = pd.DataFrame()

    #imports data for all images (in individual folders) within results folder
    for image_num in range(len(image_files)):
        image_code = image_files[image_num]
        if image_code == "Figures": #If have figure folder in there skip it
            continue
        results_folder = downscaled_data_folder + "/" + image_code + "/out_value_list_kernel_3_moving_1"
        env_data_folder = env_input_data_folder + "/" + image_code + "/env_variables"
        gis_data_folder = env_input_data_folder + "/" + image_code

        df_results = import_all_csvs(results_folder)

        df_results = df_results.rename(columns={"index": "Fishnet_ID"})
        temp_df = df_results.pivot_table(index='Fishnet_ID', columns='class', values='out_value').reset_index()
        temp_df = temp_df.add_suffix('_temp')

        tree_key = land_cover_class_short['tree']
        temp_key = str(tree_key) + '.0_temp'
        canopy_temp_df = temp_df[["Fishnet_ID_temp", temp_key]]
        canopy_temp_df = canopy_temp_df.rename(columns={"Fishnet_ID_temp": "Fishnet_ID"})
        canopy_temp_df = canopy_temp_df.rename(columns={temp_key: "canopy_temp"})
        canopy_temp_df = canopy_temp_df.groupby('Fishnet_ID').mean().reset_index()

        canopy_density_5m, canopy_density_10m, canopy_density_25m, canopy_large_water_dist, canopy_major_road_dist, \
        canopy_park_dist, canopy_road_dist, canopy_traffic_vol_20m, canopy_traffic_vol_100m, canopy_traffic_vol_250m, \
        canopy_traffic_vol_500m, canopy_water_dist, downtown_zone_dist, green_frac, impervious_frac, pop_density_1km, \
        pop_density, TWI = import_env_data(env_data_folder)

        env_variables_df = format_env_data(canopy_density_5m, canopy_density_10m, canopy_density_25m,
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

        full_dataset_df = pd.merge(canopy_temp_df, env_variables_df, on='Fishnet_ID')
        full_dataset_df = pd.merge(full_dataset_df, cover_frac_df, on='Fishnet_ID')
        full_dataset_df['image_code'] = image_code
        full_dataset_df[image_code] = 1

        combined_image_data_df = pd.concat([combined_image_data_df, full_dataset_df], axis=0, ignore_index=True)

    combined_image_data_df = combined_image_data_df.fillna(0)
    combined_image_data_df.to_pickle('%s/full_canopy_temp_df.pkl' % ML_data_folder)
    combined_image_data_df.to_csv('%s/csv_Files/full_canopy_temp_df.csv' % ML_data_folder)

    # Filter to only cells with non-zero canopy cover
    canopy_pixels_df = combined_image_data_df[combined_image_data_df['tree_frac'] > 0.0000]

    canopy_pixels_df.to_pickle('%s/canopy_pixels_canopy_temp_df.pkl' % ML_data_folder)
    canopy_pixels_df.to_csv('%s/csv_Files/canopy_pixels_canopy_temp_df.csv' % ML_data_folder)

    canopy_temp_features = canopy_pixels_df.drop(columns=['canopy_temp', 'Fishnet_ID', 'image_code'])  # Features
    canopy_temp_target = canopy_pixels_df['canopy_temp']

    canopy_temp_features.to_pickle('%s/canopy_temp_features.pkl' % ML_data_folder)
    canopy_temp_features.to_csv('%s/csv_Files/canopy_temp_features.csv' % ML_data_folder)

    canopy_temp_target.to_pickle('%s/canopy_temp_target.pkl' % ML_data_folder)
    canopy_temp_target.to_csv('%s/csv_Files/canopy_temp_target.csv' % ML_data_folder)

    #split data set into test and train sets and save datasets
    x_train, x_test, y_train, y_test = train_test_split(canopy_temp_features, canopy_temp_target, test_size=test_split,
                                                        random_state=28)

    x_train.to_pickle('%s/canopy_temp_features_train.pkl' % ML_data_folder)
    x_train.to_csv('%s/csv_Files/canopy_temp_features_train.csv' % ML_data_folder)

    x_test.to_pickle('%s/canopy_temp_features_test.pkl' % ML_data_folder)
    x_test.to_csv('%s/csv_Files/canopy_temp_features_test.csv' % ML_data_folder)

    y_train.to_pickle('%s/canopy_temp_target_train.pkl' % ML_data_folder)
    y_train.to_csv('%s/csv_Files/canopy_temp_target_train.csv' % ML_data_folder)

    y_test.to_pickle('%s/canopy_temp_target_test.pkl' % ML_data_folder)
    y_test.to_csv('%s/csv_Files/canopy_temp_target_test.csv' % ML_data_folder)


if __name__ == '__main__':
    import_all_data(downscaled_data_folder = "Results", ML_data_folder = "ML_data_files", env_input_data_folder = "GIS_Processed_Data",
                    test_split=0.2)

