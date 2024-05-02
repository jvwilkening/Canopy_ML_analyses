import rasterio
import pandas as pd
import numpy as np
import os
from cover_dict import land_cover_class_short
from utility_functions import import_all_csvs

def return_visible_files(directory): #ignores system files and only returns actual files
    visible_files = [file for file in os.listdir(directory) if not file.startswith('.')]
    return visible_files

def export_predicted_data(ML_data_folder, downscaled_data_folder):
    image_files = return_visible_files(downscaled_data_folder)
    ##Set to true if use combined cover classes (9 total)
    short_cover_set = True


    cover_key = land_cover_class_short

    for image_num in range(len(image_files)):
        image_code = image_files[image_num]
        if image_code == "Figures":
            continue
        #notes https://geobgu.xyz/py/rasterio.html#creating-raster-from-array
        example_raster_file = "GIS_Processed_Data/" + image_code + "/example_raster.tif"
        template_raster = rasterio.open(example_raster_file, "r")
        results_folder = downscaled_data_folder + "/" + image_code + "/out_value_list_kernel_3_moving_1"
        total_rows = template_raster.height
        total_cols = template_raster.width

        df_results = import_all_csvs(results_folder)
        df_results = df_results.rename(columns={"index": "Fishnet_ID"})
        df_results = df_results.drop(columns=['type'])

        temp_df = df_results.loc[df_results['class'] == cover_key['tree']].reset_index()

        temp_df = temp_df.groupby('Fishnet_ID').mean().reset_index()

        temp_df = temp_df[temp_df['value_fraction'] > 0.0000]
        temp_df['delta_T'] = temp_df['out_value'] - temp_df['indep_value']

        LST_df = df_results[df_results['class'] == 1.0]
        LST_df = LST_df.rename(columns={"indep_value": "LST"})
        LST_df = LST_df.groupby('Fishnet_ID').mean().reset_index()

        max_LST = max(LST_df['LST'])
        min_LST = min(LST_df['LST'])

        #Import new cover predictions
        new_imp_df = pd.read_pickle('%s/canopy_increase_over_impervious_predicted.pkl' % ML_data_folder)
        new_grass_df = pd.read_pickle('%s/canopy_increase_over_grass_predicted.pkl' % ML_data_folder)

        # Separate out images
        new_imp_df = new_imp_df[new_imp_df['image_code'] == image_code]
        new_grass_df = new_grass_df[new_grass_df['image_code'] == image_code]

        # Calculate CUTI for new cover set-ups
        new_imp_df['CUTI'] = new_imp_df['tree_frac'] * (max_LST - new_imp_df['predicted']) / (max_LST - min_LST)
        new_grass_df['CUTI'] = new_grass_df['tree_frac'] * (max_LST - new_grass_df['predicted']) / (
                    max_LST - min_LST)

        subfolder = results_folder + '/compiled_dataframes'
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        # save dataframe to results folder as pkl and csv
        new_imp_df.to_pickle('%s/compiled_dataframes/predicted_canopy_temp_inc_imp_df.pkl' % results_folder)
        new_imp_df.to_csv('%s/compiled_dataframes/predicted_canopy_temp_inc_imp_df.csv' % results_folder)

        new_grass_df.to_pickle('%s/compiled_dataframes/predicted_canopy_temp_inc_grass_df.pkl' % results_folder)
        new_grass_df.to_csv('%s/compiled_dataframes/predicted_canopy_temp_inc_grass_df.csv' % results_folder)

        new_imp_df = pd.merge(new_imp_df, LST_df, on='Fishnet_ID')
        new_grass_df = pd.merge(new_grass_df, LST_df, on='Fishnet_ID')

        # Filter out rows where cover wasn't changed
        new_imp_df = new_imp_df[new_imp_df['change_flag'] > 0.0]
        new_grass_df = new_grass_df[new_grass_df['change_flag'] > 0.0]

        georef = template_raster.transform
        existing_crs = template_raster.crs

        res_arr_canopy_temp_imp = np.zeros((total_rows, total_cols))
        res_arr_canopy_temp_imp.fill(-9999)

        res_arr_canopy_ind_imp = np.zeros((total_rows, total_cols))
        res_arr_canopy_ind_imp.fill(-9999)

        res_arr_canopy_temp_grass = np.zeros((total_rows, total_cols))
        res_arr_canopy_temp_grass.fill(-9999)

        res_arr_canopy_ind_grass = np.zeros((total_rows, total_cols))
        res_arr_canopy_ind_grass.fill(-9999)

        for ind in new_imp_df.index:
            row_num = int(new_imp_df['nrow'][ind])
            col_num = int(new_imp_df['ncol'][ind])
            out_val = new_imp_df['predicted'][ind]
            res_arr_canopy_temp_imp[row_num, col_num] = out_val

        m_can_temp_imp = np.flipud(res_arr_canopy_temp_imp)

        for ind in new_imp_df.index:
            row_num = int(new_imp_df['nrow'][ind])
            col_num = int(new_imp_df['ncol'][ind])
            out_val = new_imp_df['CUTI'][ind]
            res_arr_canopy_ind_imp[row_num, col_num] = out_val

        m_can_ind_imp = np.flipud(res_arr_canopy_ind_imp)

        for ind in new_grass_df.index:
            row_num = int(new_grass_df['nrow'][ind])
            col_num = int(new_grass_df['ncol'][ind])
            out_val = new_grass_df['predicted'][ind]
            res_arr_canopy_temp_grass[row_num, col_num] = out_val

        m_can_temp_grass = np.flipud(res_arr_canopy_temp_grass)

        for ind in new_grass_df.index:
            row_num = int(new_grass_df['nrow'][ind])
            col_num = int(new_grass_df['ncol'][ind])
            out_val = new_grass_df['CUTI'][ind]
            res_arr_canopy_ind_grass[row_num, col_num] = out_val

        m_can_ind_grass = np.flipud(res_arr_canopy_ind_grass)

        canopy_temp_imp_file = results_folder + "/canopy_temperature_increase_over_impervious_" + image_code + ".tif"

        new_dataset = rasterio.open(
            canopy_temp_imp_file, "w",
            driver="GTiff",
            height=m_can_temp_imp.shape[0],
            width=m_can_temp_imp.shape[1],
            count=1,
            nodata=-9999,
            dtype=m_can_temp_imp.dtype,
            crs=existing_crs,
            transform=georef
        )

        new_dataset.write(m_can_temp_imp, 1)
        new_dataset.close()

        canopy_ind_imp_file = results_folder + "/canopy_index_increase_over_impervious_" + image_code + ".tif"

        new_dataset = rasterio.open(
            canopy_ind_imp_file, "w",
            driver="GTiff",
            height=m_can_ind_imp.shape[0],
            width=m_can_ind_imp.shape[1],
            count=1,
            nodata=-9999,
            dtype=m_can_ind_imp.dtype,
            crs=existing_crs,
            transform=georef
        )

        new_dataset.write(m_can_ind_imp, 1)
        new_dataset.close()

        canopy_temp_grass_file = results_folder + "/canopy_temperature_increase_over_grass_" + image_code + ".tif"

        new_dataset = rasterio.open(
            canopy_temp_grass_file, "w",
            driver="GTiff",
            height=m_can_temp_grass.shape[0],
            width=m_can_temp_grass.shape[1],
            count=1,
            nodata=-9999,
            dtype=m_can_temp_grass.dtype,
            crs=existing_crs,
            transform=georef
        )

        new_dataset.write(m_can_temp_grass, 1)
        new_dataset.close()

        canopy_ind_grass_file = results_folder + "/canopy_index_increase_over_grass_" + image_code + ".tif"

        new_dataset = rasterio.open(
            canopy_ind_grass_file, "w",
            driver="GTiff",
            height=m_can_ind_grass.shape[0],
            width=m_can_ind_grass.shape[1],
            count=1,
            nodata=-9999,
            dtype=m_can_ind_grass.dtype,
            crs=existing_crs,
            transform=georef
        )

        new_dataset.write(m_can_ind_grass, 1)
        new_dataset.close()



if __name__ == '__main__':
    export_predicted_data(ML_data_folder="ML_data_files", downscaled_data_folder='Results')