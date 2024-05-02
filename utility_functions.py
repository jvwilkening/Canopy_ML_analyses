import pandas as pd
import glob
import os
from cover_dict import land_cover_class_short

def import_gis_data(raw_data_folder):
    #imports processed GIS data for LST, emissivity, and landcover fraction

    raw_lst_df = pd.read_csv('%s/msp_lst.csv' % raw_data_folder, header=0)
    raw_emisWB_df = pd.read_csv('%s/msp_emisWB.csv' % raw_data_folder, header=0)
    agriculture_frac_df = pd.read_csv('%s/agriculture_frac.csv' % raw_data_folder, header=0)
    buildings_frac_df = pd.read_csv('%s/buildings_frac.csv' % raw_data_folder, header=0)
    coniferous_frac_df = pd.read_csv('%s/coniferous_frac.csv' % raw_data_folder, header=0)
    deciduous_frac_df = pd.read_csv('%s/deciduous_frac.csv' % raw_data_folder, header=0)
    emerg_wetland_frac_df = pd.read_csv('%s/emerg_wetland_frac.csv' % raw_data_folder, header=0)
    extraction_frac_df = pd.read_csv('%s/extraction_frac.csv' % raw_data_folder, header=0)
    forest_wetland_frac_df = pd.read_csv('%s/forest_wetland_frac.csv' % raw_data_folder, header=0)
    grass_frac_df = pd.read_csv('%s/grass_frac.csv' % raw_data_folder, header=0)
    lakes_frac_df = pd.read_csv('%s/lakes_frac.csv' % raw_data_folder, header=0)
    river_frac_df = pd.read_csv('%s/river_frac.csv' % raw_data_folder, header=0)
    roads_frac_df = pd.read_csv('%s/roads_frac.csv' % raw_data_folder, header=0)
    soil_frac_df = pd.read_csv('%s/soil_frac.csv' % raw_data_folder, header=0)

    return raw_lst_df, raw_emisWB_df, agriculture_frac_df, buildings_frac_df, coniferous_frac_df, deciduous_frac_df, \
            emerg_wetland_frac_df, extraction_frac_df, forest_wetland_frac_df, grass_frac_df, lakes_frac_df, \
            river_frac_df, roads_frac_df, soil_frac_df

def import_emis_data():
    #impoorts standardized emissivity values for different landcover types
    emis_vals_df = pd.read_csv('land_cover_emissivities.csv', header=0)
    return emis_vals_df

def import_emis_data_short():
    #impoorts standardized emissivity values for different landcover types
    emis_vals_df = pd.read_csv('land_cover_emissivities_short.csv', header=0)
    return emis_vals_df

def format_covers(raw_lst_df, agriculture_frac_df, buildings_frac_df, coniferous_frac_df, deciduous_frac_df, \
            emerg_wetland_frac_df, extraction_frac_df, forest_wetland_frac_df, grass_frac_df, lakes_frac_df, \
            river_frac_df, roads_frac_df, soil_frac_df, total_pixels, combine_classes = False):

    #find total number of cells in LST dataset
    total_cells = total_pixels

    if combine_classes == True: #combines deciduous, coniferous, and forest_wetland into "tree", extraction into soil, and rivers and lakes into "water"
        deciduous_frac_df['MEAN'] = deciduous_frac_df['MEAN'] + coniferous_frac_df['MEAN'] + forest_wetland_frac_df['MEAN']
        soil_frac_df['MEAN'] = soil_frac_df['MEAN'] + extraction_frac_df['MEAN']
        river_frac_df['MEAN'] = river_frac_df['MEAN'] + lakes_frac_df['MEAN']

        #Fill in -9999 as for missing landcover data cells (think just happens around edges
        agriculture_frac_df = fill_in_cover(agriculture_frac_df, total_cells)
        buildings_frac_df = fill_in_cover(buildings_frac_df, total_cells)
        deciduous_frac_df = fill_in_cover(deciduous_frac_df, total_cells)
        emerg_wetland_frac_df = fill_in_cover(emerg_wetland_frac_df, total_cells)
        #forest_wetland_frac_df = fill_in_cover(forest_wetland_frac_df, total_cells)
        grass_frac_df = fill_in_cover(grass_frac_df, total_cells)
        river_frac_df = fill_in_cover(river_frac_df, total_cells)
        roads_frac_df = fill_in_cover(roads_frac_df, total_cells)
        soil_frac_df = fill_in_cover(soil_frac_df, total_cells)

        #Add in 'Class' column with corresponding land cover ID
        agriculture_frac_df.insert(len(agriculture_frac_df.columns), column='Class', value=land_cover_class_short['agriculture'])
        buildings_frac_df.insert(len(buildings_frac_df.columns), column='Class', value=land_cover_class_short['buildings'])
        deciduous_frac_df.insert(len(deciduous_frac_df.columns), column='Class', value=land_cover_class_short['tree'])
        emerg_wetland_frac_df.insert(len(emerg_wetland_frac_df.columns), column='Class', value=land_cover_class_short['emerg_wetland'])
        #forest_wetland_frac_df.insert(len(forest_wetland_frac_df.columns), column='Class', value=land_cover_class_short['forest_wetland'])
        grass_frac_df.insert(len(grass_frac_df.columns), column='Class', value=land_cover_class_short['grass'])
        river_frac_df.insert(len(river_frac_df.columns), column='Class', value=land_cover_class_short['water'])
        roads_frac_df.insert(len(roads_frac_df.columns), column='Class', value=land_cover_class_short['roads'])
        soil_frac_df.insert(len(soil_frac_df.columns), column='Class', value=land_cover_class_short['soil'])

        #combine all into one dataframe
        df_landcover = pd.concat([agriculture_frac_df, buildings_frac_df, deciduous_frac_df, \
                                  emerg_wetland_frac_df, grass_frac_df, \
                                 river_frac_df, roads_frac_df, soil_frac_df], ignore_index=True)

    else:
        #Fill in -9999 as for missing landcover data cells (think just happens around edges
        agriculture_frac_df = fill_in_cover(agriculture_frac_df, total_cells)
        buildings_frac_df = fill_in_cover(buildings_frac_df, total_cells)
        coniferous_frac_df = fill_in_cover(coniferous_frac_df, total_cells)
        deciduous_frac_df = fill_in_cover(deciduous_frac_df, total_cells)
        emerg_wetland_frac_df = fill_in_cover(emerg_wetland_frac_df, total_cells)
        extraction_frac_df = fill_in_cover(extraction_frac_df, total_cells)
        forest_wetland_frac_df = fill_in_cover(forest_wetland_frac_df, total_cells)
        grass_frac_df = fill_in_cover(grass_frac_df, total_cells)
        lakes_frac_df = fill_in_cover(lakes_frac_df, total_cells)
        river_frac_df = fill_in_cover(river_frac_df, total_cells)
        roads_frac_df = fill_in_cover(roads_frac_df, total_cells)
        soil_frac_df = fill_in_cover(soil_frac_df, total_cells)



        #Add in 'Class' column with corresponding land cover ID
        agriculture_frac_df.insert(len(agriculture_frac_df.columns), column='Class', value=land_cover_class['agriculture'])
        buildings_frac_df.insert(len(buildings_frac_df.columns), column='Class', value=land_cover_class['buildings'])
        coniferous_frac_df.insert(len(coniferous_frac_df.columns), column='Class', value=land_cover_class['coniferous'])
        deciduous_frac_df.insert(len(deciduous_frac_df.columns), column='Class', value=land_cover_class['deciduous'])
        emerg_wetland_frac_df.insert(len(emerg_wetland_frac_df.columns), column='Class', value=land_cover_class['emerg_wetland'])
        extraction_frac_df.insert(len(extraction_frac_df.columns), column='Class', value=land_cover_class['extraction'])
        forest_wetland_frac_df.insert(len(forest_wetland_frac_df.columns), column='Class', value=land_cover_class['forest_wetland'])
        grass_frac_df.insert(len(grass_frac_df.columns), column='Class', value=land_cover_class['grass'])
        lakes_frac_df.insert(len(lakes_frac_df.columns), column='Class', value=land_cover_class['lakes'])
        river_frac_df.insert(len(river_frac_df.columns), column='Class', value=land_cover_class['river'])
        roads_frac_df.insert(len(roads_frac_df.columns), column='Class', value=land_cover_class['roads'])
        soil_frac_df.insert(len(soil_frac_df.columns), column='Class', value=land_cover_class['soil'])

        #combine all into one dataframe
        df_landcover = pd.concat([agriculture_frac_df, buildings_frac_df, coniferous_frac_df, deciduous_frac_df, \
                                  emerg_wetland_frac_df, extraction_frac_df, forest_wetland_frac_df, grass_frac_df, \
                                  lakes_frac_df, river_frac_df, roads_frac_df, soil_frac_df], ignore_index=True)
    return df_landcover

def fill_in_cover(cover_fraction, total_cells, Null_num=-9999.0, rename_column=True):
    total_cells = int(total_cells)
    for i in range(1,total_cells+1):
        flag = i in cover_fraction['Value'].values #Value shows corresponding OID from fishnet
        if flag == False: #if value is missing
            new_row = pd.Series({"OID_":i, "Value":i, "COUNT":1.0, "AREA":0.000000396776248, "MEAN":Null_num})
            cover_fraction = pd.concat([cover_fraction, new_row.to_frame().T], ignore_index=True)
    cover_fraction = cover_fraction.sort_values(by=['Value']).reset_index(drop=True)
    if rename_column is True:
        cover_fraction = cover_fraction.drop(columns=['OID_'])
    cover_fraction = cover_fraction.rename(columns={"Value": "OID"})

    return cover_fraction

def import_env_data(raw_data_folder):
    canopy_density_5m = pd.read_csv('%s/canopy_density_5m.csv' % raw_data_folder, header=0)
    canopy_density_10m = pd.read_csv('%s/canopy_density_10m.csv' % raw_data_folder, header=0)
    canopy_density_25m = pd.read_csv('%s/canopy_density_25m.csv' % raw_data_folder, header=0)
    canopy_large_water_dist = pd.read_csv('%s/canopy_large_water_dist.csv' % raw_data_folder, header=0)
    canopy_major_road_dist = pd.read_csv('%s/canopy_major_road_dist.csv' % raw_data_folder, header=0)
    canopy_park_dist = pd.read_csv('%s/canopy_park_dist.csv' % raw_data_folder, header=0)
    canopy_road_dist = pd.read_csv('%s/canopy_road_dist.csv' % raw_data_folder, header=0)
    canopy_traffic_vol_20m = pd.read_csv('%s/canopy_traffic_vol_20m.csv' % raw_data_folder, header=0)
    canopy_traffic_vol_100m = pd.read_csv('%s/canopy_traffic_vol_100m.csv' % raw_data_folder, header=0)
    canopy_traffic_vol_250m = pd.read_csv('%s/canopy_traffic_vol_250m.csv' % raw_data_folder, header=0)
    canopy_traffic_vol_500m = pd.read_csv('%s/canopy_traffic_vol_500m.csv' % raw_data_folder, header=0)
    canopy_water_dist = pd.read_csv('%s/canopy_water_dist.csv' % raw_data_folder, header=0)
    downtown_zone_dist = pd.read_csv('%s/downtown_zone_dist.csv' % raw_data_folder, header=0)
    green_frac = pd.read_csv('%s/green_frac.csv' % raw_data_folder, header=0)
    impervious_frac = pd.read_csv('%s/impervious_frac.csv' % raw_data_folder, header=0)
    pop_density_1km = pd.read_csv('%s/pop_density_1km.csv' % raw_data_folder, header=0)
    pop_density = pd.read_csv('%s/pop_density.csv' % raw_data_folder, header=0)
    TWI = pd.read_csv('%s/TWI.csv' % raw_data_folder, header=0)

    return canopy_density_5m, canopy_density_10m, canopy_density_25m, canopy_large_water_dist, canopy_major_road_dist, \
        canopy_park_dist, canopy_road_dist, canopy_traffic_vol_20m, canopy_traffic_vol_100m, canopy_traffic_vol_250m, \
           canopy_traffic_vol_500m, canopy_water_dist, downtown_zone_dist, green_frac, impervious_frac, pop_density_1km, \
           pop_density, TWI

def import_env_cell_data(raw_data_folder):
    cells_large_water_dist = pd.read_csv('%s/Cell_Basis/cells_large_water_dist.csv' % raw_data_folder, header=0)
    cells_park_dist = pd.read_csv('%s/Cell_Basis/cells_park_dist.csv' % raw_data_folder, header=0)
    cells_road_dist = pd.read_csv('%s/Cell_Basis/cells_road_dist.csv' % raw_data_folder, header=0)
    cells_traffic_vol_500m = pd.read_csv('%s/Cell_Basis/cells_large_water_dist.csv' % raw_data_folder, header=0)


    return cells_large_water_dist, cells_park_dist, cells_road_dist, cells_traffic_vol_500m

def format_env_data(canopy_density_5m, canopy_density_10m, canopy_density_25m, canopy_large_water_dist, canopy_major_road_dist, \
        canopy_park_dist, canopy_road_dist, canopy_traffic_vol_20m, canopy_traffic_vol_100m, canopy_traffic_vol_250m, \
        canopy_traffic_vol_500m, canopy_water_dist, downtown_zone_dist, green_frac, impervious_frac, pop_density_1km, \
        pop_density, TWI):

        canopy_density_5m = canopy_density_5m.drop(columns=['COUNT', 'AREA', 'OID_'])
        canopy_density_5m = canopy_density_5m.rename(columns={"MEAN": "canopy_density_5m"})

        canopy_density_10m = canopy_density_10m.drop(columns=['COUNT', 'AREA', 'OID_'])
        canopy_density_10m = canopy_density_10m.rename(columns={"MEAN": "canopy_density_10m"})

        canopy_density_25m = canopy_density_25m.drop(columns=['COUNT', 'AREA', 'OID_'])
        canopy_density_25m = canopy_density_25m.rename(columns={"MEAN": "canopy_density_25m"})

        canopy_large_water_dist = canopy_large_water_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
        canopy_large_water_dist = canopy_large_water_dist.rename(columns={"MEAN": "canopy_large_water_dist"})

        canopy_major_road_dist = canopy_major_road_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
        canopy_major_road_dist = canopy_major_road_dist.rename(columns={"MEAN": "canopy_major_road_dist"})

        canopy_park_dist = canopy_park_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
        canopy_park_dist = canopy_park_dist.rename(columns={"MEAN": "canopy_park_dist"})

        canopy_road_dist = canopy_road_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
        canopy_road_dist = canopy_road_dist.rename(columns={"MEAN": "canopy_road_dist"})

        canopy_traffic_vol_20m = canopy_traffic_vol_20m.drop(columns=['COUNT', 'AREA', 'OID_'])
        canopy_traffic_vol_20m = canopy_traffic_vol_20m.rename(columns={"MEAN": "canopy_traffic_vol_20m"})

        canopy_traffic_vol_100m = canopy_traffic_vol_100m.drop(columns=['COUNT', 'AREA', 'OID_'])
        canopy_traffic_vol_100m = canopy_traffic_vol_100m.rename(columns={"MEAN": "canopy_traffic_vol_100m"})

        canopy_traffic_vol_250m = canopy_traffic_vol_250m.drop(columns=['COUNT', 'AREA', 'OID_'])
        canopy_traffic_vol_250m = canopy_traffic_vol_250m.rename(columns={"MEAN": "canopy_traffic_vol_250m"})

        canopy_traffic_vol_500m = canopy_traffic_vol_500m.drop(columns=['COUNT', 'AREA', 'OID_'])
        canopy_traffic_vol_500m = canopy_traffic_vol_500m.rename(columns={"MEAN": "canopy_traffic_vol_500m"})

        canopy_water_dist = canopy_water_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
        canopy_water_dist = canopy_water_dist.rename(columns={"MEAN": "canopy_water_dist"})

        downtown_zone_dist = downtown_zone_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
        downtown_zone_dist = downtown_zone_dist.rename(columns={"MEAN": "downtown_zone_dist"})

        green_frac = green_frac.drop(columns=['COUNT', 'AREA', 'OID_'])
        green_frac = green_frac.rename(columns={"MEAN": "green_frac"})

        impervious_frac = impervious_frac.drop(columns=['COUNT', 'AREA', 'OID_'])
        impervious_frac = impervious_frac.rename(columns={"MEAN": "impervious_frac"})

        pop_density_1km = pop_density_1km.drop(columns=['COUNT', 'AREA', 'OID_'])
        pop_density_1km = pop_density_1km.rename(columns={"MEAN": "pop_density_1km"})

        pop_density = pop_density.drop(columns=['COUNT', 'AREA', 'OID_'])
        pop_density = pop_density.rename(columns={"MEAN": "pop_density"})

        TWI = TWI.drop(columns=['COUNT', 'AREA', 'OID_'])
        TWI = TWI.rename(columns={"MEAN": "TWI"})

        merged_env_data = pd.merge(canopy_density_5m, canopy_density_10m, on='Value')
        merged_env_data = pd.merge(merged_env_data, canopy_density_25m, on='Value')
        merged_env_data = pd.merge(merged_env_data, canopy_large_water_dist, on='Value')
        merged_env_data = pd.merge(merged_env_data, canopy_major_road_dist, on='Value')
        merged_env_data = pd.merge(merged_env_data, canopy_road_dist, on='Value')
        merged_env_data = pd.merge(merged_env_data, canopy_traffic_vol_20m, on='Value')
        merged_env_data = pd.merge(merged_env_data, canopy_traffic_vol_100m, on='Value')
        merged_env_data = pd.merge(merged_env_data, canopy_traffic_vol_250m, on='Value')
        merged_env_data = pd.merge(merged_env_data, canopy_traffic_vol_500m, on='Value')
        merged_env_data = pd.merge(merged_env_data, canopy_water_dist, on='Value')
        merged_env_data = pd.merge(merged_env_data, downtown_zone_dist, on='Value')
        merged_env_data = pd.merge(merged_env_data, green_frac, on='Value')
        merged_env_data = pd.merge(merged_env_data, impervious_frac, on='Value')
        merged_env_data = pd.merge(merged_env_data, pop_density, on='Value')
        merged_env_data = pd.merge(merged_env_data, pop_density_1km, on='Value')
        merged_env_data = pd.merge(merged_env_data, TWI, on='Value')
        merged_env_data = pd.merge(merged_env_data, canopy_park_dist, on='Value')

        merged_env_data = merged_env_data.rename(columns={"Value": "Fishnet_ID"})

        return merged_env_data


def format_env_data_all_cells(canopy_density_5m, canopy_density_10m, canopy_density_25m, canopy_large_water_dist,
                    canopy_major_road_dist, canopy_park_dist, canopy_road_dist, canopy_traffic_vol_20m,
                    canopy_traffic_vol_100m,canopy_traffic_vol_250m, canopy_traffic_vol_500m, canopy_water_dist,
                    downtown_zone_dist, green_frac, impervious_frac, pop_density_1km, pop_density, TWI):
    canopy_density_5m = canopy_density_5m.drop(columns=['COUNT', 'AREA', 'OID_'])
    canopy_density_5m = canopy_density_5m.rename(columns={"MEAN": "canopy_density_5m"})

    canopy_density_10m = canopy_density_10m.drop(columns=['COUNT', 'AREA', 'OID_'])
    canopy_density_10m = canopy_density_10m.rename(columns={"MEAN": "canopy_density_10m"})

    canopy_density_25m = canopy_density_25m.drop(columns=['COUNT', 'AREA', 'OID_'])
    canopy_density_25m = canopy_density_25m.rename(columns={"MEAN": "canopy_density_25m"})

    canopy_large_water_dist = canopy_large_water_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
    canopy_large_water_dist = canopy_large_water_dist.rename(columns={"MEAN": "canopy_large_water_dist"})

    canopy_major_road_dist = canopy_major_road_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
    canopy_major_road_dist = canopy_major_road_dist.rename(columns={"MEAN": "canopy_major_road_dist"})

    canopy_park_dist = canopy_park_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
    canopy_park_dist = canopy_park_dist.rename(columns={"MEAN": "canopy_park_dist"})

    canopy_road_dist = canopy_road_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
    canopy_road_dist = canopy_road_dist.rename(columns={"MEAN": "canopy_road_dist"})

    canopy_traffic_vol_20m = canopy_traffic_vol_20m.drop(columns=['COUNT', 'AREA', 'OID_'])
    canopy_traffic_vol_20m = canopy_traffic_vol_20m.rename(columns={"MEAN": "canopy_traffic_vol_20m"})

    canopy_traffic_vol_100m = canopy_traffic_vol_100m.drop(columns=['COUNT', 'AREA', 'OID_'])
    canopy_traffic_vol_100m = canopy_traffic_vol_100m.rename(columns={"MEAN": "canopy_traffic_vol_100m"})

    canopy_traffic_vol_250m = canopy_traffic_vol_250m.drop(columns=['COUNT', 'AREA', 'OID_'])
    canopy_traffic_vol_250m = canopy_traffic_vol_250m.rename(columns={"MEAN": "canopy_traffic_vol_250m"})

    canopy_traffic_vol_500m = canopy_traffic_vol_500m.drop(columns=['COUNT', 'AREA', 'OID_'])
    canopy_traffic_vol_500m = canopy_traffic_vol_500m.rename(columns={"MEAN": "canopy_traffic_vol_500m"})

    canopy_water_dist = canopy_water_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
    canopy_water_dist = canopy_water_dist.rename(columns={"MEAN": "canopy_water_dist"})

    downtown_zone_dist = downtown_zone_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
    downtown_zone_dist = downtown_zone_dist.rename(columns={"MEAN": "downtown_zone_dist"})

    green_frac = green_frac.drop(columns=['COUNT', 'AREA', 'OID_'])
    green_frac = green_frac.rename(columns={"MEAN": "green_frac"})

    impervious_frac = impervious_frac.drop(columns=['COUNT', 'AREA', 'OID_'])
    impervious_frac = impervious_frac.rename(columns={"MEAN": "impervious_frac"})

    pop_density_1km = pop_density_1km.drop(columns=['COUNT', 'AREA', 'OID_'])
    pop_density_1km = pop_density_1km.rename(columns={"MEAN": "pop_density_1km"})

    pop_density = pop_density.drop(columns=['COUNT', 'AREA', 'OID_'])
    pop_density = pop_density.rename(columns={"MEAN": "pop_density"})

    TWI = TWI.drop(columns=['COUNT', 'AREA', 'OID_'])
    TWI = TWI.rename(columns={"MEAN": "TWI"})

    merged_env_data = pd.merge(canopy_density_5m, canopy_density_10m, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, canopy_density_25m, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, canopy_large_water_dist, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, canopy_major_road_dist, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, canopy_road_dist, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, canopy_traffic_vol_20m, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, canopy_traffic_vol_100m, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, canopy_traffic_vol_250m, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, canopy_traffic_vol_500m, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, canopy_water_dist, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, downtown_zone_dist, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, green_frac, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, impervious_frac, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, pop_density, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, pop_density_1km, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, TWI, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, canopy_park_dist, on='Value', how='outer')

    merged_env_data = merged_env_data.rename(columns={"Value": "Fishnet_ID"})

    return merged_env_data

def format_env_data_cell_basis(cells_large_water_dist, cells_park_dist, cells_road_dist, cells_traffic_vol_500m):
    cells_large_water_dist = cells_large_water_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
    cells_large_water_dist = cells_large_water_dist.rename(columns={"MEAN": "cells_large_water_dist"})

    cells_park_dist = cells_park_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
    cells_park_dist = cells_park_dist.rename(columns={"MEAN": "cells_park_dist"})

    cells_road_dist = cells_road_dist.drop(columns=['COUNT', 'AREA', 'OID_'])
    cells_road_dist = cells_road_dist.rename(columns={"MEAN": "cells_road_dist"})

    cells_traffic_vol_500m = cells_traffic_vol_500m.drop(columns=['COUNT', 'AREA', 'OID_'])
    cells_traffic_vol_500m = cells_traffic_vol_500m.rename(columns={"MEAN": "cells_traffic_vol_500m"})


    merged_env_data = pd.merge(cells_large_water_dist, cells_park_dist, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, cells_road_dist, on='Value', how='outer')
    merged_env_data = pd.merge(merged_env_data, cells_traffic_vol_500m, on='Value', how='outer')

    merged_env_data = merged_env_data.rename(columns={"Value": "Fishnet_ID"})

    return merged_env_data


def create_cover_df(agriculture_frac_df, buildings_frac_df, coniferous_frac_df, deciduous_frac_df, \
            emerg_wetland_frac_df, extraction_frac_df, forest_wetland_frac_df, grass_frac_df, lakes_frac_df, \
            river_frac_df, roads_frac_df, soil_frac_df):

        agriculture_frac_df = agriculture_frac_df.drop(columns=['COUNT', 'AREA', 'OID_'])
        agriculture_frac_df = agriculture_frac_df.rename(columns={"MEAN": "agriculture_frac"})

        buildings_frac_df = buildings_frac_df.drop(columns=['COUNT', 'AREA', 'OID_'])
        buildings_frac_df = buildings_frac_df.rename(columns={"MEAN": "buildings_frac"})

        coniferous_frac_df = coniferous_frac_df.drop(columns=['COUNT', 'AREA', 'OID_'])
        coniferous_frac_df = coniferous_frac_df.rename(columns={"MEAN": "coniferous_frac"})

        deciduous_frac_df = deciduous_frac_df.drop(columns=['COUNT', 'AREA', 'OID_'])
        deciduous_frac_df = deciduous_frac_df.rename(columns={"MEAN": "deciduous_frac"})

        emerg_wetland_frac_df = emerg_wetland_frac_df.drop(columns=['COUNT', 'AREA', 'OID_'])
        emerg_wetland_frac_df = emerg_wetland_frac_df.rename(columns={"MEAN": "emerg_wetland_frac"})

        extraction_frac_df = extraction_frac_df.drop(columns=['COUNT', 'AREA', 'OID_'])
        extraction_frac_df = extraction_frac_df.rename(columns={"MEAN": "extraction_frac"})

        forest_wetland_frac_df = forest_wetland_frac_df.drop(columns=['COUNT', 'AREA', 'OID_'])
        forest_wetland_frac_df = forest_wetland_frac_df.rename(columns={"MEAN": "forest_wetland_frac"})

        grass_frac_df = grass_frac_df.drop(columns=['COUNT', 'AREA', 'OID_'])
        grass_frac_df = grass_frac_df.rename(columns={"MEAN": "grass_frac"})

        lakes_frac_df = lakes_frac_df.drop(columns=['COUNT', 'AREA', 'OID_'])
        lakes_frac_df = lakes_frac_df.rename(columns={"MEAN": "lakes_frac"})

        river_frac_df = river_frac_df.drop(columns=['COUNT', 'AREA', 'OID_'])
        river_frac_df = river_frac_df.rename(columns={"MEAN": "river_frac"})

        roads_frac_df = roads_frac_df.drop(columns=['COUNT', 'AREA', 'OID_'])
        roads_frac_df = roads_frac_df.rename(columns={"MEAN": "roads_frac"})

        soil_frac_df = soil_frac_df.drop(columns=['COUNT', 'AREA', 'OID_'])
        soil_frac_df = soil_frac_df.rename(columns={"MEAN": "soil_frac"})

        tree_df = pd.merge(deciduous_frac_df, coniferous_frac_df, on='Value')
        tree_df = pd.merge(tree_df, forest_wetland_frac_df, on='Value')
        tree_df['tree_frac'] = tree_df['coniferous_frac'] + tree_df['deciduous_frac'] + tree_df['forest_wetland_frac']
        tree_df = tree_df.drop(columns=['coniferous_frac', 'deciduous_frac', 'forest_wetland_frac'])

        merged_cover_data = pd.merge(agriculture_frac_df, buildings_frac_df, on='Value')
        merged_cover_data = pd.merge(merged_cover_data, tree_df, on='Value')
        merged_cover_data = pd.merge(merged_cover_data, emerg_wetland_frac_df, on='Value')
        merged_cover_data = pd.merge(merged_cover_data, extraction_frac_df, on='Value')
        #merged_cover_data = pd.merge(merged_cover_data, forest_wetland_frac_df, on='Value')
        merged_cover_data = pd.merge(merged_cover_data, grass_frac_df, on='Value')
        merged_cover_data = pd.merge(merged_cover_data, lakes_frac_df, on='Value')
        merged_cover_data = pd.merge(merged_cover_data, river_frac_df, on='Value')
        merged_cover_data = pd.merge(merged_cover_data, roads_frac_df, on='Value')
        merged_cover_data = pd.merge(merged_cover_data, soil_frac_df, on='Value')

        merged_cover_data = merged_cover_data.rename(columns={"Value": "Fishnet_ID"})

        return merged_cover_data

def import_all_csvs(results_folder):
    csv_files = glob.glob(os.path.join(results_folder, "*.csv"))
    counter = 0
    for f in csv_files:
        if counter == 0:
            # read the csv file
            df_results = pd.read_csv(f)
        else:
            file_results = pd.read_csv(f)
            df_results = pd.concat([df_results, file_results])
        counter = counter + 1
    return df_results