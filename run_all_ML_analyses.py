from import_all_data import import_all_data
from tune_rf_hyperparameters import tune_rf_hyperparameters
from rf_feature_selection import rf_feature_selection
from set_up_future_landcover import set_up_future_landcover
from run_future_canopy_scenario import run_future_canopy
from export_prediction_data_as_raster import export_predicted_data
from SHAP_analysis import SHAP_analysis
from plot_SHAP_data import plot_SHAP_data
import os

'''
Runs all functions to create RF model and then runs future canopy scenario and exports rasters. Then runs SHAP analysis
on RF model and plots corresponding figures of SHAP results.

Date: May 2, 2024
Author: Jeannie Wilkening
'''
##Define directories
#where to output data
ML_data_folder = 'ML_data_files'
#where the downscaled data is saved
downscaled_data_folder = 'Results' #folder with subfolders for each downscaled image
#where environmental feature data is
env_data_folder = 'GIS_Processed_Data'

#Checks if output folder(s) exists and, if not, creates folder(s)
if not os.path.exists(ML_data_folder):
    os.makedirs(ML_data_folder)
ML_data_folder_csv = 'ML_data_files/csv_Files'
ML_data_folder_figs = 'ML_data_files/Figures'
ML_data_folder_figs_png = 'ML_data_files/Figures/PNG_Files'
if not os.path.exists(ML_data_folder_csv):
    os.makedirs(ML_data_folder_csv)
if not os.path.exists(ML_data_folder_figs):
    os.makedirs(ML_data_folder_figs)
if not os.path.exists(ML_data_folder_figs_png):
    os.makedirs(ML_data_folder_figs_png)

###Runs functions
#RF tuning and feature selection
import_all_data(downscaled_data_folder, ML_data_folder, env_data_folder, test_split=0.2) #imports datasets
tune_rf_hyperparameters(ML_data_folder, initial=True) #initial hyperparameter tuning
rf_feature_selection(ML_data_folder) #recursive feature selection
tune_rf_hyperparameters(ML_data_folder, initial=False) #secondary hyperparameter tuning

#Create future cover scenarios and run model for predictions
set_up_future_landcover(ML_data_folder, downscaled_data_folder, cover_frac_increase = 0.05)
run_future_canopy(ML_data_folder)

#Exports predictions as rasters
export_predicted_data(ML_data_folder, downscaled_data_folder=downscaled_data_folder)

#Run SHAP analysis and evaluation of model performance
SHAP_analysis(ML_data_folder= "ML_data_files")

#Plot results of SHAP analysis
plot_SHAP_data(ML_data_folder="ML_data_files")


