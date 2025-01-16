# Canopy_ML_analyses

Updated: May 2024 

This repository contains the code used for the machine learning analyses in the manuscript: Wilkening and Feng(2025) "Canopy temperature reveals disparities in urban canopy benefits" published in AGU Advances.

This code is run in Python 3.10, and the necessary accompanying packages and versions are listed in the requirements.txt file.

Author Info: Jean Wilkening (jvwilkening@berkeley.edu)

### Code

All the analyses can be run using the run_all_ML_analyses.py script, which will import and format the data from downscaling analyses, execute the random forest model tuning and feature selection, use the model to predict canopy temperature for the new cover scenarios, and perform SHAP analysis on the model. These are all run by individual functions within the script, which reside in the other separate python scripts within the repository (import_all_data.py, tune_rf_hyperparameters.py, rf_feature_selection.py, set_up_future_landcover.py, run_future_canopy.py, export_predicted_data.py, SHAP_analysis.py, and plot_SHAP_data.py). The run_all script should be able to be run by setting up and defining the directories for the input data, and the output will be saved into a separate ML_data_folder directory.

As example input, there are downscaled data for one of the analyzed images in the Results directory, and the GIS_Processed_Data contains the land cover data for that image. The cover_dict.py file contains the key for the landcover data, and the initial_hyperparameter_vals.py contains the values over which to conduct the initial hyperparameter tuning. 

The utility_functions.py stores additional helper functions that are used throughout the code. The plot_raster_example.py shows an example script for creating the raster plots that are used throughout the manuscript.
