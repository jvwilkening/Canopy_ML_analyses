from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import pickle
import csv
import numpy as np
import pandas as pd
from pprintpp import pprint
import initial_hyperparameter_vals


def tune_rf_hyperparameters(ML_data_folder, initial):
    search_iter = 200  # Iterations to perform in random hyperparameter search
    cv_folds = 3  # number of folds to use for cross validation
    n_jobs = 4  # number of cores to use for parallel processing of hyperparameter sets, set to -1 to use all available cores

    if initial is True: #create ranges from initial_hyperparameter_vals file
    # Create the random grid
        random_grid = {'n_estimators': initial_hyperparameter_vals.n_estimators,
                       'max_features': initial_hyperparameter_vals.max_features,
                       'max_depth': initial_hyperparameter_vals.max_depth,
                       'min_samples_split': initial_hyperparameter_vals.min_samples_split,
                       'min_samples_leaf': initial_hyperparameter_vals.min_samples_leaf,
                       'bootstrap': initial_hyperparameter_vals.bootstrap,
                       'min_impurity_decrease': initial_hyperparameter_vals.min_impurity_decrease}
    else: #create ranges based on initial pass results
        hyperparam_vals = pickle.load(open("%s/best_parameters_initial.pkl" % ML_data_folder, "rb"))
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=max(25, hyperparam_vals['n_estimators']-100),
                                                    stop=(hyperparam_vals['n_estimators']+100), num=15)]
        # Number of features to consider at every split
        max_features = ['sqrt', 'log2', hyperparam_vals['max_features'] - 1, hyperparam_vals['max_features'],
                        hyperparam_vals['max_features'] + 1]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(max(1, hyperparam_vals['max_depth']-10),
                                                 (hyperparam_vals['max_depth']+10), num=10)]
        # Minimum number of samples required to split a node
        if hyperparam_vals['min_samples_split'] > 2:
            min_samples_split = [hyperparam_vals['min_samples_split'] - 1, hyperparam_vals['min_samples_split'],
                                hyperparam_vals['min_samples_leaf'] + 1]
        else:
            min_samples_split = [hyperparam_vals['min_samples_split'],
                                hyperparam_vals['min_samples_split'] + 1, hyperparam_vals['min_samples_split'] + 2]
        # Minimum number of samples required at each leaf node
        if hyperparam_vals['min_samples_leaf'] > 1:
            min_samples_leaf = [hyperparam_vals['min_samples_leaf'] - 1, hyperparam_vals['min_samples_leaf'],
                                hyperparam_vals['min_samples_leaf'] + 1]
        else:
            min_samples_leaf = [hyperparam_vals['min_samples_leaf'],
                                hyperparam_vals['min_samples_leaf'] + 1, hyperparam_vals['min_samples_leaf'] + 2]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Minimum reduction in node impurity at each split
        min_impurity_decrease = [hyperparam_vals['min_impurity_decrease']*0.1, hyperparam_vals['min_impurity_decrease'],
                                 hyperparam_vals['min_impurity_decrease']*10.0]
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap,
                       'min_impurity_decrease': min_impurity_decrease}
    #Load training data
    x_train = pd.read_pickle('%s/canopy_temp_features_train.pkl' % ML_data_folder)
    y_train = pd.read_pickle('%s/canopy_temp_target_train.pkl' % ML_data_folder)

    if initial is False: #if secondary tuning use parsimonious feature set from feature selection
        best_features = pd.read_pickle('%s/best_features.pkl' % ML_data_folder)
        x_train_best_features = x_train[[c for c in x_train.columns if c in best_features]]
        x_train = x_train_best_features


    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across  different combinations, and using set number of cores (think can set to -1 to use all available cores)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=search_iter, cv=cv_folds,
                                   verbose=2, random_state=42, n_jobs=n_jobs)
    # Fit the random search model
    rf_random.fit(x_train, y_train)

    # Prints best params
    pprint(rf_random.best_params_)

    # Save tuning results
    if initial is True:
        with open('%s/best_parameters_initial.pkl' % ML_data_folder, "wb") as f:
            pickle.dump(rf_random.best_params_, f)

        with open('%s/csv_Files/best_parameters_initial.csv' % ML_data_folder, "w", newline="") as fp:
            # Create a writer object
            writer = csv.DictWriter(fp, fieldnames=rf_random.best_params_.keys())

            # Write the header row
            writer.writeheader()

            # Write the data rows
            writer.writerow(rf_random.best_params_)

        with open('%s/initial_param_results.pkl' % ML_data_folder, "wb") as f:
            pickle.dump(rf_random.cv_results_, f)

        with open('%s/csv_Files/initial_param_results.csv' % ML_data_folder, "w", newline="") as fp:
            # Create a writer object
            writer = csv.DictWriter(fp, fieldnames=rf_random.cv_results_.keys())

            # Write the header row
            writer.writeheader()

            # Write the data rows
            writer.writerow(rf_random.cv_results_)
    else:
        with open('%s/best_parameters_secondary.pkl' % ML_data_folder, "wb") as f:
            pickle.dump(rf_random.best_params_, f)

        with open('%s/csv_Files/best_parameters_secondary.csv' % ML_data_folder, "w", newline="") as fp:
            # Create a writer object
            writer = csv.DictWriter(fp, fieldnames=rf_random.best_params_.keys())

            # Write the header row
            writer.writeheader()

            # Write the data rows
            writer.writerow(rf_random.best_params_)

        with open('%s/secondary_param_results.pkl' % ML_data_folder, "wb") as f:
            pickle.dump(rf_random.cv_results_, f)

        with open('%s/csv_Files/secondary_param_results.csv' % ML_data_folder, "w", newline="") as fp:
            # Create a writer object
            writer = csv.DictWriter(fp, fieldnames=rf_random.cv_results_.keys())

            # Write the header row
            writer.writeheader()

            # Write the data rows
            writer.writerow(rf_random.cv_results_)


if __name__ == '__main__':
    tune_rf_hyperparameters(ML_data_folder= "ML_data_files", initial=False)