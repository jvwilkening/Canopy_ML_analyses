import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor


def run_future_canopy(ML_data_folder):
    # Use feature set identified by feature selection procedure
    best_features = pd.read_pickle('%s/best_features.pkl' % ML_data_folder)

    x_train = pd.read_pickle('%s/canopy_temp_features_train.pkl' % ML_data_folder)
    y_train = pd.read_pickle('%s/canopy_temp_target_train.pkl' % ML_data_folder)
    x_train_best_features = x_train[[c for c in x_train.columns if c in best_features]]
    x_train = x_train_best_features

    new_imp_df = pd.read_pickle('%s/canopy_increase_over_impervious_full.pkl' % ML_data_folder)
    new_grass_df = pd.read_pickle('%s/canopy_increase_over_grass_full.pkl' % ML_data_folder)

    imp_features = new_imp_df[[c for c in new_imp_df.columns if c in best_features]]
    grass_features = new_grass_df[[c for c in new_grass_df.columns if c in best_features]]

    imp_features = imp_features[x_train.columns]
    grass_features = grass_features[x_train.columns]

    ##load and assign hyperparameter values from tuning procedure
    hyperparam_vals = pickle.load(open("%s/best_parameters_secondary.pkl" % ML_data_folder, "rb"))

    # Number of trees in random forest
    n_estimators = hyperparam_vals['n_estimators']
    # Number of features to consider at every split
    max_features = hyperparam_vals['max_features']
    # Maximum number of levels in tree
    max_depth = hyperparam_vals['max_depth']
    # Minimum number of samples required to split a node
    min_samples_split = hyperparam_vals['min_samples_split']
    # Minimum number of samples required at each leaf node
    min_samples_leaf = hyperparam_vals['min_samples_leaf']
    # Method of selecting samples for training each tree
    bootstrap = hyperparam_vals['bootstrap']
    # Minimum reduction in node impurity at each split
    min_impurity_decrease = hyperparam_vals['min_impurity_decrease']

    # Initialize RF model
    model = RandomForestRegressor(random_state=13, n_estimators=n_estimators, max_features=max_features,
                                         max_depth=max_depth, min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf, bootstrap= bootstrap,
                                         min_impurity_decrease= min_impurity_decrease)

    # Fitting the Random Forest Regression model to the data
    model.fit(x_train, y_train)

    temp_pred_imp = model.predict(imp_features)
    new_imp_df['predicted'] = temp_pred_imp

    temp_pred_grass = model.predict(grass_features)
    new_grass_df['predicted'] = temp_pred_grass

    new_imp_df.to_pickle('%s/canopy_increase_over_impervious_predicted.pkl' % ML_data_folder)
    new_imp_df.to_csv('%s/csv_Files/canopy_increase_over_impervious_predicted.csv' % ML_data_folder)

    new_grass_df.to_pickle('%s/canopy_increase_over_grass_predicted.pkl' % ML_data_folder)
    new_grass_df.to_csv('%s/csv_Files/canopy_increase_over_grass_predicted.csv' % ML_data_folder)

if __name__ == '__main__':
    run_future_canopy(ML_data_folder='ML_data_files')



