import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, RFE


def rf_feature_selection(ML_data_folder):
    cv_folds = 5 # number of folds to use for cross validation
    n_jobs = 5 # number of cores to use for parallel processing of hyperparameter sets, set to -1 to use all available cores

    # Load training data
    x_train = pd.read_pickle('%s/canopy_temp_features_train.pkl' % ML_data_folder)
    y_train = pd.read_pickle('%s/canopy_temp_target_train.pkl' % ML_data_folder)

    feature_corr = x_train.corr() #calculate correlation matrix between all features

    ##load and assign hyperparameter values from initial tuning
    hyperparam_vals = pickle.load(open("%s/best_parameters_initial.pkl" % ML_data_folder, "rb"))

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

    ###Initial feature elimination based on high correlation with higher performing features

    cv_estimator_init = RandomForestRegressor(random_state=13, n_estimators=n_estimators, max_features=max_features,
                                         max_depth=max_depth, min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf, bootstrap= bootstrap,
                                         min_impurity_decrease= min_impurity_decrease)
    cv_estimator_init.fit(x_train, y_train)
    init_feature_importances = np.stack((cv_estimator_init.feature_names_in_, cv_estimator_init.feature_importances_), axis=1)
    init_features_sorted = init_feature_importances[init_feature_importances[:, 1].argsort()]

    features_to_keep = []

    for i in range(len(init_features_sorted)):
        keep_feature = True
        feature_name = init_features_sorted[i, 0]
        for j in range(i+1, len(init_features_sorted)):
            compare_feature_name = init_features_sorted[j, 0]
            abs_corr = abs(feature_corr.loc[feature_name][compare_feature_name])
            if abs_corr > 0.5:
                keep_feature = False
        if keep_feature is True:
            features_to_keep.append(feature_name)

    x_train_subset = x_train[[c for c in x_train.columns if c in features_to_keep]]

    x_train_subset.to_pickle('%s/canopy_temp_features_train_first_feature_subset.pkl' % ML_data_folder)
    x_train_subset.to_csv('%s/csv_Files/canopy_temp_features_train_first_feature_subset.csv' % ML_data_folder)

    ###Recursive feature selection with cross-validation for canopy temp
    cv_estimator = RandomForestRegressor(random_state=13, n_estimators=n_estimators, max_features=max_features,
                                         max_depth=max_depth, min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf, bootstrap= bootstrap,
                                         min_impurity_decrease= min_impurity_decrease)
    #cv_estimator.fit(x_train_subset, y_train) #don't need to refit full model, will do that anyway in the RFECV routine
    cv_selector = RFECV(cv_estimator, cv=cv_folds, step=1, scoring='r2', n_jobs=n_jobs, verbose=3)
    cv_selector = cv_selector.fit(x_train_subset, y_train)
    rfecv_mask = cv_selector.get_support()  # list of booleans
    rfecv_features = []

    rfecv_scores = pd.DataFrame({'score': cv_selector.cv_results_['mean_test_score'], }, columns=['score'])
    threshold_score = 0.95 * max(rfecv_scores['score'])
    succeeding = rfecv_scores[rfecv_scores['score'] > threshold_score]
    num_features = min(succeeding.index) + 1

    final_selector = RFE(cv_estimator, n_features_to_select=num_features, step=1, verbose=3)
    final_selector = final_selector.fit(x_train_subset, y_train)
    rfe_mask = final_selector.get_support()  # list of booleans
    rfe_features = []

    for bool, feature in zip(rfe_mask, x_train_subset.columns):
        if bool:
            rfe_features.append(feature)
    print("Optimal number of features : ", final_selector.n_features_)
    print("Best features : ", rfe_features)
    n_features = x_train_subset.shape[1]

    ## Option to plot the CV scores over the feature selection process
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score \n of number of selected features")
    # plt.plot(range(1, len(cv_selector.cv_results_['split0_test_score']) + 1),
    #          cv_selector.cv_results_['split0_test_score'])
    # plt.plot(range(1, len(cv_selector.cv_results_['split1_test_score']) + 1),
    #          cv_selector.cv_results_['split1_test_score'])
    # plt.plot(range(1, len(cv_selector.cv_results_['split2_test_score']) + 1),
    #          cv_selector.cv_results_['split2_test_score'])
    # plt.plot(range(1, len(cv_selector.cv_results_['split3_test_score']) + 1),
    #          cv_selector.cv_results_['split3_test_score'])
    # plt.plot(range(1, len(cv_selector.cv_results_['split4_test_score']) + 1),
    #          cv_selector.cv_results_['split4_test_score'])
    #
    # plt.savefig('%s/Figures/cv_scores_feature_selection.pdf' % ML_data_folder)

    # Save list of best features
    with open('%s/best_features.pkl' % ML_data_folder, "wb") as f:
        pickle.dump(rfe_features, f)

    with open('%s/csv_Files/best_features.csv' % ML_data_folder, 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(rfe_features)

    combined_feature_rankings = np.stack((final_selector.feature_names_in_, final_selector.ranking_)) #array of features and ranking
    combined_feature_rankings_df = pd.DataFrame(combined_feature_rankings)

    with open('%s/feature_rankings.pkl' % ML_data_folder, "wb") as f:
        pickle.dump(combined_feature_rankings, f)

    combined_feature_rankings_df.to_csv('%s/csv_Files/feature_rankings.csv' % ML_data_folder, index=False, header=False)

    with open('%s/csv_Files/feature_rankings.csv' % ML_data_folder, 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(combined_feature_rankings)

    with open('%s/feature_selection_full_results.pkl' % ML_data_folder, "wb") as f:
        pickle.dump(cv_selector.cv_results_, f)

    with open('%s/csv_Files/feature_selection_full_results.csv' % ML_data_folder, "w", newline="") as fp:
        # Create a writer object
        writer = csv.DictWriter(fp, fieldnames=cv_selector.cv_results_.keys())

        # Write the header row
        writer.writeheader()

        # Write the data rows
        writer.writerow(cv_selector.cv_results_)


if __name__ == '__main__':
    rf_feature_selection(ML_data_folder= "ML_data_files")