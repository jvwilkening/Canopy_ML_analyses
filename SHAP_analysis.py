import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, median_absolute_error

'''
Runs analyses of the RF model, including assessing model performance and running SHAP analysis
'''

def evaluate(model, test_features, test_labels, output_folder):
    #Evaluates RF model performance against the test data set and saves output to txt file
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    difference = predictions - test_labels
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    R2 = r2_score(test_labels, predictions)
    med_error = median_absolute_error(test_labels, predictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    mean_temp_obs = np.mean(test_labels)
    obs_diff_from_mean = abs(test_labels - mean_temp_obs)
    test_features['predicted'] = predictions
    test_features['actual'] = test_labels
    test_features['abs_error'] = errors
    test_features['pred-act'] = difference
    test_features['actual_abs_dev_mean'] = obs_diff_from_mean
    with open("%s/csv_Files/model_performance.txt" % output_folder, "w") as f:
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)), file=f)
        print('Accuracy = {:0.2f}%.'.format(accuracy), file=f)
        print('R2 = {:0.4f}'.format(R2), file=f)
        print('Median absolute error = {:0.4f} degrees'.format(med_error), file=f)
        print('Correlation with difference and features in test set', file=f)
        print(test_features.corrwith(test_features['pred-act']), file=f)
        print('Correlation with abs error and features in test set', file=f)
        print(test_features.corrwith(test_features['abs_error']), file=f)
        f.close()

    plt.figure()
    test_features.plot.hexbin('actual', 'pred-act', bins='log')
    plt.xlim([295,330])
    plt.ylim([-10, 10])
    plt.savefig('%s/Figures/difference_vs_actual_density_plot.pdf' % output_folder)
    plt.savefig('%s/Figures/PNG_Files/difference_vs_actual_density_plot.png' % output_folder)
    plt.close()

    plt.figure()
    test_features.plot.hexbin('actual_abs_dev_mean', 'abs_error', bins='log')
    plt.xlim([0,15])
    plt.ylim([0,10])
    plt.savefig('%s/Figures/abs_value_obs_v_mean.pdf' % output_folder)
    plt.savefig('%s/Figures/PNG_Files/abs_value_obs_v_mean.png' % output_folder)
    plt.close()

    test_features.to_pickle('%s/test_set_with_predictions.pkl' % output_folder)
    test_features.to_csv('%s/csv_Files/test_set_with_predictions.csv' % output_folder)
    plt.figure()
    sns.set_style("ticks")
    test_features.hist('pred-act', bins=120)
    plt.xlim(-7.5, 7.5)
    plt.xlabel('Predicted-Observed Temperature')
    plt.savefig('%s/Figures/prediction_errors_hist.pdf' % output_folder)
    plt.close()
    return accuracy


def SHAP_analysis(ML_data_folder):

    # Load training data
    x_train = pd.read_pickle('%s/canopy_temp_features_train.pkl' % ML_data_folder)
    y_train = pd.read_pickle('%s/canopy_temp_target_train.pkl' % ML_data_folder)

    x_test = pd.read_pickle('%s/canopy_temp_features_test.pkl' % ML_data_folder)
    y_test = pd.read_pickle('%s/canopy_temp_target_test.pkl' % ML_data_folder)

    ##load and assign hyperparameter values from initial tuning
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

    # Load feature names and create dataframe
    best_features = pd.read_pickle('%s/best_features.pkl' % ML_data_folder)
    x_train_best_features = x_train[[c for c in x_train.columns if c in best_features]]
    x_train = x_train_best_features

    x_test_best_features = x_test[[c for c in x_test.columns if c in best_features]]
    x_test = x_test_best_features

    # Define RF model
    model = RandomForestRegressor(random_state=13, n_estimators=n_estimators, max_features=max_features,
                                         max_depth=max_depth, min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf, bootstrap= bootstrap,
                                         min_impurity_decrease= min_impurity_decrease)

    # Fitting the Random Forest Regression model to the data
    model.fit(x_train, y_train)

    # Evaluates model performance against test set
    model_accuracy = evaluate(model, x_test, y_test, ML_data_folder)

    # Subset data for running SHAP analysis (otherwise very slow to run over full dataset and similar output)
    x_train_shap_sample = shap.utils.sample(x_train, nsamples=2000, random_state=13)

    # Run SHAP analysis
    rf_explainer = shap.Explainer(model)
    print("explainer complete")
    shap_values = rf_explainer.shap_values(x_train_shap_sample)
    print("shap_values complete")

    # Saves SHAP output
    np.savetxt('%s/csv_Files/shap_values.csv' % ML_data_folder, shap_values)
    with open('%s/shap_values.pkl' % ML_data_folder, "wb") as f:
        pickle.dump(shap_values, f)
    x_train_shap_sample.to_csv('%s/csv_Files/x_shap_values.csv' % ML_data_folder)
    with open('%s/x_train_shap_sample.pkl' % ML_data_folder, "wb") as f:
        pickle.dump(x_train_shap_sample, f)
    #

    # Plot mean absolute SHAP as bar plot
    plt.figure()
    shap.summary_plot(shap_values, x_train_shap_sample, plot_type="bar", show=False)
    plt.savefig('%s/Figures/shap_summary_bar_plot.pdf' % ML_data_folder)
    plt.close()


    # Plot SHAP values as swarm plot
    plt.figure()
    shap.summary_plot(shap_values, x_train_shap_sample, show=False)
    plt.savefig('%s/Figures/shap_summary_swarm_plot.pdf' % ML_data_folder)
    plt.close()

    # Create dependence plots for each feature
    for i in best_features:
        plt.figure()
        shap.dependence_plot(i, shap_values, x_train_shap_sample, show=False, interaction_index=None, alpha=0.1)
        plt.savefig(ML_data_folder + '/Figures/' + 'shap_dependence_' + str(i) +
                    '.pdf')
        plt.close()
        plt.figure()
        shap.dependence_plot(i, shap_values, x_train_shap_sample, show=False, alpha=0.1)
        plt.savefig(ML_data_folder + '/Figures/' + 'shap_dependence_interaction_' + str(i) +
                    '.pdf')
        plt.close()


if __name__ == '__main__':
    SHAP_analysis(ML_data_folder= "ML_data_files")