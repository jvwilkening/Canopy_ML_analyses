import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import pickle
import warnings

def plot_SHAP_data(ML_data_folder):
    #skips over some of the warning messages that can pop up creating a lot of figs
    warnings.filterwarnings("ignore")

    #read in data
    best_features_full = pd.read_pickle('%s/best_features.pkl' % ML_data_folder)
    best_features = best_features_full[:8]

    shap_values = pd.read_pickle('%s/shap_values.pkl' % ML_data_folder)
    x_train_shap_sample = pd.read_pickle('%s/x_train_shap_sample.pkl' % ML_data_folder)

    # Can filter out certain feature values (eg date stamps) if you want
    # shap_values_sub = shap_values[:,:8]
    # x_train_shap_sample_sub = x_train_shap_sample.iloc[:,:8]

    # plot bar plot for mean SHAP value of each feature
    # plt.figure()
    # shap.summary_plot(shap_values_sub, x_train_shap_sample_sub, plot_type="bar", show=False)
    # plt.savefig('%s/Figures/shap_summary_bar_plot.pdf' % ML_data_folder,  dpi=300)
    # plt.savefig('%s/Figures/PNG_Files/shap_summary_bar_plot.png' % ML_data_folder,  dpi=300)
    # plt.close()

    plt.figure()
    shap.summary_plot(shap_values, x_train_shap_sample, plot_type="bar", show=False)
    plt.savefig('%s/Figures/shap_summary_bar_plot_full.pdf' % ML_data_folder,  dpi=300)
    plt.savefig('%s/Figures/PNG_Files/shap_summary_bar_plot_full.png' % ML_data_folder,  dpi=300)
    plt.close()

    #print mean shap values from bar chart
    mean_shap_vals = pd.DataFrame((zip(x_train_shap_sample.columns[np.argsort(np.abs(shap_values).mean(0))][::-1],
    -np.sort(-np.abs(shap_values).mean(0)))),
    columns=["feature", "importance"])
    print(mean_shap_vals)

    # Plot SHAP values
    # plt.figure()
    # shap.summary_plot(shap_values_sub, x_train_shap_sample_sub, show=False, cmap="viridis", alpha=0.7)
    # plt.savefig('%s/Figures/shap_summary_swarm_plot.pdf' % ML_data_folder,  dpi=300)
    # plt.savefig('%s/Figures/PNG_Files/shap_summary_swarm_plot.png' % ML_data_folder,  dpi=300)
    # plt.close()

    plt.figure()
    shap.summary_plot(shap_values, x_train_shap_sample, show=False, cmap="viridis", alpha=0.7)
    plt.savefig('%s/Figures/shap_summary_swarm_plot_full.pdf' % ML_data_folder,  dpi=300)
    plt.savefig('%s/Figures/PNG_Files/shap_summary_swarm_plot_full.png' % ML_data_folder,  dpi=300)
    plt.close()

    for i in best_features_full:
        plt.figure()
        shap.dependence_plot(i, shap_values, x_train_shap_sample, show=False, interaction_index=None, alpha=0.1)
        plt.savefig(ML_data_folder + '/Figures/' + 'shap_dependence_' + str(i) +
                    '.pdf')
        plt.savefig(ML_data_folder + '/Figures/PNG_Files/' + 'shap_dependence_' + str(i) +
                    '.png')
        plt.close()
        plt.figure()
        shap.dependence_plot(i, shap_values, x_train_shap_sample, show=False, alpha=0.1)
        plt.savefig(ML_data_folder + '/Figures/' + 'shap_dependence_interaction_' + str(i) +
                    '.pdf')
        plt.savefig(ML_data_folder + '/Figures/PNG_Files/' + 'shap_dependence_interaction_' + str(i) +
                    '.png')
        plt.close()


if __name__ == '__main__':
    plot_SHAP_data(ML_data_folder='ML_data_files')


