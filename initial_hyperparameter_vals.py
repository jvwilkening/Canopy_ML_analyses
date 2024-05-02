import numpy as np

###Set initial values to use to form hyperparameter search space

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=50, stop=500, num=10)]
# Number of features to consider at every split
max_features = ['sqrt', 'log2', 8, 10]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 25, num=10)]
#max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 4, 6]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 3, 5]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Minimum reduction in node impurity at each split
min_impurity_decrease = [1.0e-5, 1.0e-10, 1.0e-3]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'min_impurity_decrease': min_impurity_decrease}