# DSLR
Introduction to machine learning by applying logistic regression.

Requires python3, numpy, pandas and plotly modules.

pip install plotly or python3 -m pip install plotly.

The data used to learn is in Resources/

Run:

python3 describe.py resources/dataset_train.csv (Display information about all features)

python3 histogram.py resources/dataset_train.csv (Display a histogram of each feature)

python3 scatter_plot.py resources/dataset_train.csv (Display a scatter plot of the two most similar features)

python3 pair_plot.py resources/dataset_train.csv (Display a scatter plot matrix of the features that will be used)

python3 logreg_train.py resources/dataset_train.csv (Export logistic regression weights in weights.npy)

python3 logreg_predict.py resources/dataset_test.csv weights.npy (Export logistic regression prediction in houses.csv)
