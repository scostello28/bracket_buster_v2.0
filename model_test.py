import pandas as pd
import pickle
import numpy as np

from sklearn import metrics
from sklearn.model_selection import cross_val_score as cvs

from filters import games_up_to_2021_tourney_filter, tourney2021_filter, games_up_to_2021_tourney_filter, data_for_model, set_up_data
from scraping_utils import check_for_file


if __name__ == "__main__":

    source_dir = "3_model_data"
    data = pd.read_pickle(f"{source_dir}/gamelog_exp_clust.pkl")

    # test models
    Xy_train, Xy_test = data_for_model(data, feature_set='exp_tcf')
    Xy_train_no_clust, Xy_test_no_clust = data_for_model(data, feature_set='gamelogs')

    X_train, y_train, X_test, y_test = set_up_data(Xy_train, Xy_test)
    X_train_no_clust, y_train_no_clust, X_test_no_clust, y_test_no_clust = set_up_data(Xy_train_no_clust, Xy_test_no_clust)

    models = {}
    model_paths = [
        "lr_2021_fit_model_testing",
        "rf_2021_fit_model_testing",
        "gb_2021_fit_model_testing",
        "lr_2021_fit_model_no_clust_testing",
        "rf_2021_fit_model_no_clust_testing",
        "gb_2021_fit_model_no_clust_testing"
    ]

    model_dir_path = 'fit_models'

    for model_path in model_paths:
        with open(f"{model_dir_path}/{model_path}.pkl", 'rb') as f:
            pickled_model = pickle.load(f)
            models[model_path] = pickled_model

    for model_name, model in models.items():
        if "no_clust" in model_name:
            Xt, yt, X, y = X_train_no_clust, y_train_no_clust, X_test_no_clust, y_test_no_clust
        else:
            Xt, yt, X, y = X_train, y_train, X_test, y_test

        # cv_score = np.mean(cvs(models, Xt, yt, scoring='accuracy', cv=5))
        y_hat = model.predict(X)
        score = metrics.accuracy_score(y, y_hat)

        print("--------------------------")
        print(f"Model: {model_name}")
        # print(f"CV Score: {cv_score}")
        print(f"Accuracy: {score:2f}")
        print("\n")
