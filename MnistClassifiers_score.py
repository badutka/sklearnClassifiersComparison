from sklearn.datasets import fetch_openml, get_data_home
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import time
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import cross_val_score


def set_mocked_data():
    X = [[1, 13, 3], [1, 22, 3],
         [1, 10, 3], [2, 2, 3],
         [2, 3, 3], [2, 4, 3],
         [2, 1, 3], [1, 15, 3],
         [1, 17, 3], [1, 20, 3]]
    X = np.asarray(X)
    y = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    y = np.asarray(y)

    return X, y


def set_data():
    # X, y = set_mocked_data()

    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, cache=True,
                        data_home="C:/Users/crush_000/scikit_learn_data")

    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X, y, X_train, y_train, X_test, y_test


def measure_time(filename_cv: str, filename_tt: str, classifier: str, parameter: str, parameter_values: list):
    X, y, X_train, y_train, X_test, y_test = set_data()

    with open("../measures_mnist/" + filename_cv, 'w') as file:
        for k in parameter_values:
            for i in range(8, 9):
                print("Cross Validation, n_jobs = %s, %s = %s" % (i, parameter, k))
                file.write("Cross Validation, n_jobs = %s, %s = %s\n" % (i, parameter, k))
                # exec("clf = " + classifier + "(n_jobs=i, " + parameter + "=k)", None, locals())
                clf = eval(classifier + "(n_jobs=i, " + parameter + "=k)")
                # score = cross_validate(clf, X, y, cv=10)
                score = cross_val_score(clf, X, y, cv=10)
                print(score.mean())
                file.write(str(score.mean()))

    with open("../measures_mnist/" + filename_tt, 'w') as file:
        for k in parameter_values:
            for i in range(8, 9):
                print("train-test split, n_jobs = %s, %s = %s" % (i, parameter, k))
                file.write("train-test split, n_jobs = %s, %s = %s\n" % (i, parameter, k))
                clf = eval(classifier + "(n_jobs=i, " + parameter + "=k)")
                clf.fit(X_train, y_train)
                score = (clf.score(X_test, y_test))
                print(score)
                file.write(str(score))


def main():
    # ------------------linearRegression------------------
    measure_time("reg/MREGCVLN_score.txt", "reg/MREGTTLN_score.txt", "LinearRegression", "normalize",
                 [False, True])
    measure_time("reg/MREGCVLF_score.txt", "reg/MREGTTLF_score.txt", "LinearRegression", "fit_intercept",
                 [False, True])

    # -------------PassiveAggressive----------------
    measure_time("pa/MPACVMI_score.txt", "pa/MPANTTMI_score.txt", "PassiveAggressiveClassifier", "max_iter",
                 [100, 300, 500, 700, 900, 1000])
    measure_time("pa/MPACVNI_score.txt", "pa/MPANTTNI_score.txt", "PassiveAggressiveClassifier", "n_iter_no_change",
                 [1, 2, 4, 5, 6, 8])
    measure_time("pa/MPACVF_score.txt", "pa/MPANTTF_score.txt", "PassiveAggressiveClassifier", "fit_intercept",
                 [True, False])
    measure_time("pa/MPACVSH_score.txt", "pa/MPANTTSH_score.txt", "PassiveAggressiveClassifier", "shuffle",
                 [True, False])

    # ------------------Random Forest----------
    measure_time("rf/MRFCVP_score.txt", "rf/MRFTTP_score.txt", "RandomForestClassifier", "n_estimators",
                 [10, 20, 30, 11, 14, 17])
    measure_time("rf/MRFCVD_score.txt", "rf/MRFTTD_score.txt", "RandomForestClassifier", "max_depth",
                 [10, 20, 30, 11, 14, 17])

    # -------------ExtraTree
    measure_time("ef/MEFCVP_score.txt", "ef/MEFTTP_score.txt", "ExtraTreesClassifier", "n_estimators",
                 [10, 20, 50, 100, 200, 300])
    measure_time("ef/MEFCVD_score.txt", "ef/MEFTTD_score.txt", "ExtraTreesClassifier", "max_depth",
                 [2, 4, 6, 8, 10, 12])

    # -------------Perceptron-----
    measure_time("perc/MPECVMI_score.txt", "perc/MPETTMI_score.txt", "Perceptron", "max_iter",
                 [100, 300, 500, 700, 900, 1000])
    measure_time("perc/MPECVNI_score.txt", "perc/MPETTNI_score.txt", "Perceptron", "n_iter_no_change",
                 [1, 2, 4, 5, 6, 8])
    measure_time("perc/MPECVF_score.txt", "perc/MPETTF_score.txt", "Perceptron", "fit_intercept", [True, False])
    measure_time("perc/MPECVE0_score.txt", "perc/MPETTE0_score.txt", "Perceptron", "eta0", [1, 2, 3, 4, 5])

    return 0


if __name__ == "__main__":
    main()
