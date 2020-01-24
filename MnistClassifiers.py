from sklearn.datasets import fetch_openml, get_data_home
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.model_selection import cross_validate
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveClassifier


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
            for i in range(1, 9):
                print("Cross Validation, n_jobs = %s, %s = %s" % (i, parameter, k))
                file.write("Cross Validation, n_jobs = %s, %s = %s\n" % (i, parameter, k))
                for j in range(10):
                    t0 = time.time()
                    # exec("clf = " + classifier + "(n_jobs=i, " + parameter + "=k)", None, locals())
                    clf = eval(classifier + "(n_jobs=i, " + parameter + "=k)")
                    score = cross_validate(clf, X, y, cv=10)
                    run_time = time.time() - t0
                    print(run_time)
                    file.write(str(run_time) + '\n')

    with open("../measures_mnist/" + filename_tt, 'w') as file:
        for k in parameter_values:
            for i in range(1, 9):
                print("train-test split, n_jobs = %s, %s = %s" % (i, parameter, k))
                file.write("train-test split, n_jobs = %s, %s = %s\n" % (i, parameter, k))
                for j in range(10):
                    t0 = time.time()
                    clf = eval(classifier + "(n_jobs=i, " + parameter + "=k)")
                    clf.fit(X_train, y_train)
                    score = (clf.score(X_test, y_test))
                    run_time = time.time() - t0
                    print(run_time)
                    file.write(str(run_time) + '\n')


def measure_time_regLG(filename_cv: str, filename_tt: str, classifier: str, parameter: str, parameter_values: list):
    X, y, X_train, y_train, X_test, y_test = set_data()

    with open("../measures_mnist/" + filename_cv, 'w') as file:
        for k in parameter_values:
            for i in range(1, 9):
                print("Cross Validation, n_jobs = %s, %s = %s" % (i, parameter, k))
                file.write("Cross Validation, n_jobs = %s, %s = %s\n" % (i, parameter, k))
                for j in range(10):
                    t0 = time.time()
                    # exec("clf = " + classifier + "(n_jobs=i, " + parameter + "=k)", None, locals())
                    clf = eval(classifier + "(n_jobs=i, max_iter=1000," + parameter + "=k)")
                    score = cross_validate(clf, X, y, cv=10)
                    run_time = time.time() - t0
                    print(run_time)
                    file.write(str(run_time) + '\n')

    with open("../measures_mnist/" + filename_tt, 'w') as file:
        for k in parameter_values:
            for i in range(1, 9):
                print("train-test split, n_jobs = %s, %s = %s" % (i, parameter, k))
                file.write("train-test split, n_jobs = %s, %s = %s\n" % (i, parameter, k))
                for j in range(10):
                    t0 = time.time()
                    clf = eval(classifier + "(n_jobs=i, max_iter=1000," + parameter + "=k)")
                    clf.fit(X_train, y_train)
                    score = (clf.score(X_test, y_test))
                    run_time = time.time() - t0
                    print(run_time)
                    file.write(str(run_time) + '\n')


def main():
    # # ------------------ KNN ------------------
    # measure_time("knn/MKNNCVN.txt", "knn/MKNNTTN.txt", "KNeighborsClassifier", "n_neighbors", [3, 5])
    # measure_time("knn/MKNNCVN.txt", "knn/MKNNTTN.txt", "KNeighborsClassifier", "n_neighbors", [3, 5, 7, 9, 11, 13])
    # measure_time("knn/MKNNCVLS.txt", "knn/MKNNTTLS.txt", "KNeighborsClassifier", "leaf_size", [10, 30, 60, 100, 150, 210])
    # measure_time("knn/MKNNCVP.txt", "knn/MKNNTTP.txt", "KNeighborsClassifier", "p", [2, 5, 8, 11, 14, 17])

    # # ------------------Bagging-------------
    # measure_time("bag/MBAGCVLS.txt", "bag/MBAGTTLS.txt", "BaggingClassifier", "n_estimators",
    #              [10, 20, 50, 100, 200, 300])
    # measure_time("bag/MBAGCVLS.txt", "bag/MBAGTTLS.txt", "BaggingClassifier", "max_samples",
    #              [1, 2, 3, 4, 5, 6])

    # # ------------------linearRegression------------------
    # measure_time("reg/MREGCVLN.txt", "reg/MREGTTLN.txt", "LinearRegression", "normalize",
    #              [False, True])
    # measure_time("reg/MREGCVLF.txt", "reg/MREGTTLF.txt", "LinearRegression", "fit_intercept",
    #              [False, True])

    # # ---------------LogisticRegression------------------
    # measure_time_regLG("regLG/MREGLGCVLF.txt", "reg/MREGLGTTLF.txt", "LogisticRegression", "fit_intercept",
    #               [False, True])

    # ------------------linearRegression------------------
    measure_time("reg/MREGCVLN.txt", "reg/MREGTTLN.txt", "LinearRegression", "normalize",
                 [False, True])
    measure_time("reg/MREGCVLF.txt", "reg/MREGTTLF.txt", "LinearRegression", "fit_intercept",
                 [False, True])

    # -------------PassiveAggressive----------------
    measure_time("pa/MPACVMI.txt", "pa/MPANTTMI.txt", "PassiveAggressiveClassifier", "max_iter",
                 [100, 300, 500, 700, 900, 1000])
    measure_time("pa/MPACVNI.txt", "pa/MPANTTNI.txt", "PassiveAggressiveClassifier", "n_iter_no_change",
                 [1, 2, 4, 5, 6, 8])
    measure_time("pa/MPACVF.txt", "pa/MPANTTF.txt", "PassiveAggressiveClassifier", "fit_intercept",
                 [True, False])
    measure_time("pa/MPACVSH.txt", "pa/MPANTTSH.txt", "PassiveAggressiveClassifier", "shuffle",
                 [True, False])

    # ------------------Random Forest----------
    measure_time("rf/MRFCVP.txt", "rf/MRFTTP.txt", "RandomForestClassifier", "n_estimators", [10, 20, 30, 11, 14, 17])
    measure_time("rf/MRFCVD.txt", "rf/MRFTTD.txt", "RandomForestClassifier", "max_depth", [10, 20, 30, 11, 14, 17])

    # -------------ExtraTree
    measure_time("ef/MEFCVP.txt", "ef/MEFTTP.txt", "ExtraTreesClassifier", "n_estimators", [10, 20, 50, 100, 200, 300])
    measure_time("ef/MEFCVD.txt", "ef/MEFTTD.txt", "ExtraTreesClassifier", "max_depth", [2, 4, 6, 8, 10, 12])

    # -------------Perceptron-----
    measure_time("perc/MPECVMI.txt", "perc/MPETTMI.txt", "Perceptron", "max_iter", [100, 300, 500, 700, 900, 1000])
    measure_time("perc/MPECVNI.txt", "perc/MPETTNI.txt", "Perceptron", "n_iter_no_change", [1, 2, 4, 5, 6, 8])
    measure_time("perc/MPECVF.txt", "perc/MPETTF.txt", "Perceptron", "fit_intercept", [True, False])
    measure_time("perc/MPECVE0.txt", "perc/MPETTE0.txt", "Perceptron", "eta0", [1, 2, 3, 4, 5])
    return 0


if __name__ == "__main__":
    main()
