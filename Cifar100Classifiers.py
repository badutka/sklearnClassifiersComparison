import os
import sys
import time
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveClassifier


def load_dataset(path=".."):
    # training data
    data = pickle.load(open(os.path.join(path, "cifar100", "train"), 'rb'), encoding='latin1')
    X_train = data['data']
    y_train = np.asarray(data['fine_labels'], np.int8)

    # test data
    data = pickle.load(open(os.path.join(path, 'cifar100', 'test'), 'rb'), encoding='latin1')
    X_test = data['data']
    y_test = np.asarray(data['fine_labels'], np.int8)

    # reshape
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    nsamples, nx, ny, nz = X_train.shape
    X_train = X_train.reshape((nsamples, nx * ny * nz))
    nsamples, nx, ny, nz = X_test.shape
    X_test = X_test.reshape((nsamples, nx * ny * nz))
    X, y = np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X, y, X_train, y_train, X_test, y_test


def measure_time(filename_cv: str, filename_tt: str, classifier: str, parameter: str, parameter_values: list):
    X, y, X_train, y_train, X_test, y_test = load_dataset()

    with open("../measures_cifar100/" + filename_cv, 'w') as file:
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

    with open("../measures_cifar100/" + filename_tt, 'w') as file:
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


def main():

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

    # ------------------linearRegression------------------
    measure_time("reg/MREGCVLN.txt", "reg/MREGTTLN.txt", "LinearRegression", "normalize",
                 [False, True])

    measure_time("reg/MREGCVLF.txt", "reg/MREGTTLF.txt", "LinearRegression", "fit_intercept",
                 [False, True])

    # -------------PassiveAggresive----------------
    measure_time("pa/MPACVMI.txt", "pa/MPANTTMI.txt", "PassiveAggressiveClassifier", "max_iter",
                 [100, 300, 500, 700, 900, 1000])
    measure_time("pa/MPACVNI.txt", "pa/MPANTTNI.txt", "PassiveAggressiveClassifier", "n_iter_no_change",
                 [1, 2, 4, 5, 6, 8])
    measure_time("pa/MPACVF.txt", "pa/MPANTTF.txt", "PassiveAggressiveClassifier", "fit_intercept",
                 [True, False])
    measure_time("pa/MPACVSH.txt", "pa/MPANTTSH.txt", "PassiveAggressiveClassifier", "shuffle",
                 [True, False])

    # ------------------Random Forest----------
    measure_time("rf/MRFCVP.txt", "rf/MRFTTP.txt", "RandomForestClassifier", "n_estimators", [10, 20, 50, 100, 200, 300])
    measure_time("rf/MRFCVD.txt", "rf/MRFTTD.txt", "RandomForestClassifier", "max_depth", [2, 4, 6, 8, 10, 12])

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
