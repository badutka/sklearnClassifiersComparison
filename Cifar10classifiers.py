import numpy as np
import time
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveClassifier


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


batch1 = unpickle("../cifar10/data_batch_1")
batch2 = unpickle("../cifar10/data_batch_2")
batch3 = unpickle("../cifar10/data_batch_3")
batch4 = unpickle("../cifar10/data_batch_4")
batch5 = unpickle("../cifar10/data_batch_5")
test_batch = unpickle("../cifar10/test_batch")


def load_data0(btch):
    labels = btch[b'labels']
    imgs = btch[b'data'].reshape((-1, 32, 32, 3))

    res = []
    for ii in range(imgs.shape[0]):
        img = imgs[ii].copy()
        # img = np.transpose(img.flatten().reshape(3,32,32))
        img = np.fliplr(np.rot90(np.transpose(img.flatten().reshape(3, 32, 32)), k=-1))
        res.append(img)
    imgs = np.stack(res)
    return labels, imgs


def load_data():
    x_train_l = []
    y_train_l = []
    for ibatch in [batch1, batch2, batch3, batch4, batch5]:
        labels, imgs = load_data0(ibatch)
        x_train_l.append(imgs)
        y_train_l.extend(labels)
    x_train = np.vstack(x_train_l)
    y_train = np.vstack(y_train_l)

    x_test_l = []
    y_test_l = []
    labels, imgs = load_data0(test_batch)
    x_test_l.append(imgs)
    y_test_l.extend(labels)
    x_test = np.vstack(x_test_l)
    y_test = np.vstack(y_test_l)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()

nsamples, nx, ny, nz = x_train.shape
X_train = x_train.reshape((nsamples, nx * ny * nz))
y_train = np.ravel(y_train)
nsamples, nx, ny, nz = x_test.shape
X_test = x_test.reshape((nsamples, nx * ny * nz))
y_test = np.ravel(y_test)
X, y = np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])


def measure_time(filename_cv: str, filename_tt: str, classifier: str, parameter: str, parameter_values: list):
    with open("../measures_cifar10/" + filename_cv, 'w') as file:
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

    with open("../measures_cifar10/" + filename_tt, 'w') as file:
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
