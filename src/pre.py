from __future__ import print_function

import dataset
import numpy as np
import augment as aug
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


class Pre(object):
    def __init__(self, name=str(), rootdir=str(), data_thresh=0, verbose=False):
        """
        Defaults to an empty dataset descriptor.
        """
        self.dataset_ = dataset.Descriptor(name, rootdir, data_thresh, verbose)

    @classmethod
    def createData(cls, name, rootdir, data_thresh=0, verbose=False):
        return cls(name, rootdir, data_thresh, verbose)

    @classmethod
    def loadData(cls, directory):
        p = Pre()
        p.dataset_.importFile(directory)
        return p

    def saveData(self, directory):
        self.dataset_.exportFile(directory)

    def renameData(self, name):
        self.dataset_.data["name"] = name

    def getClassCount(self):
        _, y = self.dataset_.get_data_paths_and_labels()
        return len(set(y))

    def get_train_and_test_data(self):

        x, y = self.dataset_.get_data_paths_and_labels()

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        ss  = sss.split(x, y)
        train_index, test_index = ss.next()

        X_train  = [x[i] for i in train_index]
        y_train = [y[i] for i in train_index]

        X_test   = [x[i] for i in test_index]
        y_test  = [y[i] for i in test_index]

        yield X_train, y_train, X_test, y_test

    def get_cv_train_and_test_data(self, n_splits=4, balance=False):

        x, y = self.dataset_.get_data_paths_and_labels()
        # x, y = self.dataset_.get_n_vs_many_data_paths_and_labels(n_classes=3)

        x, y = np.array(x), np.array(y)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        for train_index, test_index in skf.split(x, y):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if balance:
                # Balanced training data
                btd = dataset.Descriptor("balanced_train_data")
                btd.populate_paths_labels(X_train.tolist(), y_train.tolist())
                btd.balance()
                X_train, y_train = btd.get_data_paths_and_labels()
                X_train, y_train = np.array(X_train), np.array(y_train)

            yield X_train, y_train, X_test, y_test

    def get_augmented_train_and_test_data(self, n_augmentations=10, n_splits=5, balance=False, augmentation_dir="../scratch/augmented/"):

        split_id = 0

        for X_train, y_train, X_test, y_test in self.get_cv_train_and_test_data(n_splits=n_splits, balance=balance):

            split_id += 1

            tx, ty = aug.augment_data(
                augmentation_dir=augmentation_dir + str(n_splits) + "/" + str(split_id),
                original_paths=X_train,
                original_labels=y_train,
                resize_size=224,
                n_augmentations=n_augmentations)

            X_train = np.concatenate((X_train, np.asarray(tx)), axis=0)
            y_train = np.concatenate((y_train, np.asarray(ty)), axis=0)

            yield X_train, y_train, X_test, y_test


def test(directory='./D3.json', balance=False, folds=5):
    p = Pre().loadData(directory)
    it = p.get_cv_train_and_test_data(folds, balance)
    for f in range(folds):
        x, y, z, w = next(it)
        assert(len(set(y)) == p.getClassCount())
