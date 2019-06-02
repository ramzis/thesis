import os
import sys
import pprint
import scandir
import jsonpickle
import numpy as np


class Descriptor(object):
    """
    Defines a dataset descriptor.
    name    - unique identifier string.
    rootdir - root directory, used to populate dataset.
              Expected format at location:

              - Root:
                - Class:
                    * Image
                    * ...
                - ...
    """
    def __init__(self, name, rootdir=str(), data_thresh=4, verbose=False):
        """
        If a directory is provided, initialises data as:
        data = {
            "name": str(),
            "rootdir": str(),
            "length": int()
            "classes": [
                {
                    "paths" : [],
                    "labels": []
                }
            ]
        }
        Otherwise creates an empty descriptor.
        """
        self.data = {}
        self.data["name"] = name
        self.data["rootdir"] = rootdir
        self.data["length"] = int()
        self.data["classes"] = []
        # Populate descriptor if directory is given
        if rootdir:
            self.populate(rootdir, data_thresh, verbose)


    def exportFile(self, directory):
        """
        Saves the dataset descriptor in JSON format.
        Uses the file name defined in data.
        directory - file save path directory, i.e.: "./dir/desc_0.json"
        """
        try:
            json_ = jsonpickle.encode(self.data)
            #  directory = os.path.join(directory, self.data["name"] + ".json")
            with open(directory, 'w') as f:
                f.write(json_)
            return True
        except IOError as err:
            print("I/O error: {0}".format(err))
            return False
        except ValueError:
            print("Could not encode class to json.")
            return False
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
            return False

    def importFile(self, directory):
        """
        Loads a dataset descriptor from a JSON file.
        directory - file load path directory, i.e.: "./dir/desc_0.json"
        """
        try:
            with open(directory, 'r') as f:
                json_ = f.read()
                self.data = jsonpickle.decode(json_)
            return True
        except IOError as err:
            print("I/O error: {0}".format(err))
            return False
        except ValueError:
            print("Could not decode json data.")
            return False
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
            return False

    def populate(self, rootdir, data_thresh=0, verbose=False):
        """
        Initialises a dataset descriptor by finding data in a given root directory.
        rootdir -  root directory, used to populate dataset
        data_thresh - minimum examples required per class
        verbose - prints detailed initialisation information
        """
        print("Finding image data paths and labels...")

        # Find all class folders
        class_folders = [entry.path for entry in scandir.scandir(rootdir)
                         if entry.is_dir()]
        class_folders.sort()

        # Reset class counter
        self.data["length"] = 0

        # Reset data classes
        self.data["classes"] = []

        # Loop through class folders
        for class_folder in class_folders:

            # List of class image paths
            X = []
            # List of class labels
            Y = []

            # Find image files in each folder
            files = scandir.scandir(class_folder)

            # Create a new list of valid class images
            class_images = [
                i.path for i in files
                if os.path.splitext(i.path)[1] in ['.jpg', '.jpeg']]

            # Ignore folders lacking data
            if len(class_images) >= data_thresh:

                # Count classes in dataset
                self.data["length"] += 1
                # Add class paths and labels
                # The labels are the class folder name
                X = class_images
                Y = len(class_images) * [os.path.basename(class_folder)]
                self.data["classes"].append({"paths": X, "labels": Y})

        if verbose:
            print("Using %d/%d classes."
                  % (self.data["length"], len(class_folders)))

        print("Completed.")

    def populate_paths_labels(self, paths, labels, data_thresh=0, verbose=False):
        """
        Initialises a dataset descriptor from a given list of paths and labels.
        paths - data paths
        labels - data labels
        verbose - prints detailed initialisation information
        """
        if verbose:
            print("Creating dataset from data paths and labels...")

        # Reset class counter
        self.data["length"] = 0

        # Reset data classes
        self.data["classes"] = []

        # List of class image paths
        X = []
        # List of class labels
        Y = []

        # Set first label
        curr_label = labels[0] if len(labels) > 0 else None

        for idx, (path, label) in enumerate(zip(paths, labels)):

            if label == curr_label and idx != len(labels) - 1:

                X.append(path)
                Y.append(label)

            else:
                # Ignore folders lacking data
                if len(X) >= data_thresh:
                    # Count classes in dataset
                    self.data["length"] += 1
                    # Add class paths and labels
                    self.data["classes"].append({"paths": X, "labels": Y})

                # Reset class counter
                X = []
                X.append(path)
                Y = []
                Y.append(label)
                # Change label
                curr_label = label

        if verbose:
            print("Completed.")

    def get_data_paths_and_labels(self, numbered_labels=True):
        '''
        Returns a list of all data paths in the descriptor and a second list
        containing the corrseponding class labels.
        numbered_labels - whether to return original labels or to index them from 0
        '''
        X = []
        Y = []

        for idx, c in enumerate(self.data["classes"]):
            X += c["paths"]
            if numbered_labels:
                Y += [idx] * len(c["paths"])
            else:
                Y += c["labels"]

        return X, Y


    def get_top_n_data_paths_and_labels(self, n_classes=2, numbered_labels=True):
        '''
        Returns a list of class labels and paths of the n most populated classes
        in the descriptor.
        n_classes - the number of classes to return
        numbered_labels - whether to return original labels or to index them from 0
        '''
        X = []
        Y = []

        count_per_class = np.array([len(c["paths"]) for c in self.data["classes"]])
        if(len(count_per_class) < n_classes):
            print("Warning: trying to get top {} classes from {} available! \
                Using {}.".format(n_classes, len(count_per_class), len(count_per_class)))
            n_classes = len(count_per_class)

        # Get the top n-1 class indices
        top_n_1_classes = np.argsort(-count_per_class)[:n_classes-1].tolist()
        # Create a list of labels for each class
        if numbered_labels:
            class_labels = range(n_classes)
        else:
            class_labels = [self.data["classes"][c]["labels"][0] for c in top_n_1_classes]

        for idx, c in enumerate(self.data["classes"]):
            # Find the class label
            i = top_n_1_classes.index(idx) if idx in top_n_1_classes else -1
            label = class_labels[i]
            X += c["paths"]
            Y += [label] * len(c["labels"])

        return X, Y

    def balance(self):
        """
        Balances data so each class has the same number of images.
        The number is chosen to be the maximum of all classes.
        """
        # Find max no of examples per class
        m = max([len(c["paths"]) for c in self.data["classes"]])
        # Loop through each class
        for class_idx, c in enumerate(self.data["classes"]):
            # Count the no of examples per class
            l = len(c["paths"])
            # Ignore empty classes
            if l <= 0:
                continue
            # If there are less examples than the max
            if l < m:
                # Find how many samples need to be added
                d = m - l
                # Find sample indices
                if d <= l:
                    # Sample d without replacement
                    idxs = np.random.choice(l, d, replace=False).tolist()
                else:
                    # Sample l without replacement
                    idxs1 = np.random.choice(l, l, replace=False).tolist()
                    # and d-l with replacement
                    idxs2 = np.random.choice(l, d - l, replace=True).tolist()
                    idxs = idxs1 + idxs2
                # Add samples to data descriptor
                ps = [c["paths"][i] for i in idxs]
                ls = [c["labels"][0]] * d
                self.data["classes"][class_idx]["paths"] += ps
                self.data["classes"][class_idx]["labels"] += ls


def test():
    dataset_ = Descriptor(name="dataset1", rootdir="./raw/", verbose=True)
    dataset_.exportFile("./")
    pp = pprint.PrettyPrinter(indent=4)
    if dataset_.importFile('./dataset1.json'):
        pp.pprint(dataset_.data)
        print ""

    X, Y = dataset_.get_data_paths_and_labels()
    pp.pprint(X)
    pp.pprint(Y)


def test2():
    dataset_ = Descriptor(name="dataset1")
    pp = pprint.PrettyPrinter(indent=4)
    if dataset_.importFile("dataset1.json"):
        print "Pre balance data:"
        pp.pprint(dataset_.data)
        dataset_.balance()
        print "Post balance data:"
        pp.pprint(dataset_.data)
        dataset_.exportFile("./")
