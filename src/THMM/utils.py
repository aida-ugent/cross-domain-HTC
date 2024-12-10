from networkx import DiGraph

import pickle
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = 'ROOT'


class BinarizerBase:
    """
    Base class for binarizer.
    """

    def __init__(self):
        self.mlb = None

    def fit(self, labels):
        """
        fit the binarizer, to be implemented in the child.
        :param labels:
        :return:
        """
        raise NotImplementedError("fit method needs to be implemented in child class.")

    def transform(self, y):
        """
        transform the list of labels into binary values, to be implemented in the child.
        :param y: list, String
        :return: numpy array.
        """
        raise NotImplementedError("transform method needs to be implemented in child class.")

    def fit_transform(self, y):
        """
        fit the binarizer and transform the list of labels into binary values, to be implemented in the child..
        :param y: list, String
        :return: numpy array.
        """
        self.fit(y)
        return self.transform(y)

    def get_sklearn_mlb_from_pred(self, y, pred_prob=0.5, prob_value=False):
        """
        Get the binary values for prediction probability from the binarizer.
        :param y: list, String
        :return: numpy array.
        """
        raise NotImplementedError("this method needs to be implemented in child class.")



class BinarizerTHMM_TMM(BinarizerBase):
    """
    Binarizer for multi-task hierarchical classifier. It wraps the sklearn binarizer.
    """

    def __init__(self):
        super(BinarizerTHMM_TMM, self).__init__()

    def fit(self, y):
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(y)

    def transform(self, y):
        if not self.mlb:
            raise Exception("binarizer not trained")
        y = self.mlb.transform(y)
        print(y.shape)
        y = y.astype(int)  # Ensure y is of integer type

        # Create y_binary of shape (n_samples, n_classes, 2)
        y_binary = np.stack([y, 1 - y], axis=2)

        return y_binary  # Shape: (n_samples, n_classes, 2)


    def get_sklearn_mlb_from_pred(self, y, pred_prob=0.5, prob_value=False):
        predictions = list()
        y = [list(item) for item in y]
        for index in range(len(y[0])):
            instance_list = list()
            for class_index in range(len(self.mlb.classes_)):
                if prob_value:
                    instance_list.append(y[class_index][index][0])
                else:
                    if y[class_index][index][0] > pred_prob:
                        instance_list.append(self.mlb.classes_[class_index])
            predictions.append(instance_list)
        if prob_value:
            return np.asarray([np.asarray(row) for row in predictions])
        else:
            return self.mlb.transform(predictions)

    def inverse_transform(self, y, pred_prob=0.5):
        predictions = list()
        y = [list(item) for item in y]
        for index in range(len(y[0])):
            instance_list = list()
            for class_index in range(len(self.mlb.classes_)):
                if y[class_index][index][0] > pred_prob:
                    instance_list.append(self.mlb.classes_[class_index])
            predictions.append(instance_list)
        return predictions



def extend_hierarchy(hierarchy, y_labs, level=3):
    for samples_t in y_labs:
        if not isinstance(samples_t, list):
            samples = [samples_t]
        else:
            samples = samples_t
        if level == 3:
            for lab in samples:
                par_1 = lab[0]
                par_2 = lab[:3]
                child = lab[:]

                if par_1 not in hierarchy[ROOT]:
                    hierarchy[ROOT].append(par_1)
                if par_1 not in hierarchy:
                    hierarchy[par_1] = [par_2]
                else:
                    if par_2 not in hierarchy[par_1]:
                        hierarchy[par_1].append(par_2)
                if par_2 not in hierarchy:
                    hierarchy[par_2] = [child]
                else:
                    if child not in hierarchy[par_2]:
                        hierarchy[par_2].append(child)
        elif level == 2:
            for lab in samples:
                par_1 = lab[0]
                child = lab[:]
                if par_1 not in hierarchy[ROOT]:
                    hierarchy[ROOT].append(par_1)
                if par_1 not in hierarchy:
                    hierarchy[par_1] = [child]
                else:
                    if child not in hierarchy[par_1]:
                        hierarchy[par_1].append(child)
    return hierarchy


def build_hierarchy(issues, level=3):
    hierarchy = {ROOT: []}
    if level == 3:
        for i in issues:
            par_1 = i[0]
            par_2 = i[:3]
            child = i[:]

            if par_1 not in hierarchy[ROOT]:
                hierarchy[ROOT].append(par_1)
            if par_1 not in hierarchy:
                hierarchy[par_1] = [par_2]
            else:
                if par_2 not in hierarchy[par_1]:
                    hierarchy[par_1].append(par_2)
            if par_2 not in hierarchy:
                hierarchy[par_2] = [child]
            else:
                hierarchy[par_2].append(child)
    elif level == 2:
        for i in issues:
            par_1 = i[0]
            child = i[:]
            if par_1 not in hierarchy[ROOT]:
                hierarchy[ROOT].append(par_1)
            if par_1 not in hierarchy:
                hierarchy[par_1] = [child]
            else:
                hierarchy[par_1].append(child)
    return hierarchy


def create_hierarchical_tree(labels, level=3):
    """
    create the graph g, for the labels.
    :param labels:
    :return:
    """
    hierarchy_f = build_hierarchy([tj for tk in labels for tj in tk], level=level)
    class_hierarchy = extend_hierarchy(hierarchy_f, labels, level=level) # TODO why do this?
    # save
    with open('class_hierarchy.pkl', 'wb') as f:
        pickle.dump(class_hierarchy, f)
    g = DiGraph(class_hierarchy)
    # g = DiGraph(hierarchy_f)
    return g


