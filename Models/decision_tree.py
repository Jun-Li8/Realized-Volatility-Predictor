import numpy as np
import math
import statistics
from scipy import stats

class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.ndim == 1:
            if len(features) != len(self.attribute_names):
                raise ValueError(
                    "Number of features and number of attribute names must match!\n Attributes are: %s\n Features are: %s" % (self.attribute_names, features)
                )
        elif features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!\n Attributes are: %s\n Features are: %s" % (self.attribute_names, features)
            )

    def tree_attributes(self, node):
        if len(node.branches) == 0:
            return 1, 1
        elif len(node.branches) == 1:
            down, down_depth = self.tree_attributes(node.branches[0])
            num = down + 1
            depth = down_depth + 1
            return num, depth
        else:
            left, left_depth = self.tree_attributes(node.branches[0])
            right, right_depth = self.tree_attributes(node.branches[1])
            depth = max(left_depth, right_depth) + 1
            num = left + right + 1
            return num, depth

    def DTL(self, features, targets, attributes, default=None, v=None):
        if len(targets) == 0:
            T = Tree(default)
            return T
        elif sum(targets) == len(targets):
            T = Tree(1)
            return T
        elif sum(targets) == 0:
            T = Tree(0)
            return T
        elif len(attributes) == 0:
            T = Tree(stats.mode(targets.flatten()))
            return T
        else:
            max = 0
            i = 0
            # choosing best attribute
            for index in range(0, len(attributes) - 1):
                info = information_gain(features, index, targets)
                if info > max:
                    max = info
                    i = index

            best_index = self.attribute_names.index(attributes[i])

            T = Tree(v, attributes[i], best_index, None)
            for value in set(features[:, i]):
                # reducing features and targets to only include best
                new_features = []
                new_targets = []
                for index in range(0, len(features)):
                    if features[index,i] == value:
                        new_features.append(features[index])
                        new_targets.append(targets[index])

                new_features = np.asarray(new_features)
                new_targets = np.asarray(new_targets)
                new_features = np.delete(new_features, i, axis = 1)
                new_attributes = list.copy(attributes)
                new_attributes.remove(attributes[i])
                subtree = self.DTL(new_features, new_targets, new_attributes, stats.mode(new_targets.flatten()), value)

                T.branches.append(subtree)
            return T

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        self._check_input(features)

        self.tree = self.DTL(features, targets, self.attribute_names, stats.mode(targets.flatten()), None)

    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """

        self._check_input(features)

        targets = []

        for entry in features:
            if features.ndim == 1:
                entry = list(features)
            else:
                entry = list(entry)
            T = self.tree
            while T.branches:
                if len(T.branches) == 1:
                    T = T.branches[0]
                else:
                    if entry[T.attribute_index] == 1:
                        T = T.branches[1]
                    else:
                        T = T.branches[0]
            if isinstance(T.value, int):
                targets.append(T.value)
            else:
                targets.append(T.value[0])

        targets = np.asarray(targets)

        return targets

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def entropy_split(features, attribute_index, targets):
    attribute = []
    for row in features:
        attribute.append(row[attribute_index])

    attribute_ones = sum(attribute)
    attribute_zeros = len(attribute) - attribute_ones

    zero_ones = 0
    zero_zeros = 0
    one_ones = 0
    one_zeros = 0

    for index in range(0, len(attribute) - 1):
        if attribute[index] == 0:
            if targets[index] == 0:
                zero_zeros += 1
            else:
                zero_ones += 1
        else:
            if targets[index] == 0:
                one_zeros += 1
            else:
                one_ones += 1
    if one_ones == 0 or one_zeros == 0 or zero_ones == 0 or zero_zeros == 0:
        s = 0
        return s

    pos = attribute_ones/len(attribute) * (-one_ones/attribute_ones*math.log2(one_ones/attribute_ones) - one_zeros/attribute_ones*math.log2(one_zeros/attribute_ones))
    neg = attribute_zeros/len(attribute) * (-zero_ones/attribute_zeros*math.log2(zero_ones/attribute_zeros) - zero_zeros/attribute_zeros*math.log2(zero_zeros/attribute_zeros))

    s = pos + neg
    return s

def entropy(pos_list, neg_list):
    if pos_list == 0 and neg_list == 0:
        return 0
    elif neg_list == 0:
        pos_fraction = pos_list / (pos_list + neg_list)
        return -pos_fraction * math.log2(pos_fraction)
    elif pos_list == 0:
        neg_fraction = neg_list / (pos_list + neg_list)
        return neg_fraction * math.log2(neg_fraction)
    else:
        pos_fraction = pos_list / (pos_list + neg_list)
        neg_fraction = neg_list / (pos_list + neg_list)
        return -pos_fraction * math.log2(pos_fraction) - neg_fraction * math.log2(neg_fraction)

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """

    feature_columns = features[:, attribute_index]
    n, p, nn, np, pn, pp = 0,0,0,0,0,0

    for row in range(0, len(feature_columns)):
        if feature_columns[row] == 1:
            p += 1
            if targets[row] == 1:
                pp += 1
        else:
            n += 1
            if targets[row] == 1:
                np += 1

    np_prob = 0 if n == 0 else np/n
    pp_prob = 0 if p == 0 else pp/p

    nn_prob = 1 - np_prob
    pn_prob = 1 - pp_prob
    n_entropy = entropy(np_prob, nn_prob)
    p_entropy = entropy(pp_prob, pn_prob)

    total = n + p
    tp_prob = (np + pp) / total
    tn_prob = 1 - tp_prob
    t_entropy = entropy(tp_prob, tn_prob)

    n_frac = n / total
    p_frac = p / total

    return t_entropy - (n_frac * n_entropy + p_frac * p_entropy)

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
