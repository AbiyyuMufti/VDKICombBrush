from our_own_ai_process.ai_process import AiProcess
from our_own_ai_process.utility import check_purity, classify_data, \
    split_data, determine_best_split, classify_example
import pprint
import pandas as pd
import numpy as np


def get_potential_splits(data):
    """
    :param data: training data
    :return: list of potential split values for each column or feature
    """

    # make a empty dictionary for potential split value in each feature
    potential_splits = {}

    # extract number of columns in data frame
    _, n_columns = data.shape

    # get unique values in each column / feature
    # excluding the last column which is the label
    for column_index in range(n_columns - 1):
        # make empty array for each column, here will be fulled with split values
        potential_splits[column_index] = []

        # extract all values for particular feature or column index
        # some data might have same value in that feature -> count it as one data
        values = data[:, column_index]
        unique_values = np.unique(values)

        # for-loop going in each value in that column
        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]

                # best value for splitting is the middle between each two data
                potential_split = (current_value + previous_value) / 2
                potential_splits[column_index].append(potential_split)
    return potential_splits


def decision_tree_algorithm(df, counter=0, max_depth=10, min_samples=2):
    """
    the main algorithm for decision tree
    :param df: panda data frame or the training data
    :param counter: counter for recursive tracking (the depth of the tree)
    :param max_depth: maximum possible depth of splits
    :param min_samples: minimal samples or data needed to run the splitting process
    :return: tree object
    """
    # data preparations:
    # delete the header or the column name from the data frame
    # the other functions will not work if the header is still attached in data frame
    if counter == 0:
        # save the column name for the nodes in the tree
        # global variable is needed so it can still be called during recursive part (when counter >0)
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df

    # base cases (stop condition so recursive not go infinitive):
    # if the data set already pure -> only contains one class -> data can be classified
    # if the counter reach the max depth of tree -> classification through which class has more member
    # if the samples or the dataset too small (< min_samples)
    if (check_purity(data)) or (counter == max_depth) or (len(data) < min_samples):
        classification = classify_data(data)
        return classification

    # recursive part:
    # after data preparation (counter = 0), counter will gets bigger until it reaches max_depth
    else:
        counter += 1

        # first getting all the potential split value for each feature or column
        # determine which feature and split value has the lowest overall entropy
        # lowest overall entropy means the feature and split value
        # splitting the data based on the best feature and split value
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        # (sub) tree making based on best feature and best split value to split
        # question asks if a data with certain feature and value is <= best split value
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, round(split_value, 2))
        sub_tree = {question: []}

        # answer is yes or no
        # yes means data are <= split value -> data_below, and vice versa for no answer
        # yes answer will produce classification because the data are pure
        # no answer will go into recursive part again and again till all classified or max dept reached
        yes_answer = decision_tree_algorithm(data_below, counter, max_depth, min_samples)
        no_answer = decision_tree_algorithm(data_above, counter, max_depth, min_samples)

        # to set answer or class for an end of a split result
        # so the same class name not appeared in twice for yes_answer and no_answer
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


class OurDecisionTree(AiProcess):
    def __init__(self, depth=10):
        super().__init__()
        self.tree = None
        self.depth = depth

    def train(self, **kwargs):
        self.tree = decision_tree_algorithm(self.train_data, max_depth=self.depth)

    def predict(self, test=None, **kwargs):
        if test is not None:
            self.test_data = test
        try:
            self.prediction = self.test_data.apply(classify_example, axis=1, args=(self.tree,)).tolist()
            self.test_labels = self.test_data.labels
            return self.prediction
        except AttributeError as e:
            print("Please train your tree first!", e)

    def plot_tree(self):
        pprint.pprint(self.tree, width=50)


if __name__ == '__main__':
    df = pd.read_csv(r"D:\VDKICombBrush\ImagesFeatures.csv")
    ODT = OurDecisionTree()
    ODT.fit(df, 0.1)
    ODT.train()
    ODT.predict()
    ODT.plot_tree()
    print(*ODT.review())