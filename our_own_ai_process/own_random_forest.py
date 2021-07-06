import pandas as pd
import numpy as np
import random
from our_own_ai_process.utility import check_purity,\
    classify_data, split_data, determine_best_split, classify_example
from our_own_ai_process.ai_process import AiProcess


def get_potential_splits(data, random_subspace):
    """
    :param data: training data
    :param random_subspace: introduce randomness in the set of features to get different decision tree algorithm
    :return: list of potential split values for each column or feature
    """

    # make a empty dictionary for potential split value in each feature
    potential_splits = {}

    # extract number of columns in data frame
    # excluding the last column which is the label
    _, n_columns = data.shape

    # list of number of features or columns
    column_indices = list(range(n_columns - 1))

    # if random_subspace not None and smaller than sum of features or columns:
    # the second condition is needed, so sum of features used in one tree is still under or same as sum of all features
    if random_subspace and random_subspace <= len(column_indices):
        # select k-random features from sum of all features in training dataset
        column_indices = random.sample(population=column_indices, k=random_subspace)

    # get unique values in that random columns / features
    for column_index in column_indices:
        # make empty array for k-random features or column, here will be fulled with split values
        potential_splits[column_index] = []

        # extract all values for that feature or column index
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


def decision_tree_algorithm(df, counter=0, max_depth=10, min_samples=2, random_subspace=None):
    """
    :param df: panda data frame or the training data
    :param counter: counter for recursive tracking (the depth of the tree)
    :param max_depth: maximum possible depth of splits
    :param min_samples: minimal samples or data needed to run the splitting process
    :param random_subspace: introduce randomness in the set of features to get different decision tree algorithm
    :return:
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

    # recursive part
    # after data preparation (counter = 0), counter will gets bigger until it reaches max_depth
    else:
        counter += 1

        # first getting all the potential split value for each feature or column
        # determine which feature and split value has the lowest overall entropy
        # lowest overall entropy means the feature and split value
        # splitting the data based on the best feature and split value
        potential_splits = get_potential_splits(data, random_subspace)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        # (sub) tree making based on best feature and best split value to split
        # question asks if a data with certain feature and value is <= best split value
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification

        # instantiate sub-tree and determine the question
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}

        # answer is yes or no
        # yes means data are <= split value -> data_below, and vice versa for no answer
        # yes answer will produce classification because the data are pure
        # no answer will go into recursive part again and again till all classified or max dept reached
        yes_answer = decision_tree_algorithm(data_below, counter, max_depth, min_samples, random_subspace)
        no_answer = decision_tree_algorithm(data_above, counter, max_depth, min_samples, random_subspace)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


def bootstrapping(train_df, n_bootstrap):
    """
    approach to introduce randomness to new data set
    :param train_df: training dataset
    :param n_bootstrap: how big the new dataset to be made from training dataset
    :return: new training dataset with some duplication of the data itself
    """

    # make random indices for data-duplicating from training dataset
    # index is must be in range of 0 (first data/ row) to last data/ row
    # size of the bootstrap is the size of the new training dataset
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]

    return df_bootstrapped


def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    """
    the main algorithm for random forest to generate forest object
    :param train_df: training dataset
    :param n_trees: how many trees already created
    :param n_bootstrap: how big the new dataset to be made from training dataset
    :param n_features: how many random features used in each tree
    :param dt_max_depth: max depth of decision tree
    :return: list of all (decision) trees
    """

    forest = []

    # for-loop to make different decision tree based on different training dataset and features
    for i in range(n_trees):
        # new dataset to be trained to make one decision tree
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)

        # make one tree based on the new (bootstrapped) data set and n-features
        # tree will be append in forest list
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)

    return forest


def decision_tree_predictions(test_df, tree):
    """
    execute prediction based on decision tree method inside the random forest
    :param test_df: test dataset or subhect
    :param tree: the trained tree from forest
    :return: predicted class 0 or 1 (brush or comb) of test dataset based on particular tree
    """

    predictions = test_df.apply(classify_example, args=(tree,), axis=1)
    return predictions


def random_forest_predictions(test_df, forest):
    """
    predict the test subject category based on the random forest methode
    :param test_df: test dataset or subject
    :param forest: forest consist of trees been made based on new (bootstrapped) dataset and n-features
    :return: predicted class 0 or 1 (brush or comb) of test dataset based on all tree / forest
    """

    # make new dictionary for prediction based on each tree
    df_predictions = {}

    # for-loop to get all prediction from each tree
    for i in range(len(forest)):

        # column name based on current tree
        column_name = "tree_{}".format(i)

        # calling function decision_tree_prediction for current tree
        predictions = decision_tree_predictions(test_df, tree=forest[i])

        # fulling the predicted class for each test data or subject based on current tree
        df_predictions[column_name] = predictions

    # make new dataframe from all predictions
    # select the predicted class for each test data based on which class predicted the most
    df_predictions = pd.DataFrame(df_predictions)
    random_forest_prediction = df_predictions.mode(axis=1)[0]

    return random_forest_prediction


class OurRandomForrest(AiProcess):
    def __init__(self):
        super().__init__()
        self.forest = None

    def train(self, **kwargs):
        self.forest = random_forest_algorithm(self.train_data, n_trees=5, n_bootstrap=800, n_features=10,
                                              dt_max_depth=10)

    def predict(self, test=None, **kwargs):
        try:
            predictions = random_forest_predictions(self.test_data, self.forest)
            self.prediction = predictions.tolist()
            self.test_labels = self.test_data.labels
            return self.prediction
        except AttributeError as e:
            print("Please train your forest first!", e)


if __name__ == '__main__':
    df = pd.read_csv(r"D:\VDKICombBrush\ImagesFeatures.csv")
    ORF = OurRandomForrest()
    ORF.fit(df, 0.1)
    ORF.train()
    ORF.predict()
    print(*ORF.review())
