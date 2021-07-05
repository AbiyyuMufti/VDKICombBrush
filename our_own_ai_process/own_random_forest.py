import pandas as pd
import numpy as np
import random
from utility import check_purity, classify_data, split_data, determine_best_split, classify_example
from ai_process import AiProcess


def get_potential_splits(data, random_subspace):
    """
    # POTENTIAL SPLIT FOR NODE Inside the forest
    :param data:
    :param random_subspace:
    :return:
    """
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns - 1))  # excluding the last column which is the label

    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_subspace)

    for column_index in column_indices:
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2

                potential_splits[column_index].append(potential_split)

    return potential_splits


def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5, random_subspace=None):
    """
    :param df:
    :param counter:
    :param min_samples:
    :param max_depth:
    :param random_subspace:
    :return:
    """
    # data preparations
    if counter == 0:  # at first, data still data frame and it needs to be converted to the numpy 2Darray (without header)
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df

    # base cases (stop condition so recursive not go infinitiv)
    if (check_purity(data)) or (counter == max_depth):
        classification = classify_data(data)
        return classification

    # recursive part
    else:
        counter += 1

        # helper functions
        potential_splits = get_potential_splits(data, random_subspace)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification

        # instantiate sub-tree and determine the question
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}

        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, max_depth, random_subspace)
        no_answer = decision_tree_algorithm(data_above, counter, max_depth, random_subspace)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


def bootstrapping(train_df, n_bootstrap):
    """
    :param train_df:
    :param n_bootstrap:
    :return:
    """
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    return df_bootstrapped


def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    """
    the main algorithm for random forest to generate forest object
    :param train_df:
    :param n_trees:
    :param n_bootstrap:
    :param n_features:
    :param dt_max_depth:
    :return:
    """
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)

    return forest


def decision_tree_predictions(test_df, tree):
    """
    execute prediction based on decision tree method inside the random forest
    :param test_df: test subject
    :param tree: the trained tree
    :return:
    """
    predictions = test_df.apply(classify_example, args=(tree,), axis=1)
    return predictions


def random_forest_predictions(test_df, forest):
    """
    predict the test subject category based on the random forest methode
    :param test_df: test subject
    :param forest: the trained forest
    :return:
    """
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_prediction = df_predictions.mode(axis=1)[0]
    return random_forest_prediction


class OurRandomForrest(AiProcess):
    def __init__(self):
        super().__init__()
        self.forest = None

    def train(self, **kwargs):
        self.forest = random_forest_algorithm(self.train_data, n_trees=4, n_bootstrap=800, n_features=2, dt_max_depth=4)

    def predict(self, test=None, **kwargs):
        try:
            predictions = random_forest_predictions(self.test_data, self.forest)
            self.prediction = predictions.tolist()
            self.test_labels = self.test_data.labels
            return self.prediction
        except AttributeError as e:
            print("Please train your forest first!", e)


if __name__ == '__main__':
    df = pd.read_csv(r"D:\VDKICombBrush\800ImagesFeatures.csv")
    ORF = OurRandomForrest()
    ORF.fit(df, 0.1)
    ORF.train()
    ORF.predict()
    print(*ORF.review())
