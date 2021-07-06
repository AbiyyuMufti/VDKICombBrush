import random

import numpy as np


def train_test_split(df, test_size):
    """
    :param df: dataframe to be split to train and test dataset
    :param test_size: ratio for making the test dataset
    :return: train and test dataset
    """

    # check if the last column (label-column) in the dataframe has name "labels"
    # create and add column labels and drop the old label-column
    if df.columns[-1] != 'labels':
        df["labels"] = df.iloc[:, -1]
        df = df.drop(df.iloc[:, -2:-1], axis=1)

    # if the test_size is given as float-type, e.g.: 0.2 instead of 20 (20%)
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    # make list of all of indices from dataframe
    indices = df.index.tolist()

    # make k-new random indices based on all of indices
    test_indices = random.sample(population=indices, k=test_size)

    # make new test dataset from the initial dataframe based on the test_indices
    # make new train dataset by dropping or excluding the test_indices
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


def check_purity(data):
    """
    see if the label_column of dataset consists only one class or label 0 or 1 (brush or comb)
    :param data: dataset
    :return: boolean (true or false) whether the label_column of dataset consists only one class
    """

    # get the last column which is label_column
    # get the class or label type or name (0 or 1)
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    # unique_classes = 0 and 1 (comb, brush) -> length = 2
    # if length only 1 -> only 1 class -> data pure -> True
    if len(unique_classes) == 1:
        return True
    else:
        return False


def classify_data(data):
    """
    classify data even the data not yet pure
    :param data: dataset
    :return: class or labels 0 or 1 (comb or brush)
    """

    # get the last column from dataset which is label_column
    # get classes or labels in the dataset and count how often the classes appears
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    # to know which class appears most often by using argmax(), named it in variable index
    # using index to know which class or labels is that.
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


def split_data(data, split_column, split_value):
    """
    splitting or grouping data based on the split_value
    :param data: training data or dataset from training data
    :param split_column: column or feature, in which the data of that column want to be splitted
    :param split_value: boundary value for splitting
    :return: data below and above based on boundary (split_value)
    """

    # get all data of the particular feature or column
    split_column_values = data[:, split_column]

    # split the data set with split_value as boundary
    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values > split_value]

    return data_below, data_above


def calculate_entropy(data):
    """
    get the entropy value based on the probability -> small entropy is good for splitting or node
    :param data: training data or dataset from training data
    :return: entropy value
    """

    # get the label column and counts how many data have the label 0 or 1 (comb or brush)
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    # calculate entropy based on probability for comb and brush
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def calculate_overall_entropy(data_below, data_above):
    """
    :param data_below
    :param data_above
    :return: overall entropy based on a particular split value and entropy for data above and below that split
    """

    # calculate probability or comparison between data below and above the split value
    # first calculate how many data in data_below and above, then calculate the probability
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))

    return overall_entropy


def determine_best_split(data, potential_splits):
    """
    :param data: training data
    :param potential_splits: dictionary or list of potential split value for each feature
    :return: best feature and its split value based on lowest entropy and lowest overall entropy
    """

    # set the first overall entropy value
    # the value will update itself based on the function calculate_overall_entropy
    overall_entropy = 99

    # for-loop: going through for each column or feature
    for column_index in potential_splits:

        # 2nd for-loop going through for each values in that column or feature
        for value in potential_splits[column_index]:

            # split (set of) training data based on the particular feature and split_value
            # then, calculate the overall entropy for that scenario
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            # seek the lowest overall entropy
            # the feature and split value with lowest overall entropy becomes the node in the tree
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


def classify_example(example, tree):
    """
    predict the test subject category
    :param example: test subject
    :param tree: tree created from training dataset
    :return: predicted class 0 or 1 (comb or brush)
    """

    # make a string variable in form of a list from the tree
    question = list(tree.keys())[0]

    # splitting the question into the column/ feature, operator (<=), and value
    # those variables are used to get the value of certain feature from the test data and compare it
    feature_name, comparison_operator, value = question.split()

    # Boolean-type for variable answer
    # change the type of variable value to float, so the value can be compared with value from test data
    if example[feature_name] <= float(value):
        # True -> value from test data <= (split) value from tree; vice versa for False (else-case)
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # base case:
    # if the answer is not a dictionary (contains only one object) -> return answer or the class
    if not isinstance(answer, dict):
        return answer

    # recursive part:
    # the answer is still dictionary -> becomes new (residual) tree
    # the first question of this (residual) tree is asked or compared to the next feature and value of test data
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)
