import random

import numpy as np


def train_test_split(df, test_size):

    if df.columns[-1] != 'labels':
        df["labels"] = df.iloc[:, -1]
        df = df.drop(df.iloc[:, -2:-1], axis=1)

    # if the test_size is proportion to df
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))  # eg: test_size = 0.2 -> 20% x df

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


def check_purity(data):  # IF THE DATA IS ALR PURE ( ONLY 1 CLASS IN THIS DATA SET)

    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    # unique_classes = 0 and 1 (kamm, burst) -> length = 2
    # if length only 1 -> only 1 class -> data pure -> True
    if len(unique_classes) == 1:
        return True
    else:
        return False


def classify_data(data):

    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


def split_data(data, split_column, split_value):  # SPLIT DATA TO DETERMINE QUESTION/ NODE
    split_column_values = data[:, split_column]

    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values > split_value]

    return data_below, data_above


def calculate_entropy(data):

    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def calculate_overall_entropy(data_below, data_above):  # CALCULATE OVERALL ENTROPY FOR NODE
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))

    return overall_entropy


def determine_best_split(data, potential_splits):  # DETERMINE BEST SPLIT OR NODE
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value


def classify_example(example, tree):
    """
    predict the test subject category
    :param example: test subject
    :param tree: the trained tree
    :return:
    """
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split()

    # ask question
    if example[feature_name] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer

    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)
