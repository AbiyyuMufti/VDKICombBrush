# LIBRARIES
import numpy as np
import pandas as pd
import random
from RandomForest_helper_funct import determine_type_of_feature

# 1. IF THE DATA IS ALR PURE ( ONLY 1 CLASS IN THIS DATA SET)
def check_purity(data):

    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    #unique_classes = 0 and 1 (kamm, burst) -> length = 2
    #if length only 1 -> only 1 class -> data pure -> True
    if len(unique_classes) == 1:
        return True
    else:
        return False

# 2. CLASSIFYING DATA
def classify_data(data):

    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification

# 3. POTENTIAL SPLIT FOR NODE
def get_potential_splits(data, random_subspace):

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


# 4. CALCULATE THE ENTROPY (LOWEST ENTROPY?)
def calculate_entropy(data):

    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy

# 5. CALCULATE OVERALL ENTROPY
def calculate_overall_entropy(data_below, data_above):

    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below)
                      + p_data_above * calculate_entropy(data_above))

    return overall_entropy

# 6. DETERMINE BEST SPLIT OR NODE
def determine_best_split(data, potential_splits):

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

# 7. SPLIT DATA TO DETERMINE QUESTION/ NODE
def split_data(data, split_column, split_value):

    split_column_values = data[:, split_column]

    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values >  split_value]

    return data_below, data_above

# 8. DECISION TREE
def decision_tree_algorithm(df, counter=0, min_samples = 2, max_depth = 10, random_subspace = None):

    # data preparations
    if counter == 0:          #at first, data still data frame and it needs to be converted to the numpy 2Darray (without header)
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

        # instantiate sub-tree
        feature_name= COLUMN_HEADERS[split_column]
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

# 9. PREDICTION FOR ONE EXAMPLE
def predict_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer

    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)

# 10. PREDICTION FOR ALL EXAMPLES
def decision_tree_predictions(test_df, tree):
    predictions = test_df.apply(predict_example, args=(tree,), axis=1)
    return predictions