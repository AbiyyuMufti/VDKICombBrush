from ai_process import AiProcess
from utility import check_purity, classify_data, split_data, determine_best_split
import pprint
import pandas as pd
import numpy as np


def get_potential_splits(data):  # POTENTIAL SPLIT FOR NODE

    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):  # excluding the last column which is the label
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


def decision_tree_algorithm(df, counter=0, max_depth=10):  # DECISION TREE
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
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        # instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}

        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, max_depth)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree

def classify_example(example, tree):
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



'''
def calculate_accuracy(df, tree):  # ACCURACY
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = df.classification == df.labels

    accuracy = df["classification_correct"].mean()

    return accuracy
'''


class OurDecisionTree(AiProcess):
    def __init__(self):
        super().__init__()
        self.tree = None

    def predict(self, test=None):
        # self.train_data
        # self.test_data
        self.tree = decision_tree_algorithm(self.train_data)
        self.prediction = self.test_data.apply(classify_example, axis=1, args=(self.tree,)).tolist()
        self.test_labels = self.test_data.labels

    def plot_tree(self):
        pprint.pprint(self.tree)


if __name__ == '__main__':
    df = pd.read_csv(r"D:\2021\01_HSKA\SS2021\VdKI\Project\03_git\VDKICombBrush\800ImagesFeatures.csv")
    ODT = OurDecisionTree()
    ODT.fit(df, 0.1)
    ODT.predict()
    ODT.plot_tree()
    a, b, c = ODT.review()
    print("HASIL  :", a, b, c)
    print(ODT.test_data.head())
