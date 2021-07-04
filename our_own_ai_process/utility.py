import random


def train_test_split(df, test_size):

    # if the test_size is proportion to df
    if isinstance(test_size, float):
        test_size = round(test_size * len(df)) #eg: test_size = 0.2 -> 20% x df

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df