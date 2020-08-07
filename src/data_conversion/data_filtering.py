import ast
import random

import pandas as pd

SEED = 6


def get_refined_filtered_data(input_data):
    transform_data = (
        input_data["csv_data"]
            .apply(lambda i: [j for j in ast.literal_eval(i)])
            .apply(pd.Series)
            .rename(columns={0: "label", 1: "data"})
    )
    print("The number of rows in the data before label filtering is {}".format(len(transform_data)))

    label_categories = len(transform_data['label'].unique())
    print("Number of unique labels before filtering: {}".format(label_categories))

    # Remove the data of the labels which contains more than 5 percent of the total transformed data
    label_counts = transform_data['label'].value_counts()
    limit = int(len(transform_data) * 0.05)
    labels_with_excess_data = label_counts.index[:len(list(filter(lambda x: x >= limit, label_counts.values)))].values
    print("The labels that have more than 5% of data are {}".format(labels_with_excess_data))

    refined_data = transform_data.copy()
    random.seed(SEED)
    for i in range(len(labels_with_excess_data)):
        indexes_to_drop = random.sample(list(transform_data.groupby("label").groups[labels_with_excess_data[i]].values),
                                        label_counts[labels_with_excess_data[i]] - limit)
        refined_data.drop(transform_data.index[indexes_to_drop], inplace=True)
    refined_data.reset_index(drop=True, inplace=True)

    # Remove the data of the labels which contains less than 10 percent of the refined data
    refined_label_counts = refined_data['label'].value_counts()
    filtered_labels = list(filter(lambda x: x >= 10, (refined_label_counts.values / max(refined_label_counts)) * 100))
    unwanted_labels = refined_label_counts.index[len(filtered_labels):].values

    return refined_data[~refined_data['label'].isin(unwanted_labels)]
