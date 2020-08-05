import ast

import pandas as pd


def get_data_stats(path):
    """The functions gives stats about the Mycroft Data Format like the size, unique labels

    :param path: Path of the csv file which is extracted from the web tables
    :return: Stats about the data size and unique labels
    """
    input_data = pd.read_csv(path, names=["csv_data"])

    transform_data = (
        input_data["csv_data"]
            .apply(lambda i: [j for j in ast.literal_eval(i)])
            .apply(pd.Series)
            .rename(columns={0: "label", 1: "data"})
    )

    data = pd.DataFrame(transform_data['data'])
    print("The number of rows in the data are {}".format(len(data)))

    labels = pd.DataFrame(transform_data['label'])
    label_categories = len(labels['label'].unique())
    print("Number of unique labels: {}".format(label_categories))

# Can be used when you want to check the stats of a file directly
# get_data_stats("../../resources/output/mycroft_with_filter.csv")
