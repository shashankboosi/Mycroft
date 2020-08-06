import ast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    print("The number of rows in the data before label filtering is {}".format(len(transform_data)))

    label_categories = len(transform_data['label'].unique())
    print("Number of unique labels before filtering: {}".format(label_categories))

    label_counts = transform_data['label'].value_counts()
    filtered_labels = list(filter(lambda x: x >= 1, (label_counts.values / max(label_counts)) * 100))
    unwanted_labels = label_counts.index[len(filtered_labels):].values
    filtered_data = transform_data[~transform_data['label'].isin(unwanted_labels)]

    data = pd.DataFrame(filtered_data['data'])
    print("The number of rows in the data after filtering labels is {}".format(len(data)))

    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x=label_counts.index[:len(filtered_labels)], y=label_counts[:len(filtered_labels)])
    chart.set_xticklabels(
        chart.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        fontweight='light'
    )
    plt.ylabel('Count')
    plt.xlabel('Labels')
    plt.show()

    print("Number of unique labels after filtering: {}".format(len(filtered_data['label'].unique())))


# Can be used when you want to check the stats of a file directly
# get_data_stats("../../resources/output/mycroft_10000_tables.csv")
