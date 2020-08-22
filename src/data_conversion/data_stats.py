import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_filtering import get_refined_filtered_data


def get_data_stats(path):
    """The functions gives stats about the Mycroft Data Format like the size, unique labels

    :param path: Path of the csv file which is extracted from the web tables
    :return: Stats about the data size and unique labels
    """
    input_data = pd.read_csv(path, names=["csv_data"])
    filtered_data = get_refined_filtered_data(input_data)

    data = pd.DataFrame(filtered_data['data'])
    label_counts = filtered_data['label'].value_counts()
    print("The number of rows in the data after filtering and refining labels is {}".format(len(data)))

    # Plot of the final labels and its count used for feature extraction and modelling
    plt.figure(figsize=(12, 10))
    chart = sns.barplot(x=label_counts.index, y=label_counts)
    chart.set_xticklabels(
        chart.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        fontweight='light'
    )
    plt.ylabel('Label Count')
    plt.xlabel('Labels')
    plt.savefig('../../resources/images/label_count.png')
    plt.show()

    print("Number of unique labels after filtering and refining: {}".format(len(filtered_data['label'].unique())))


# Can be used when you want to check the stats of a file directly
get_data_stats("../../resources/output/mycroft_10000_tables.csv")
