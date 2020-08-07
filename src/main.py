import argparse
import ast
import os
import pickle
import random
import sys
import warnings

import numpy as np
import pandas as pd
from features.build_features import build_features
from helpers.utils import output_file
from models.train_predict_model import train_val_predict_model

warnings.filterwarnings("ignore")
sys.path.append("..")


def extract_features(data, labels):
    print('Extracting features.')
    return build_features(data), labels.values.flatten()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mycroft Semantic Type detection')
    parser.add_argument('--input_data', '-i', default='sherlock', type=str,
                        help="Choose the type of data (options: sherlock, mycroft)")
    parser.add_argument('--extract', '-e', default=False, type=bool,
                        help="Choose if you want to generate features or not")
    parser.add_argument('--split', '-spt', default=False, type=bool,
                        help="Choose if you want to split the data or not")
    parser.add_argument('--train_split', '-ts', default=0.7, type=float,
                        help="Choose the percentage of the train data split (e.g: 0.7 -> 70% train)")
    parser.add_argument('--no_of_tables', '-num', default=5000, type=int,
                        help="Choose the files with number of tables that is required for processing (options: 5000, "
                             "10000, 100000, 500000)")
    parser.add_argument('--sample', '-smp', default=False, type=bool,
                        help="Choose if you want to use sample or not")

    args = parser.parse_args()

    if args.input_data == 'sherlock':
        data = pd.read_csv('../resources/data/sherlock/raw/test_values.csv', sep=',', index_col=0, header=None)
        labels = pd.read_csv('../resources/data/sherlock/raw/test_labels.csv', sep=',', index_col=0, header=None)

        data.head()
        labels.head()

        label_categories = len(np.unique(np.concatenate(labels.values)))

        if args.extract:
            X, Y = extract_features(data, labels)
            print('Features extracted')
        else:
            # Load pre-extracted features of sample file
            with open('../resources/data/sherlock/processed/X_train.data', 'rb') as f:
                X = pickle.load(f)

            with open('../resources/data/sherlock/processed/y_train.data', 'rb') as f:
                Y = pickle.load(f)

    elif args.input_data == 'mycroft':

        if not args.sample:
            if not os.path.exists(os.path.normpath(
                    os.path.join(os.path.dirname(__file__), '..', 'resources', 'output',
                                 'mycroft_{}_tables.csv'.format(args.no_of_tables)))):
                exit("Please generate the mycroft data with no of tables: {} :)".format(args.no_of_tables))

        if args.sample:
            input_data = pd.read_csv("../resources/output/sample_with_filter.csv".format(args.no_of_tables),
                                     names=["csv_data"])
        else:
            input_data = pd.read_csv("../resources/output/mycroft_{}_tables.csv".format(args.no_of_tables),
                                     names=["csv_data"])

        transform_data = (
            input_data["csv_data"]
                .apply(lambda i: [j for j in ast.literal_eval(i)])
                .apply(pd.Series)
                .rename(columns={0: "label", 1: "data"})
        )

        # Remove the data of the labels which contains more than 5 percent of the total transformed data
        label_counts = transform_data['label'].value_counts()
        limit = int(len(transform_data) * 0.05)
        labels_with_excess_data = label_counts.index[
                                  :len(list(filter(lambda x: x >= limit, label_counts.values)))].values

        refined_data = transform_data.copy()
        for i in range(len(labels_with_excess_data)):
            indexes_to_drop = random.sample(
                list(transform_data.groupby("label").groups[labels_with_excess_data[i]].values),
                label_counts[labels_with_excess_data[i]] - limit)
            refined_data.drop(transform_data.index[indexes_to_drop], inplace=True)
        refined_data.reset_index(drop=True, inplace=True)

        # Remove the data of the labels which contains less than 10 percent of the refined data
        refined_label_counts = refined_data['label'].value_counts()
        filtered_labels = list(
            filter(lambda x: x >= 10, (refined_label_counts.values / max(refined_label_counts)) * 100)
        )
        unwanted_labels = refined_label_counts.index[len(filtered_labels):].values
        filtered_data = refined_data[~refined_data['label'].isin(unwanted_labels)]

        data = pd.DataFrame(filtered_data['data'])
        labels = pd.DataFrame(filtered_data['label'])

        label_categories = len(labels['label'].unique())

        if args.sample:
            feature_train_path = "../resources/output/features/sample_train_data.p"
            feature_test_path = "../resources/output/features/sample_test_data.p"
        else:
            feature_train_path = "../resources/output/features/train_data_{}.p".format(args.no_of_tables)
            feature_test_path = "../resources/output/features/test_data_{}.p".format(args.no_of_tables)

        if args.extract:
            X, Y = extract_features(data, labels)
            print('Features extracted')

            # Save the extracted features
            output_file(X, feature_train_path)
            output_file(Y, feature_test_path)
        else:

            # Load pre-extracted features of sample file
            with open(feature_train_path, 'rb') as f:
                X = pickle.load(f)

            with open(feature_test_path, 'rb') as f:
                Y = pickle.load(f)
    else:
        sys.exit("Choose the appropriate arguments for the input data")

    train_val_predict_model(X, Y, args.input_data, args.train_split, args.split, label_categories)
