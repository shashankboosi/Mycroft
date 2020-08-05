import argparse
import ast
import pickle
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
    parser.add_argument('--split', '-s', default=False, type=bool,
                        help="Choose if you want to split the data or not")
    parser.add_argument('--train_split', '-ts', default=0.7, type=float,
                        help="Choose the percentage of the train data split (e.g: 0.7 -> 70% train)")

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
        input_data = pd.read_csv("../resources/output/sample_with_filter.csv", names=["csv_data"])

        transform_data = (
            input_data["csv_data"]
                .apply(lambda i: [j for j in ast.literal_eval(i)])
                .apply(pd.Series)
                .rename(columns={0: "label", 1: "data"})
        )

        data = pd.DataFrame(transform_data['data'])
        labels = pd.DataFrame(transform_data['label'])

        label_categories = len(labels['label'].unique())

        if args.extract:
            X, Y = extract_features(data, labels)
            print('Features extracted')

            # Save the extracted features
            output_file(X, "../resources/output/train_data.p")
            output_file(Y, "../resources/output/test_data.p")
        else:
            # Load pre-extracted features of sample file
            with open('../resources/output/train_data.p', 'rb') as f:
                X = pickle.load(f)

            with open('../resources/output/test_data.p', 'rb') as f:
                Y = pickle.load(f)
    else:
        sys.exit("Choose the appropriate arguments for the input data")

    train_val_predict_model(X, Y, args.input_data, args.train_split, args.split, label_categories)
