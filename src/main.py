import ast
import pickle
import sys
import warnings
import argparse

import pandas as pd

from models.train_sherlock import train_val_model
from helpers.utils import output_file
from features.build_features import build_features

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

    args = parser.parse_args()

    if args.input_data == 'sherlock':
        data = pd.read_csv('../resources/data/sherlock/raw/test_values.csv', sep=',', index_col=0, header=None)
        labels = pd.read_csv('../resources/data/sherlock/raw/test_labels.csv', sep=',', index_col=0, header=None)

        data.head()
        labels.head()

        if args.extract:
            X_train, y_train = extract_features(data, labels)
            print('Features extracted')
        else:
            # Load pre-extracted features of sample file
            with open('../resources/data/sherlock/processed/X_train.data', 'rb') as f:
                X_train = pickle.load(f)

            with open('../resources/data/sherlock/processed/y_train.data', 'rb') as f:
                y_train = pickle.load(f)

    elif args.input_data == 'mycroft':
        data = pd.read_csv("../resources/output/test_sample_with_filter.csv", names=["csv_data"])

        labels = (
            data["csv_data"]
                .apply(lambda i: [j for j in ast.literal_eval(i)])
                .apply(pd.Series)
                .rename(columns={0: "label", 1: "data"})
        )

        data = pd.DataFrame(labels['data'][:100])
        labels = pd.DataFrame(labels['label'][:100])

        print(len(labels['label'].unique()))

        if args.extract:
            X_train, y_train = extract_features(data, labels)
            print('Features extracted')

            # Save the extracted features
            output_file(X_train, "../resources/output/train_data.p")
            output_file(y_train, "../resources/output/test_data.p")
        else:
            # Load pre-extracted features of sample file
            with open('../resources/output/train_data.p', 'rb') as f:
                X_train = pickle.load(f)

            with open('../resources/output/test_data.p', 'rb') as f:
                y_train = pickle.load(f)
    else:
        sys.exit("Choose the appropriate arguments for the input data")

    # For simplicity provide X_train as validation set.
    train_val_model(X_train, y_train, X_train, y_train, 'retrain_minimal_sample')
    print('Trained new model.')

    # Predict labels using the retrained model (with nn_id retrain_minimal_sample)
    # predicted_labels = predict_sherlock(X_train, 'retrain_minimal_sample')
    # print('Predicted labels: ', predicted_labels, 'true labels: ', y_train)
    #
    # f1_score(y_train, predicted_labels, average='weighted')
