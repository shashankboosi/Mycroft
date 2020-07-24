import ast
import pickle
import sys
import warnings

import pandas as pd

from src.deploy.train_sherlock import train_predict_sherlock
from src.helpers.utils import output_file

warnings.filterwarnings("ignore")
sys.path.append("..")

x = pd.read_csv("../resources/output/test_sample_with_filter.csv", names=["csv_data"])

y = (
    x["csv_data"]
        .apply(lambda i: [j for j in ast.literal_eval(i)])
        .apply(pd.Series)
        .rename(columns={0: "label", 1: "data"})
)

print(y.head())
print(y["data"])

data = pd.DataFrame(y['data'][:100])
print(data.head())
print(data.shape)

labs = pd.DataFrame(y['label'][:100])
print(labs.head())
print(labs.shape)

print(len(labs['label'].unique()))

# Load pre-extracted features of sample file
with open('../resources/output/train_data.p', 'rb') as f:
    X_train = pickle.load(f)

with open('../resources/output/test_data.p', 'rb') as f:
    y_train = pickle.load(f)

# X_train = build_features(data)
# y_train = labs.values.flatten()
# print('Extracted features.')

output_file(X_train, "../resources/output/train_data.p")
output_file(y_train, "../resources/output/test_data.p")

# For simplicity provide X_train as validation set.
train_predict_sherlock(X_train, y_train, X_train, y_train, 'retrain_minimal_sample')
print('Trained new model.')

# Predict labels using the retrained model (with nn_id retrain_minimal_sample)
# predicted_labels = predict_sherlock(X_train, 'retrain_minimal_sample')
# print('Predicted labels: ', predicted_labels, 'true labels: ', y_train)
#
# f1_score(y_train, predicted_labels, average='weighted')
