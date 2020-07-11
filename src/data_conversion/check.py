import pandas as pd
import ast

x = pd.read_csv("../../resources/output/test_sample_with_filter.csv", names=["csv_data"])

y = (
    x["csv_data"]
    .apply(lambda i: [j for j in ast.literal_eval(i)])
    .apply(pd.Series)
    .rename(columns={0: "label", 1: "data"})
)

print(y["data"])
