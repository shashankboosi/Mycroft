import pandas as pd
import ast

x = pd.read_csv("test.csv", names=['csv_data'])

print(x)
y = x.iloc[0]
print(list(y))
y = x['csv_data'].apply(
    lambda i: [j for j in ast.literal_eval(i)]
).apply(pd.Series).rename(columns={0: "label", 1: "data"})

print(y['data'])
