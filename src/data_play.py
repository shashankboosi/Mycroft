import json
import pandas as pd

file_path = '../data/sample'


def get_data_from_file(path):
    """

        This function extracts all the json objects into a list of dictionaries
        which can then be sent to pandas Data frame for cleaning.

        Example: The 'sample' data file contains 10000 json objects in a single file.
        Input : (path) - Path to the file
        Output: (dataList)  - List[Dict]

    """
    dataList = []
    with open(path) as file:
        for jsonObjects in file:
            dataDict = json.loads(jsonObjects)
            dataList.append(dataDict)

    return dataList


df = pd.DataFrame(get_data_from_file(file_path))

columns_to_store = [
    'relation', 'pageTitle', 'hasHeader', 'headerPosition', 'tableType',
    'tableOrientation', 'hasKeyColumn', 'keyColumnIndex', 'headerRowIndex'
]

columns_to_drop = [x for x in list(df.columns) if x not in columns_to_store]
df.drop(columns_to_drop, axis=1, inplace=True)

print(len(df.loc[df['hasKeyColumn'] == True]))

print(df.columns)
# Delete rows which doesn't have a header
df1 = df[df['hasHeader'] == True].reset_index(drop=True)
print(len(df1))
