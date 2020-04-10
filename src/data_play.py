import json
import pandas as pd

# The final dataset is going to be English-Language Relational Web Tables 2015

file_path = '../data/sample'


def get_data_from_file(path):
    """
    This function extracts all the json objects into a list of dictionaries
    which can then be sent to pandas Data frame for cleaning.

    Example: The 'sample' data file contains 10000 json objects in a single file.

    :param path: Path to the file
    :return: (dataList)  - List[Dict]
    """
    dataList = []
    with open(path) as file:
        for jsonObjects in file:
            dataDict = json.loads(jsonObjects)
            dataList.append(dataDict)

    return dataList


def table_display_for_relation_column(data_frame, row_number):
    """
    :param data_frame: DataFrame from which the tables are displayed
    :param row_number: Row Number of the table that is needed to be displayed
    :return: Pretty Display of List of Lists for better view
    """
    table_orientation = data_frame.iloc[row_number]['tableOrientation']
    if table_orientation == 'vertical'.upper():
        print('Vertical Oriented Table Display')
        for vertical in data_frame.iloc[row_number]['relation']:
            print(*vertical, sep='|\t\t|', end='\n')
    elif table_orientation == 'horizontal'.upper():
        print('Horizontal Oriented Table Display')
        for horizontal in zip(*data_frame.iloc[row_number]['relation']):
            print(horizontal, end='\n')
    print()


df = pd.DataFrame(get_data_from_file(file_path))

columns_to_store = [
    'relation', 'pageTitle', 'hasHeader', 'headerPosition', 'tableType',
    'tableOrientation', 'hasKeyColumn', 'keyColumnIndex', 'headerRowIndex'
]

columns_to_drop = [x for x in list(df.columns) if x not in columns_to_store]
df.drop(columns_to_drop, axis=1, inplace=True)
