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


def conversion_of_relation_column_based_on_table_orientation(relation, orientation):
    """
    This function converts the relation attribute from the web data columns into
    the format that Mycroft requires as the base dataset.
    :param relation: Attribute which describes the table and is part of Web Data Commons 2015
    :param orientation: Attribute which tells the orientation of the tables ( Eg: HORIZONTAL, VERTICAL)
    :return: returns a list of final values in the format [[label, [data]], [label, [data]] ... ]
    """
    bad_chars = ['#', '.', '!', ':', ';']
    transposed_list = relation
    if orientation == 'VERTICAL':
        transposed_list = list(map(list, zip(*relation)))

    result_list = []
    for i in range(len(transposed_list)):
        transformed_string = ''.join(i for i in transposed_list[i][0] if not i in bad_chars)
        if transformed_string != '':
            result_list.append([str(transformed_string), transposed_list[i][1:]])
        else:
            continue
    return result_list


df = pd.DataFrame(get_data_from_file(file_path))

columns_to_store = [
    'relation', 'pageTitle', 'headerPosition', 'hasHeader', 'tableType', 'tableOrientation'
]

columns_to_drop = [x for x in list(df.columns) if x not in columns_to_store]
df.drop(columns_to_drop, axis=1, inplace=True)

'''
1) Delete rows which doesn't have a header
2) Extract only the RELATION table types from the dataset with headers

There are five different table types in total but we are onlt concerned about the relational tables
print(df['tableType'].unique()) - ['LAYOUT' 'ENTITY' 'RELATION' 'MATRIX' 'OTHER']
'''
df_relation = df[(df['hasHeader'] == True) & (df['tableType'] == 'RELATION')].reset_index(drop=True)

# Drop hasHeader and tableType as they have been used
df_relation.drop(['hasHeader', 'tableType'], inplace=True, axis=1)

# Display of the tables for different orientations
# print(table_display_for_relation_column(df_headers, 2))

df_final = df_relation.apply(
    lambda row: conversion_of_relation_column_based_on_table_orientation(row['relation'], row['tableOrientation']),
    axis=1
).apply(pd.Series).stack().reset_index(drop=True)

df_final.to_csv("test.csv", header=False, index=False)
