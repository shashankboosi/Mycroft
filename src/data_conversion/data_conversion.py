import csv
import json
import os
from glob import glob

import pandas as pd


# The final dataset is going to be English-Language Relational Web Tables 2015

def get_semantic_types():
    with open('../../resources/data/semantic_types.csv', newline='') as f:
        type_list = list(csv.reader(f))[0]

    return list(map(lambda x: x.lower().replace(" ", ""), type_list))


class WebTableFormatToMycroftFormat:
    def __init__(self, input_file_path, output_file_path):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.input_data = self.get_data_from_file()
        self.semantic_types = get_semantic_types()

    def transform(self):
        df = pd.DataFrame(self.input_data)

        columns_to_store = [
            "relation",
            "pageTitle",
            "headerPosition",
            "hasHeader",
            "tableType",
            "tableOrientation",
        ]

        columns_to_drop = [x for x in list(df.columns) if x not in columns_to_store]
        df.drop(columns_to_drop, axis=1, inplace=True)

        """
        1) Delete rows which doesn't have a header
        2) Extract only the RELATION table types from the dataset with headers

        There are five different table types in total but we are onlt concerned about the relational tables
        print(df['tableType'].unique()) - ['LAYOUT' 'ENTITY' 'RELATION' 'MATRIX' 'OTHER']
        """
        df_relation = df[
            (df["hasHeader"] == True) & (df["tableType"] == "RELATION")
            ].reset_index(drop=True)

        # Drop hasHeader and tableType as they have been used
        df_relation.drop(["hasHeader", "tableType"], inplace=True, axis=1)

        # Display of the tables for different orientations
        print(self.table_display_for_relation_column(df_relation, 2))

        df_final = (
            df_relation.apply(
                lambda row: self.conversion_of_relation_column_based_on_table_orientation(
                    row["relation"], row["tableOrientation"]
                ), axis=1
            ).apply(pd.Series).stack().reset_index(drop=True)
        )

        df_final.to_csv(self.output_file_path, header=False, index=False)

    def get_data_from_file(self):
        """This function extracts all the json objects into a list of dictionaries
        which can then be sent to pandas Data frame for cleaning.

        Arguments:
            path {String} -- Path to the file/Directory

        Returns:
            List[Dict] -- data_list
        """
        data_list = []
        if os.path.isdir(self.input_file_path):
            for filename in glob(os.path.join(self.input_file_path, '*.json')):
                with open(filename, encoding="utf-8") as file:
                    data = json.load(file)
                    data_list.append(data)
        else:
            data_list = []
            with open(self.input_file_path, encoding="utf-8") as file:
                for jsonObjects in file:
                    data_dict = json.loads(jsonObjects)
                    data_list.append(data_dict)

        return data_list

    def conversion_of_relation_column_based_on_table_orientation(self, relation, orientation):
        """This function converts the relation attribute from the web data columns into
        the format that Mycroft requires as the base dataset.

        Arguments:
            relation {Pandas Series} -- Attribute which describes the table and is part of Web Data Commons 2015
            orientation {Pandas Series} -- Attribute which tells the orientation of the tables ( Eg: HORIZONTAL, VERTICAL)

        Returns:
            List -- A list of final values in the format [[label, [data]], [label, [data]] ... ]
        """
        bad_chars = ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '>', '#', '+', '-', '.', '!', '$', '\'', ';',
                     ':']
        transposed_list = relation
        if orientation == "VERTICAL":
            transposed_list = list(map(list, zip(*relation)))

        result_list = []
        for i in range(len(transposed_list)):
            transformed_string = "".join(
                i for i in transposed_list[i][0] if i not in bad_chars
            ).lower().replace(" ", "")
            if transformed_string != "" and transformed_string in self.semantic_types:
                data = transposed_list[i][1:]
                data = [i.replace('-', ' ').replace('"', ' ').strip() for i in data]
                if '' in data:
                    data = list(filter(None, data))
                if len(data) < 3:
                    continue
                result_list.append([str(transformed_string), data])
            else:
                continue
        return result_list

    @staticmethod
    def table_display_for_relation_column(data_frame, row_number):
        """Pretty Display of List of Lists for better view

        Arguments:
            data_frame {Pandas Dataframe} -- DataFrame from which the tables are displayed
            row_number {Integer} -- Row Number of the table that is needed to be displayed
        """
        table_orientation = data_frame.iloc[row_number]["tableOrientation"]
        if table_orientation == "vertical".upper():
            print("Vertical Oriented Table Display")
            for vertical in data_frame.iloc[row_number]["relation"]:
                print(*vertical, sep="|\t\t|", end="\n")
        elif table_orientation == "horizontal".upper():
            print("Horizontal Oriented Table Display")
            for horizontal in zip(*data_frame.iloc[row_number]["relation"]):
                print(horizontal, end="\n")
        print()
