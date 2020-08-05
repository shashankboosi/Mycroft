import argparse
import time

from data_conversion import WebTableFormatToMycroftFormat
from data_stats import get_data_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web Table Format To Mycroft Format Converter')
    parser.add_argument('--sample', '-s', default=False, type=bool,
                        help="Choose if you want to use sample or not")
    parser.add_argument('--no_of_tables', '-num', default=1000, type=int,
                        help="Choose the number of tables that are needed for Mycroft")

    args = parser.parse_args()

    start = time.time()
    if args.sample:
        output_path = "../../resources/output/sample_with_filter.csv"
        input_converter = WebTableFormatToMycroftFormat("../../resources/data/sample", output_path, args.no_of_tables)

    else:
        output_path = "../../resources/output/mycroft_{}_tables.csv".format(args.no_of_tables)
        input_converter = WebTableFormatToMycroftFormat("../../resources/data/mini-data", output_path,
                                                        args.no_of_tables)
    input_converter.transform()
    time_elapsed = time.time() - start
    print('Time taken for the data extraction is {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    get_data_stats(output_path)
