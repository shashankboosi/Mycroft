import argparse

from data_conversion import WebTableFormatToMycroftFormat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web Table Format To Mycroft Format Converter')
    parser.add_argument('--sample', '-s', default=False, type=bool,
                        help="Choose if you want to use sample or not")

    args = parser.parse_args()

    if args.sample:

        input_converter = WebTableFormatToMycroftFormat("../../resources/data/sample",
                                                        "../../resources/output/test_sample_with_filter.csv")

    else:
        input_converter = WebTableFormatToMycroftFormat("../../resources/data/mycroft",
                                                        "../../resources/output/mycroft_with_filter.csv")
    input_converter.transform()
