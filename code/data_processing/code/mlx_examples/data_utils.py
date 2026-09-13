# Copyright Reexpress AI, Inc. All rights reserved.

import json
import codecs


def save_by_appending_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "a", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def read_jsons_lines_file(filename_with_path):
    """
    Get the preprocessed data
    :param filename_with_path: A filepath to the preprocessed data.
    :return: A list of dictionaries
    """
    json_list = []
    with codecs.open(filename_with_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            json_list.append(json_obj)
    return json_list
