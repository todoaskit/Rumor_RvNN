import os
import random
from format_story import *


def export_with_order():
    stories = get_formatted_stories(data_path='../../../data')
    story_ids = [str(s) + '\n' for s in list(stories.story_to_id.keys())]

    train_ratio = 1 - 1/5
    train_bound = int(len(story_ids)*train_ratio)

    train_lines = story_ids[:train_bound]
    test_lines = story_ids[train_bound:]

    train_merged_file = open("../nfold/RNNtrainSet_Twitter1516" + str(6) + "_tree.txt", 'w')
    train_merged_file.writelines(train_lines)
    train_merged_file.close()

    test_merged_file = open("../nfold/RNNtestSet_Twitter1516" + str(6) + "_tree.txt", 'w')
    test_merged_file.writelines(test_lines)
    test_merged_file.close()


def merge_nfold_1516(_fold):

    if [x for x in os.listdir('../nfold') if 'Twitter1516{}'.format(_fold) in x]:
        print("Already Exist: ../nfold/RNN.*Set_Twitter1516" + str(_fold) + "_tree.txt")
        return

    train_lines = []
    test_lines = []
    for _obj in ['Twitter15', 'Twitter16']:
        _train_path = "../nfold/RNNtrainSet_" + _obj + str(_fold) + "_tree.txt"
        _test_path = "../nfold/RNNtestSet_" + _obj + str(_fold) + "_tree.txt"

        train_lines += list(open(_train_path, 'r').readlines())
        test_lines += list(open(_test_path, 'r').readlines())

    train_lines = list(set(train_lines))
    test_lines = list(set(test_lines))
    total_lines = list(set(train_lines + test_lines))
    random.shuffle(total_lines)

    train_ratio = 1 - 1/5
    train_bound = int(len(total_lines)*train_ratio)
    train_lines = total_lines[:train_bound]
    test_lines = total_lines[train_bound:]

    train_merged_file = open("../nfold/RNNtrainSet_Twitter1516" + str(_fold) + "_tree.txt", 'w')
    train_merged_file.writelines(train_lines)
    train_merged_file.close()

    test_merged_file = open("../nfold/RNNtestSet_Twitter1516" + str(_fold) + "_tree.txt", 'w')
    test_merged_file.writelines(test_lines)
    test_merged_file.close()


def merge_labels_1516():
    label_lines = []
    for _obj in ['Twitter15', 'Twitter16']:
        label_file = open("../resource/" + _obj + "_label_All.txt", 'r')
        label_lines += list(label_file.readlines())
    label_lines = list(set(label_lines))
    merge_labels = open("../resource/Twitter1516_label_All.txt", 'w')
    merge_labels.writelines(label_lines)


if __name__ == '__main__':
    export_with_order()
