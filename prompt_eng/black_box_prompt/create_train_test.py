import sys
sys.path.insert(1, '../../')
from sklearn.model_selection import train_test_split
from utils import load_from_file, deserialize_props_str
from formula_sampler import PROPS, FEASIBLE_TYPES, FILTER_TYPES, sample_formulas
import csv
from collections import defaultdict
from datasets import load_dataset


# we will create a dataset where the problem is now formulated as classification into the 5 types in the  FEASIBLE_TYPES

type_to_id = {
    "visit": 0, "sequenced_visit": 1, "ordered_visit": 2, "strictly_ordered_visit": 3, "patrolling": 4
}

if __name__ == '__main__':
  
    dataset = load_from_file("/Users/SamLiang/Desktop/LTL_prompt_eng/data/symbolic_batch1_noperm.csv")
    meta2data = defaultdict(list)
    second = defaultdict(list)

    for pattern_type, props_str, utt, ltl in dataset:
        props = deserialize_props_str(props_str)
        if pattern_type not in FILTER_TYPES:
            meta2data[(pattern_type, len(props))].append((utt, ltl))

    train_iter, valid_iter, test_iter = [], [], []  # meta data is (pattern_type, nprops) pairs
    train_meta, valid_meta, test_meta = [], [], []
    train_iter_temp = []


    seed = 484

    for (pattern_type, nprops), data in meta2data.items():
        # datasets will only contains formulas of pattern_type and nprops
        train_set, test_set = train_test_split(data, test_size=0.25, random_state=seed)
        for utt, ltl in train_set:
            train_iter_temp.append((pattern_type, nprops, utt, ltl))
        for utt, ltl in test_set:
            test_iter.append((utt, pattern_type))
            test_meta.append((ltl, PROPS[:nprops]))

    for pattern_type, nprops, utt, ltl in train_iter_temp:
        if pattern_type not in FILTER_TYPES:
            second[(pattern_type, nprops)].append((utt, ltl))

    for (pattern_type, nprops), data in second.items():
        # datasets will only contains formulas of pattern_type and nprops
        train_set, validation_set = train_test_split(data, test_size=0.15, random_state=seed)
        for utt, ltl in train_set:
            train_iter.append((utt, pattern_type))
            train_meta.append((ltl, PROPS[:nprops]))
        for utt, ltl in validation_set:
            valid_iter.append((utt, pattern_type))
            valid_meta.append((ltl, PROPS[:nprops]))

    # data = []
    # for _, _, instruction, ground_truth in dataset:
    #     data.append((instruction, ground_truth))

    # train_set, test_set = train_test_split(data, test_size=0.25, random_state=seed)
    # train_set, validation_set = train_test_split(train_set, test_size=0.15, random_state=seed)

    header = ["instruction", "label"]
    # Open the file in 'write' mode
    with open("train_set.csv", mode='w') as file:

        # Create a csv writer object
        writer = csv.writer(file)
        writer.writerow(header)
        # Write each row of the data array to the csv file
        for row in train_iter:
            writer.writerow(row)

    # Open the file in 'write' mode
    with open("validation_set.csv", mode='w') as file:

        # Create a csv writer object
        writer = csv.writer(file)
        writer.writerow(header)
        # Write each row of the data array to the csv file
        for row in valid_iter:
            writer.writerow(row)

    # Open the file in 'write' mode
    with open("test_set.csv", mode='w') as file:

        # Create a csv writer object
        writer = csv.writer(file)
        writer.writerow(header)
        # Write each row of the data array to the csv file
        for row in test_iter:
            writer.writerow(row)

    header2 = ["ltl", "props"]
    
    with open("train_meta.csv", mode='w') as file:

        # Create a csv writer object
        writer = csv.writer(file)
        writer.writerow(header2)
        # Write each row of the data array to the csv file
        for row in train_meta:
            writer.writerow(row)

    # Open the file in 'write' mode
    with open("validation_meta.csv", mode='w') as file:

        # Create a csv writer object
        writer = csv.writer(file)
        writer.writerow(header2)
        # Write each row of the data array to the csv file
        for row in valid_meta:
            writer.writerow(row)

    # Open the file in 'write' mode
    with open("test_meta.csv", mode='w') as file:

        # Create a csv writer object
        writer = csv.writer(file)
        writer.writerow(header2)
        # Write each row of the data array to the csv file
        for row in test_meta:
            writer.writerow(row)

    # data_files = {}
    # data_files["train"] = "train_set.csv"
    # data_files["validation"] = "validation_set.csv"
    # data_files["test"] = "test_set.csv"
    # extension = "csv"
    # raw_datasets = load_dataset(extension, data_files=data_files)
    # print(raw_datasets)
    # print(raw_datasets["train"]['label'])

