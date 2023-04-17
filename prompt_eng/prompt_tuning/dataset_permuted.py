"""
Generate utterance-language symbolic dataset, where propositions are letters, and train, test splits.
"""
import sys
sys.path.insert(1, '../../')
import argparse
import os
from pathlib import Path
from pprint import pprint
import random
import string
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold
import csv
from utils import load_from_file, save_to_file, append_ids_to_path, remove_id_from_path, deserialize_props_str, substitute_single_letter
from formula_sampler import PROPS, FEASIBLE_TYPES, FILTER_TYPES, sample_formulas

# def generate_lc_splits(split_fpath, portions=[0.1, 0.3, 0.5, 0.7], seed=42):
#     """
#     Split one seed/fold for plotting learning curve.
#     """
#     train_iter, train_meta, valid_iter, valid_meta = load_split_dataset(split_fpath)

#     meta2data = defaultdict(list)
#     for idx, ((utt, ltl), (pattern_type, props)) in enumerate(zip(train_iter, train_meta)):
#         meta2data[(pattern_type, len(props))].append((utt, ltl))

#     for portion in portions:
#         train_iter_new, train_meta_new = [], []
#         for (pattern_type, nprop), data in meta2data.items():
#             random.seed(seed)
#             data = sorted(data)
#             random.shuffle(data)
#             examples = data[:int(len(data)*portion)]
#             print(f"Num of {pattern_type}, {nprop}: {len(examples)}")
#             for pair in examples:
#                 train_iter_new.append(pair)
#                 train_meta_new.append((pattern_type, nprop))
#         split_dataset_name = Path(split_fpath).stem
#         lc_split_fname = f"lc_{portion}_{split_dataset_name}.pkl"
#         split_pkl_new = {"train_iter": train_iter_new, "train_meta": train_meta_new, "valid_iter": valid_iter, "valid_meta": valid_meta}
#         save_to_file(split_pkl_new, os.path.join(os.path.dirname(split_fpath), lc_split_fname))

def construct_split_dataset(data_fpath, split_dpath, holdout_type, feasible_types, filter_types, perm_props, size, seed, firstn):
    """
    K-fold cross validation for type and formula holdout. Random sample for utterance holdout.
    If perm_props=True, assume data_fpath non-permuted for utt holdout, permuted for formula, type holdouts.
    :param data_fpath: path to symbolic dataset.
    :param split_dpath: directory to save train, test split.
    :param holdout_type: type of holdout test. choices are ltl_type, ltl_formula, utt.
    :param feasible_types: all LTL types except filter types
    :param filter_types: LTL types to filter out.
    :param perm_props: True if permute propositions in utterances and their corresponding LTL formula.
    :param size: size of each fold for type and formula holdout; ratio of utterances to holdout for utterance holdout.
    :param seed: random seed for train, test split.
    :param firstn: use first n training samples of each formula for utt, formula, type holdout.
    """
    print(f"Generating train, test split for holdout type: {holdout_type}; seed: {seed}")
    dataset = load_from_file(data_fpath)
    dataset_name = f"{'_'.join(Path(data_fpath).stem.split('_')[:2])}_perm" if perm_props else Path(data_fpath).stem  # remove noperm identifier from dataset name if perm_props=True

    if holdout_type == "ltl_type":  # hold out specified pattern types
        kf = KFold(n_splits=3, random_state=seed, shuffle=True)
        for fold_idx, (train_indices, holdout_indices) in enumerate(kf.split(feasible_types)):
            csv_train = []
            csv_test = []
            holdout_types = [feasible_types[idx] for idx in holdout_indices]
            train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data is (pattern_type, nprops) pair
            formula2count = defaultdict(int)
            for pattern_type, props_str, utt, ltl in dataset:
                props = deserialize_props_str(props_str)
                if pattern_type in holdout_types:
                    valid_iter.append((utt, ltl))
                    valid_meta.append((pattern_type, props))
                    csv_test.append((utt, ltl, pattern_type, props))
                elif pattern_type in feasible_types:
                    train_iter.append((utt, ltl))
                    train_meta.append((pattern_type, props))
                    csv_train.append((utt, ltl, pattern_type, props))
            dataset_name = Path(data_fpath).stem
            split_fpath = f"{split_dpath}/{dataset_name}_{holdout_type}_{size}_{seed}_fold{fold_idx}.pkl"

            header = ["instruction", "ltl", "pattern_type", "props"]
            # Open the file in 'write' mode
            with open(f"csv_data/train_{holdout_type}_{size}_{seed}_fold{fold_idx}.csv", mode='w') as file:

                # Create a csv writer object
                writer = csv.writer(file)
                writer.writerow(header)
                # Write each row of the data array to the csv file
                for row in csv_train:
                    writer.writerow(row)

            # Open the file in 'write' mode
            with open(f"csv_data/test_{holdout_type}_{size}_{seed}_fold{fold_idx}.csv", mode='w') as file:

                # Create a csv writer object
                writer = csv.writer(file)
                writer.writerow(header)
                # Write each row of the data array to the csv file
                for row in csv_test:
                    writer.writerow(row)

            save_split_dataset(split_fpath, train_iter, train_meta, valid_iter, valid_meta,
                               size, seed, holdout_type, holdout_types)
    elif holdout_type == "ltl_formula":  # hold out specified (pattern type, nprops) pairs
        all_formulas = []
        for pattern_type, props_str, _, _ in dataset:
            props = deserialize_props_str(props_str)
            formula = (pattern_type, len(props))  # same type and nprops, diff perm of props considered same formula
            if pattern_type not in filter_types and formula not in all_formulas:
                all_formulas.append(formula)
        kf = KFold(n_splits=3, random_state=seed, shuffle=True)
        for fold_idx, (train_indices, holdout_indices) in enumerate(kf.split(all_formulas)):
            csv_train = []
            csv_test = []
            holdout_formulas = [all_formulas[idx] for idx in holdout_indices]
            train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data is (pattern_type, nprops) pairs
            formula2count = defaultdict(int)
            for pattern_type, props_str, utt, ltl in dataset:
                props = deserialize_props_str(props_str)
                formula = (pattern_type, len(props))
                if formula in holdout_formulas:
                    valid_iter.append((utt, ltl))
                    valid_meta.append((pattern_type, props))
                    csv_test.append((utt, ltl, pattern_type, props))
                elif formula in all_formulas:
                    train_iter.append((utt, ltl))
                    train_meta.append((pattern_type, props))
                    csv_train.append((utt, ltl, pattern_type, props))

            split_fpath = f"{split_dpath}/{dataset_name}_{holdout_type}_{size}_{seed}_fold{fold_idx}.pkl"

            header = ["instruction", "ltl", "pattern_type", "props"]
            # Open the file in 'write' mode
            with open(f"csv_data/train_{dataset_name}_{holdout_type}_{size}_{seed}_fold{fold_idx}.csv", mode='w') as file:

                # Create a csv writer object
                writer = csv.writer(file)
                writer.writerow(header)
                # Write each row of the data array to the csv file
                for row in csv_train:
                    writer.writerow(row)

            # Open the file in 'write' mode
            with open(f"csv_data/test_{dataset_name}_{holdout_type}_{size}_{seed}_fold{fold_idx}.csv", mode='w') as file:

                # Create a csv writer object
                writer = csv.writer(file)
                writer.writerow(header)
                # Write each row of the data array to the csv file
                for row in csv_test:
                    writer.writerow(row)

            save_split_dataset(split_fpath, train_iter, train_meta, valid_iter, valid_meta,
                               size, seed, holdout_type, holdout_formulas)
    elif holdout_type == "utt":  # hold out a specified ratio of utts for every (pattern type, nprops) pair
        train_iter, train_meta, valid_iter, valid_meta = [], [], [], []  # meta data is (pattern_type, nprops) pairs
        csv_train = []
        csv_test = []
        meta2data = defaultdict(list)
        for pattern_type, props_str, utt, ltl in dataset:
            props = deserialize_props_str(props_str)
            if pattern_type not in filter_types:
                meta2data[(pattern_type, len(props))].append((utt, ltl))
        for (pattern_type, nprops), data in meta2data.items():
            train_dataset, valid_dataset = train_test_split(data, test_size=size, random_state=seed)
            for utt, ltl in train_dataset:
                if perm_props:
                    permute(pattern_type, nprops, utt, train_iter, train_meta)
                else:
                    train_iter.append((utt, ltl))
                    train_meta.append((pattern_type, PROPS[:nprops]))
                    csv_train.append((utt, ltl, pattern_type, PROPS[:nprops]))
            for utt, ltl in valid_dataset:
                if perm_props:
                    permute(pattern_type, nprops, utt, valid_iter, valid_meta)
                else:
                    valid_iter.append((utt, ltl))
                    valid_meta.append((pattern_type, PROPS[:nprops]))
                    csv_test.append((utt, ltl, pattern_type, PROPS[:nprops]))
        
        split_fpath = f"{split_dpath}/{dataset_name}_{holdout_type}_{size}_{seed}.pkl"

        header = ["instruction", "ltl", "pattern_type", "props"]
        # Open the file in 'write' mode
        with open(f"csv_data/train_{dataset_name}_{holdout_type}_{size}_{seed}.csv", mode='w') as file:

            # Create a csv writer object
            writer = csv.writer(file)
            writer.writerow(header)
            # Write each row of the data array to the csv file
            for row in csv_train:
                writer.writerow(row)

        # Open the file in 'write' mode
        with open(f"csv_data/test_{dataset_name}_{holdout_type}_{size}_{seed}.csv", mode='w') as file:

            # Create a csv writer object
            writer = csv.writer(file)
            writer.writerow(header)
            # Write each row of the data array to the csv file
            for row in csv_test:
                writer.writerow(row)

        save_split_dataset(split_fpath, train_iter, train_meta, valid_iter, valid_meta,
                           size, seed, holdout_type, list(meta2data.keys()))
    else:
        raise ValueError(f"ERROR unrecognized holdout type: {holdout_type}.")


def permute(pattern_type, nprops, utt, data, meta):
    """
    Add utterances and LTL formulas with all possible permutations to data and meta data.
    """
    ltls_perm, props_perm = sample_formulas(pattern_type, nprops, False)  # sample ltls w/ all possible perms
    for ltl_perm, prop_perm in zip(ltls_perm, props_perm):
        sub_map = {prop_old: prop_new for prop_old, prop_new in zip(PROPS[:nprops], prop_perm)}
        utt_perm = substitute_single_letter(utt, sub_map)  # sub props in utt w/ permutation corres to ltl
        data.append((utt_perm, ltl_perm))
        meta.append((pattern_type, prop_perm))


def save_split_dataset(split_fpath, train_iter, train_meta, valid_iter, valid_meta, size, seed, holdout_type=None, holdout_meta=None):
    split_dataset = {
        "train_iter": train_iter, "train_meta": train_meta, "valid_iter": valid_iter, "valid_meta": valid_meta,
        "holdout_type": holdout_type,
        "holdout_meta": holdout_meta,
        "size": size,
        "seed": seed
    }
    save_to_file(split_dataset, split_fpath)


def save_split_dataset_new(split_fpath, train_iter, train_meta, valid_iter, valid_meta, info):
    split_dataset = {
        "train_iter": train_iter, "train_meta": train_meta, "valid_iter": valid_iter, "valid_meta": valid_meta,
        "info": info,
    }
    save_to_file(split_dataset, split_fpath)


def load_split_dataset(split_fpath):
    dataset = load_from_file(split_fpath)
    return dataset["train_iter"], dataset["train_meta"], dataset["valid_iter"], dataset["valid_meta"]


if __name__ == "__main__":
    # python dataset_symbolic.py --perm --update --merge --nexamples=1
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fpath", action="store", type=str, nargs="+", default=["data/aggregated_responses_batch1.csv", "data/aggregated_responses_batch2.csv"], help="fpath to aggregated Google form responses.")
    parser.add_argument("--perm", action="store_true", help="True if permute props after train, test split.")
    parser.add_argument("--update", action="store_true", help="True if update existing symbolic dataset w/ new responses.")
    parser.add_argument("--merge", action="store_true", help="True if merge Google form responses from batches.")
    parser.add_argument("--remove_pun", action="store_true", help="True if remove punctuations.")
    parser.add_argument("--seeds_split", action="store", type=int, nargs="+", default=[0, 1, 2, 42, 111], help="1 or more random seeds for train, test split.")
    parser.add_argument("--firstn", type=int, default=None, help="only use first n training samples per formula.")
    parser.add_argument("--nexamples", action="store", type=int, nargs="+", default=[1, 2, 3], help="number of examples per formula in prompt.")
    parser.add_argument("--seed_prompt", type=int, default=42, help="random seed for choosing prompt examples.")
    args = parser.parse_args()

    # Construct train, test split for utt holdout; permute if asked
    symbolic_fpath = "/Users/SamLiang/Desktop/LTL_prompt_eng/data/symbolic_batch1_perm.csv"
    split_dpath = "prompt_eng/prompt_tuning/data/holdout_batch1_permute"
    os.makedirs(split_dpath, exist_ok=True)

    seeds = [123]
    args.perm = False
    # 30% of dataset used for test dataset
    for seed in seeds:
        construct_split_dataset(symbolic_fpath, split_dpath, "utt", FEASIBLE_TYPES, FILTER_TYPES, args.perm, size=0.3, seed=seed, firstn=args.firstn)

    # Construct train, test split for formula, type holdout; permute if asked
    construct_split_dataset(symbolic_fpath, split_dpath, "ltl_type", FEASIBLE_TYPES, FILTER_TYPES, args.perm, size=2, seed=123, firstn=args.firstn)
    construct_split_dataset(symbolic_fpath, split_dpath, "ltl_formula", FEASIBLE_TYPES, FILTER_TYPES, args.perm, size=4, seed=123, firstn=args.firstn)