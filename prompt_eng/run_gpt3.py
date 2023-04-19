import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import time
from evaluation import evaluate_lang_0
import logging
import os
import openai
import csv
from utils import save_to_file
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import pickle
from models_util import gpt3_complete

openai.api_key = "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"
openai.organization = "org-g3Ul3oNaC7RRJGYw9fy78mdi"


def run_experiment(dataset_object, max_tokens, model_name, batch_identifier, is_cot, temp=0, num_log_probs=None, echo=False, n=1, fold=None, iter_num=None):

    # smaller datasets
    # test_dataset = dataset_object.get('smaller_valid')
    # validation_meta = dataset_object.get('smaller_valid_meta')

    holdout_type = dataset_object.get('holdout_type')

    # full datasets
    if iter_num is None:
        test_dataset = dataset_object.get('valid_iter')
    else:
        # rewritten instructions
        with open(f"{holdout_type}_rewritten_{fold}.pkl", 'rb') as file:
            print(f"Using {holdout_type} {fold} rewritten dataset\n\n")
            # dictionary with keys ['smaller_valid', 'smaller_valid_meta', 'holdout_type', 'holdout_meta', 'seed', 'size', 'dataset_size']
            test_dataset = pickle.load(file)

        # # rewritten instructions for utterance holdout
        # with open(f"/Users/SamLiang/Desktop/LTL_prompt_eng/prompt_eng/rewritten_instructions_iter6.pkl", 'rb') as file:
        #     print(f"Using {holdout_type} rewritten dataset\n\n")
        #     # dictionary with keys ['smaller_valid', 'smaller_valid_meta', 'holdout_type', 'holdout_meta', 'seed', 'size', 'dataset_size']
        #     test_dataset = pickle.load(file)

    validation_meta = dataset_object.get('valid_meta')

    seed = dataset_object.get('seed')
    size = dataset_object.get('size')

    print("test dataset size before: ", len(test_dataset))
    print("meta size before: ", len(validation_meta))
    print()

    # if holdout_type != "utt":
    #     with open(f'forbidden_formula_{fold}.pkl', 'rb') as f:
    #         # Load the array from the file using pickle.load()
    #         forbidden = pickle.load(f)
        
    #     train_dataset = dataset_object.get("train_iter")
    #     train_meta = dataset_object.get("train_meta")
    #     print("train dataset size: ", len(train_dataset))
    #     print("train meta size: ", len(train_meta))
    #     for train, meta in zip(train_dataset, train_meta):
    #         # print(instruction)
    #         # print(ground_truth)
    #         instruction = train[0]
    #         ground_truth = train[1]
    #         pattern_type = meta[0]
    #         props = meta[1]

    #         if instruction not in forbidden:
    #             test_dataset.append((instruction, ground_truth))
    #             validation_meta.append((pattern_type, props))
        
    print()
    print("seed: ", seed)
    print("size: ", size)
    print()
    print("test dataset size after: ", len(test_dataset))
    print("meta size after: ", len(validation_meta))
    print()

    # Zero shot
    # plain_prompt_location = f'data/prompts_batch{batch_identifier}_noperm/zeroshot_prompt_nexamples1_symbolic_batch{batch_identifier}_noperm_{holdout_type}_{size}_{seed}_{fold}.txt'
    # with open(plain_prompt_location, 'r') as file:
    #   prompt = file.read()


    if iter_num == "prompts_rewritten":
        if holdout_type != "utt":
            prompt_location = f'/Users/SamLiang/Desktop/LTL_prompt_eng/prompt_eng/rewritten_prompts_{holdout_type}_{size}_{seed}_{fold}.txt'
        else:
            prompt_location = f"/Users/SamLiang/Desktop/LTL_prompt_eng/prompt_eng/rewritten_prompts_utterance.txt"

        with open(prompt_location, 'r') as file:
            prompt = file.read()
    else:
        if is_cot:
            if holdout_type != "utt":
                cot_prompt_location = f'data/prompts_batch{batch_identifier}_noperm/CoT_prompt_nexamples1_symbolic_batch{batch_identifier}_noperm_{holdout_type}_{size}_{seed}_{fold}.txt'
            else:
                cot_prompt_location = f'data/prompts_batch{batch_identifier}_noperm/CoT_prompt_nexamples1_symbolic_batch{batch_identifier}_noperm_{holdout_type}.txt'
            with open(cot_prompt_location, 'r') as file:
                prompt = file.read()
        else:
            if holdout_type != "utt":
                plain_prompt_location = f'data/prompts_batch{batch_identifier}_noperm/prompt_nexamples1_symbolic_batch{batch_identifier}_noperm_{holdout_type}_{size}_{seed}_{fold}.txt'
            else:
                plain_prompt_location = f'data/prompts_batch{batch_identifier}_noperm/prompt_nexamples1_symbolic_batch{batch_identifier}_noperm_{holdout_type}.txt'
            with open(plain_prompt_location, 'r') as file:
                prompt = file.read()

    if is_cot:
        if holdout_type == "utt":
            prompt_len = prompt.count("Q:") - 1
        elif holdout_type == "ltl_formula":
            prompt_len = prompt.count("Q:") - 1
        elif holdout_type == "ltl_type":
            prompt_len = prompt.count("Q:") - 1
    else:
        if holdout_type == "utt":
            prompt_len = prompt.count("Q:") - 1
        elif holdout_type == "ltl_formula":
            prompt_len = prompt.count("Q:") - 1
        elif holdout_type == "ltl_type":
            prompt_len = prompt.count("Q:") - 1

    responses = []
    parsed_responses = []
    ground_truths = []
    instructions = []
    for instruction, ground_truth in test_dataset:
        instruction = instruction.strip()
        ground_truths.append(ground_truth)
        instructions.append(instruction)

        # new_prompt = prompt + f'\n\nQ: Translate the NLI “{instruction}” to its LTL in prefix notation.\nA:'

        # CoT and Plain Prompts
        new_prompt = prompt + f' What is "{instruction}" in LTL?\nA:'

        # Zero shot Prompts
        # new_prompt = prompt + f' What is "{instruction}" in LTL?\nA: Let\'s think step by step.'

        raw_response = gpt3_complete(new_prompt, max_tokens,
                                model_name, temp, num_log_probs, echo, n)
        if raw_response is None:
            continue
        
        response = raw_response['choices'][0]['text']
        response = response.strip()
        print(response)
        print()
        responses.append(response)

        if is_cot:
            # grab only the ltl from the model output
            res = response.split(".")
            if res[-1] != "":
                text = res[-1]
            else:
                text = res[-2]
            first_quote = text.find('is')
            parsed_response = text[first_quote+4:len(text)-1]
            print(parsed_response)
            print()
        else:
            parsed_response = response

        parsed_responses.append(parsed_response)

    accuracies_lang, total_accuracy, incorrects_eval = evaluate_lang_0(
        ground_truths, parsed_responses, responses, validation_meta, instructions)

    for idx, (input_utt, output_ltl, true_ltl, acc) in enumerate(zip(instructions, parsed_responses, ground_truths, accuracies_lang)):
        logging.info(
            f"{idx}\nInput utterance: {input_utt}\nTrue LTL: {true_ltl}\nOutput LTL: {output_ltl}\n{acc}\n")
    logging.info(f"Language to LTL translation accuracy: {total_accuracy}")

    final_results = {
        'Input utterances': instructions,
        'Output LTLs': parsed_responses,
        'Ground truth': ground_truths,
        'Accuracy': total_accuracy
    }

    if iter_num is None:
        if is_cot:
            if holdout_type != "utt":
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/validation{len(instructions)}/{seed}/normal/{fold}/CoT"
            else:
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/validation{len(instructions)}/{seed}/normal/CoT"
        else:
            if holdout_type != "utt":
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/validation{len(instructions)}/{seed}/normal/{fold}/plain"
            else:
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/validation{len(instructions)}/{seed}/normal/plain"
    else:
        if is_cot:
            if holdout_type != "utt":
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/validation{len(instructions)}/{seed}/rewritten/{iter_num}/{fold}/CoT"
            else:
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/validation{len(instructions)}/{seed}/rewritten/{iter_num}/CoT"
        else:
            if holdout_type != "utt":
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/validation{len(instructions)}/{seed}/rewritten/{iter_num}/{fold}/plain"
            else:
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/validation{len(instructions)}/{seed}/rewritten/{iter_num}/plain"

    if iter_num is None:
        save_result_path = f"{dpath}/prompts{prompt_len}_{size}_{seed}_plain.json"
        output_path = f"{dpath}/prompts{prompt_len}_model_output_{size}_{seed}_plain.txt"
        parsed_output_path = f"{dpath}/prompts{prompt_len}_parsed_output_{size}_{seed}_plain.txt"
        incorrects_path = f"{dpath}/prompts{prompt_len}_incorrects_{size}_{seed}_plain.txt"
    else:
        save_result_path = f"{dpath}/prompts{prompt_len}_{size}_{seed}_rewritten_{iter_num}.json"
        output_path = f"{dpath}/prompts{prompt_len}_model_output_{size}_{seed}_rewritten_{iter_num}.txt"
        parsed_output_path = f"{dpath}/prompts{prompt_len}_parsed_output_{size}_{seed}_rewritten_{iter_num}.txt"
        incorrects_path = f"{dpath}/prompts{prompt_len}_incorrects_{size}_{seed}_rewritten_{iter_num}.txt"
    if not os.path.exists(dpath):
        os.makedirs(f"{dpath}/")

    save_to_file(final_results, save_result_path)

    # store model outputs w/ ground truth
    with open(output_path, 'w') as f:
        for response, true_ltl in zip(responses, ground_truths):
            f.write(response + ', ' + true_ltl + '\n')

    # store parsed model outputs w/ ground truth
    with open(parsed_output_path, 'w') as f:
        for response, true_ltl in zip(parsed_responses, ground_truths):
            f.write(response + ', ' + true_ltl + '\n')

    with open(incorrects_path, 'w') as f:
        # iterate over the list of tuples and write each tuple as a line in the file
        # want --
        # Model output, Ground truth, LTL type and num propositions, English Instruction
        line = "Parsed_output, Ground_truth, LTL_type, Num_props, Instruction, Model_output\n\n"
        f.write(line)
        for eval_item in incorrects_eval:
            line = ', '.join(str(item) for item in eval_item) + '\n\n'
            f.write(line)
        f.write(f"Accuracy: {total_accuracy}")


if __name__ == '__main__':
    # use [text-davinci-003]
    models = ['text-davinci-003']

    # Old Code First Iteration
    # plain_prompt_location = f'data/prompts_batch1_noperm/prompt_nexamples1_symbolic_batch1_noperm_ltl_formula_4_123_fold4.txt'
    # plain_prompt_location = f'data/prompts_batch1_noperm/prompt_nexamples1_symbolic_batch1_noperm_ltl_type_2_123_fold1.txt'

    # if is_cot:
    #   with open(f'cot_prompt{prompt_len}.txt', 'r') as file:
    #     prompt = file.read()
    # else:
    #   with open(f'plain_prompt{prompt_len}.txt', 'r') as file:
    #     prompt = file.read()

    # instructions = []
    # ground_truths = []

    # with open(f'test_inputs_{test_input_len}.csv', 'r') as file:
    #     # read the CSV file
    #     reader = csv.reader(file)

    #     # skip the header row
    #     next(reader)

    #     # create a list of all the data points
    #     for row in reader:
    #       instructions.append(row[0])
    #       ground_truths.append(row[1])

    ### CHANGE THESE ###
    ###########################################
    batch_identifier = 1 # 1
    # fold = "fold0" # set to None if not evaluating type or formula holdout
    size = 0.92 # 0.92
    seed = 484 # 484 or 123
    is_cot = False # False or True
    iter_num = "prompts_rewritten" # set to None if not using ChatGPT rewritten datasets

    ### utterance holdout ###
    # fold = None
    # dataset_path = f'data/holdout_batch{batch_identifier}_noperm/symbolic_batch{batch_identifier}_noperm_utt_{size}_{seed}.pkl'

    folds = ["fold0", "fold1"]
    for fold in folds:
        ### type holdout ###
        # fold0 or fold1
        dataset_path = f'data/holdout_batch{batch_identifier}_noperm/symbolic_batch{batch_identifier}_noperm_ltl_type_2_123_{fold}.pkl'

        ### formula holdout ###
        # "fold0", "fold1", "fold2", "fold3", "fold4"
        # dataset_path = f'data/holdout_batch{batch_identifier}_noperm/symbolic_batch{batch_identifier}_noperm_ltl_formula_4_123_{fold}.pkl'

        ###########################################

        with open(dataset_path, 'rb') as file:
            # dictionary with keys ['smaller_valid', 'smaller_valid_meta', 'holdout_type', 'holdout_meta', 'seed', 'size', 'dataset_size']
            dataset_object = pickle.load(file)

        max_tokens = 400
        temperature = 0

        for model_name in models:
            run_experiment(dataset_object, max_tokens, model_name,
                        batch_identifier, is_cot=is_cot, temp=temperature, n=1, fold=fold, iter_num=iter_num)
