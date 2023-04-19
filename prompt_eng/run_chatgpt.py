import time
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')
from evaluation import evaluate_lang_0
import logging
import time
import os
import sys
import openai
import csv
from utils import save_to_file
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import pickle
from models_util import chatgpt_complete
import re

openai.api_key = "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"
openai.organization = "org-g3Ul3oNaC7RRJGYw9fy78mdi"

def run_experiment(dataset_object, max_tokens, model_name, batch_identifier, is_cot, temp=0, n=1, fold=None, iter_num=None):
    # smaller datasets
    # test_dataset = dataset_object.get('smaller_valid')
    # validation_meta = dataset_object.get('smaller_valid_meta')

    # # full datasets
    # test_dataset = dataset_object.get('valid_iter')

    # rewritten_dataset = rewrite_instruc(test_dataset, max_tokens, model_name, temp, n)
    # with open(f'ltl_formula_rewritten_{fold}.pkl', 'wb') as f:
    #     pickle.dump(rewritten_dataset, f) # [(rewritten_instruction, ground_truth)]
    # return

    holdout_type = dataset_object.get('holdout_type')

    # full datasets
    if iter_num is None:
        test_dataset = dataset_object.get('valid_iter')
    else:
        # rewritten instructions for type/formula holdout
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

    # Zero shot
    # plain_prompt_location = f'data/prompts_batch{batch_identifier}_noperm/zeroshot_prompt_nexamples1_symbolic_batch{batch_identifier}_noperm_{holdout_type}_{size}_{seed}_{fold}.txt'
    # with open(plain_prompt_location, 'r') as file:
    #   prompt = file.read()

    if is_cot:
        if holdout_type != "utt":
            cot_prompt_location = f'data/prompts_batch{batch_identifier}_noperm/CoT_prompt_nexamples1_symbolic_batch{batch_identifier}_noperm_{holdout_type}_{size}_{seed}_{fold}.txt'
        else:
            cot_prompt_location = f'data/prompts_batch{batch_identifier}_noperm/CoT_prompt_nexamples1_symbolic_batch{batch_identifier}_noperm_{holdout_type}.txt'
        message = create_message(cot_prompt_location)
    else:
        if holdout_type != "utt":
            plain_prompt_location = f'data/prompts_batch{batch_identifier}_noperm/prompt_nexamples1_symbolic_batch{batch_identifier}_noperm_{holdout_type}_{size}_{seed}_{fold}.txt'
        else:
            plain_prompt_location = f'data/prompts_batch{batch_identifier}_noperm/prompt_nexamples1_symbolic_batch{batch_identifier}_noperm_{holdout_type}.txt'
        message = create_message(plain_prompt_location)

    if is_cot:
        if holdout_type == "utt":
            prompt_len = len(message) // 2
        elif holdout_type == "ltl_formula":
            prompt_len = len(message) // 2
        elif holdout_type == "ltl_type":
            prompt_len = len(message) // 2
    else:
        if holdout_type == "utt":
            prompt_len = len(message) // 2
        elif holdout_type == "ltl_formula":
            prompt_len = len(message) // 2
        elif holdout_type == "ltl_type":
            prompt_len = len(message) // 2

    responses = []
    parsed_responses = []
    ground_truths = []
    instructions = []
    for instruction, ground_truth in test_dataset:
        instruction = instruction.strip()
        ground_truths.append(ground_truth)
        instructions.append(instruction)

        # CoT and Plain Prompts
        new_message = message.copy()
        new_message.append({"role": "user", "content": f'What is "{instruction}" in LTL?'})
        
        raw_response = chatgpt_complete(new_message, max_tokens, model_name, temp, n)
        if raw_response is None:
            continue
        
        response = raw_response['choices'][0]['message']['content']
        response = response.strip()

        print("chatgpt's response: ", response)
        print()

        responses.append(response)

        # this needs refinement. Sometimes, chatgpt responds in a different way like "The answer is simply ..." instead of "The answer is ..."
        if is_cot:
            # grab only the ltl from the model output
            res = response.split(".")
            if res[-1] != "":
                text = res[-1]
            else:
                text = res[-2]
            first_quote = text.find('is')
            parsed_response = text[first_quote+4:len(text)-1]
            print("chatgpt's parsed response: ", parsed_response)
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
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/{seed}/normal/{fold}/size{len(instructions)}/CoT"
            else:
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/{seed}/normal/size{len(instructions)}/CoT"
        else:
            if holdout_type != "utt":
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/{seed}/normal/{fold}/size{len(instructions)}/plain"
            else:
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/{seed}/normal/size{len(instructions)}/plain"
    else:
        if is_cot:
            if holdout_type != "utt":
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/{seed}/rewritten/{iter_num}/{fold}/size{len(instructions)}/CoT"
            else:
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/{seed}/rewritten/{iter_num}/size{len(instructions)}/CoT"
        else:
            if holdout_type != "utt":
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/{seed}/rewritten/{iter_num}/{fold}/size{len(instructions)}/plain"
            else:
                dpath = f"results/{model_name}/{batch_identifier}/{holdout_type}/{seed}/rewritten/{iter_num}/size{len(instructions)}/plain"

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

def create_message(prompt_txt):
  message = []
  message.append({"role": "system", "content": "You are a semantic parser that translates English instructions to linear temporal logic (LTL) formulas."})

  with open(prompt_txt, 'r') as f:
    lines = f.readlines()

  # don't want start nor ending line 2 lines (which are '\n' and 'Q: ')
  for i in range(2, len(lines) - 2, 3):
    q = lines[i].strip()[3:]  # remove "Q: " from the beginning of the line. Strip() removes '\n' as well
    a = lines[i+1].strip()[3:]  # remove "A: " from the beginning of the line
    message.append({"role": "user", "content": q})
    message.append({"role": "assistant", "content": a})

  return message

def rewrite_instruc(test_dataset, max_tokens, model_name, temp, n):
    message = []
    # message.append({"role": "system", "content": "I will be translating English instructions to linear temporal logic formulas. However, these instructions can be unclear and ambiguous. I want you to rewrite them so that they are clear and can be more easily translated to linear temporal logic formulas."})
    
    # iteration 5
    # message.append({"role": "system", "content": "You will rewrite English instructions, if necessary, that I provide into clearer English instructions so that they can be more easily translated to linear temporal logic formulas. You must use simple verbs. For example, use \"visit\" instead of \"go to\" or \"stop by\".  You should use temporal connectives like \"eventually\", \"then\", \"always\", and \"until\" to specify temporal relationships between events. For example, use \"eventually\" instead of \"at some point in time\". \"a\", \"b\", \"c\", \"d\", and \"h\" are all landmarks. Write the response in the following format: Instruction: {response}. {response} should be replaced with the actual response you want to give."})

    # iteration 6
    message.append({"role": "system", "content": "You will rewrite English instructions that I provide into clearer English instructions, if necessary, so that they can be more easily translated to linear temporal logic formulas. You must use simple verbs. For example, use \"visit\" instead of \"go to\" or \"stop by\".  When necessary, you can use temporal connectives like \"eventually\", \"then\", \"always\", and \"until\" to specify temporal relationships between events. For example, use \"eventually\" instead of \"at some point in time\". \"a\", \"b\", \"c\", \"d\", and \"h\" are all landmarks. Write the response in the following format: Instruction: {response}. {response} should be replaced with the actual response you want to give."})
    # print(message)
    # response = complete(message, max_tokens, model_name, temp, n)
    # response = response['choices'][0]['message']['content']
    # response = response.strip()
    # print("initial: ", response)
    # print()
    # return
    # message.append({"role": "assistant", "content": response})

    rewritten_test = []
    for instruction, ground_truth in test_dataset:
      first_if = False
      new_message = message.copy()
      instruction = instruction.strip()
      new_message.append({"role": "user", "content": f'{instruction}'})
      print("Beginning of for loop. Passing in the message: ", new_message)
      print()

      response = chatgpt_complete(new_message, max_tokens, model_name, temp, n)
      response = response['choices'][0]['message']['content']
      response = response.strip()
      print("ChatGPT response: ", response)

      confusion_matrix = ["more context", "unclear", "cannot be translated", "clarify", "does not seem to be a valid", "understand", "information"]
      for word in confusion_matrix:
        if word in response:
            print("in here")
            rewritten_test.append((instruction, ground_truth))
            continue
      
      if "Instruction: " not in response:
          first_if = True
          print()
          print("'Instruction: ' not present. First if statement")
          new_message.append({"role": "assistant", "content": response})
          new_message.append({"role": "user", "content": "Your response was not in the correct format that I requested. Write the response in the following format: Instruction: {response}. {response} should be replaced with the actual response you want to give. Give me your previous response in this format."})
          response = chatgpt_complete(new_message, max_tokens, model_name, temp, n)
          response = response['choices'][0]['message']['content']
          response = response.strip()
          print("second response: ", response)
          print()

      if "Instruction: " not in response:
          result = instruction
      else:
        #   if first_if:
        #     new_message.append({"role": "assistant", "content": response})
        #     message = new_message.copy()
          result = response[response.index("Instruction: ")+len("Instruction: "):]
          print("result: ", result)
          print()
          print()

      rewritten_test.append((result, ground_truth))

    return rewritten_test

def extract_sentences(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        pattern = re.compile(r'"([^"]*)"')
        sentences = pattern.findall(content)
        return sentences

def rewrite_prompts(rewritten_sentences, file_path, output_file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    print(len(rewritten_sentences))
    counter = 0
    for i, line in enumerate(lines):
        if line.startswith("Q:") and i != len(lines) - 1:
            quote_start = line.index('"')
            quote_end = line.index('"', quote_start + 1)
            original_sentence = line[quote_start+1:quote_end]
            new_sentence = rewritten_sentences[counter]
            counter += 1
            lines[i] = line.replace(original_sentence, new_sentence)

    with open(output_file_path, "w") as f:
        f.writelines(lines)

def rewrite_prompts_(prompts, max_tokens, model_name, temp, n):
    message = []

    # iteration 6
    message.append({"role": "system", "content": "You will rewrite English instructions that I provide into clearer English instructions, if necessary, so that they can be more easily translated to linear temporal logic formulas. You must use simple verbs. For example, use \"visit\" instead of \"go to\" or \"stop by\".  When necessary, you can use temporal connectives like \"eventually\", \"then\", \"always\", and \"until\" to specify temporal relationships between events. For example, use \"eventually\" instead of \"at some point in time\". \"a\", \"b\", \"c\", \"d\", and \"h\" are all landmarks. Write the response in the following format: Instruction: {response}. {response} should be replaced with the actual response you want to give."})

    rewritten_prompts = []
    for prompt in prompts:
      first_if = False
      new_message = message.copy()
      instruction = prompt.strip()
      new_message.append({"role": "user", "content": f'{instruction}'})
      print("Beginning of for loop. Passing in the message: ", new_message)
      print()

      response = chatgpt_complete(new_message, max_tokens, model_name, temp, n)
      response = response['choices'][0]['message']['content']
      response = response.strip()
      print("ChatGPT response: ", response)

      confusion_matrix = ["more context", "unclear", "cannot be translated", "clarify", "does not seem to be a valid", "understand", "information"]
      for word in confusion_matrix:
        if word in response:
            print("in here")
            rewrite_prompts.append(instruction)
            continue
      
      if "Instruction: " not in response:
          first_if = True
          print()
          print("'Instruction: ' not present. First if statement")
          new_message.append({"role": "assistant", "content": response})
          new_message.append({"role": "user", "content": "Your response was not in the correct format that I requested. Write the response in the following format: Instruction: {response}. {response} should be replaced with the actual response you want to give. Give me your previous response in this format."})
          response = chatgpt_complete(new_message, max_tokens, model_name, temp, n)
          response = response['choices'][0]['message']['content']
          response = response.strip()
          print("second response: ", response)
          print()

      if "Instruction: " not in response:
          result = instruction
      else:
          result = response[response.index("Instruction: ")+len("Instruction: "):]
          print("result: ", result)
          print()
          print()

      rewritten_prompts.append(result)

    return rewritten_prompts


if __name__ == '__main__':
    # use [gpt-3.5-turbo]
    models = ['gpt-3.5-turbo']

    ### CHANGE THESE ###
    ###########################################
    batch_identifier = 1
    #   fold = "fold1"
    size = 0.92
    seed = 123 # 484 or 123
    is_cot = False
    iter_num = "iter1"

    ### utterance holdout ###
    fold = None
    dataset_path = f'data/holdout_batch{batch_identifier}_noperm/symbolic_batch{batch_identifier}_noperm_utt_{size}_{seed}.pkl'

    # folds = ["fold0", "fold1", "fold2", "fold3", "fold4"]
    # folds = ["fold0", "fold1"]
    # for fold in folds:
    #     ### formula holdout ###
    #     # dataset_path = f'data/holdout_batch{batch_identifier}_noperm/symbolic_batch{batch_identifier}_noperm_ltl_formula_4_123_{fold}.pkl'

    #     ### type holdout ###
    #     # fold0 and fold1
    #     dataset_path = f'data/holdout_batch{batch_identifier}_noperm/symbolic_batch{batch_identifier}_noperm_ltl_type_2_123_{fold}.pkl'

    #     # ###########################################

    with open(dataset_path, 'rb') as file:
        # dictionary with keys ['smaller_valid', 'smaller_valid_meta', 'holdout_type', 'holdout_meta', 'seed', 'size', 'dataset_size']
        dataset_object = pickle.load(file)

    max_tokens = 400
    temperature = 0.1

    for model_name in models:
        run_experiment(dataset_object, max_tokens, model_name,
                        batch_identifier, is_cot=is_cot, temp=temperature, n=1, fold=fold, iter_num=iter_num)
