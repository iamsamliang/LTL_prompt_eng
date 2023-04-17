import re

if __name__ == '__main__':
    # result = []
    # # open the file for reading
    # with open('data/prompts_batch1_noperm/prompt_nexamples1_symbolic_batch1_noperm_utt.txt', 'r') as f:
    #   for line in f:
    #       if line.startswith("A:"):
    #           ltl_formula = line.split(": ")[1].strip()
    #           infix = prefix_to_infix(ltl_formula)
    #           print(ltl_formula)
    #           print(infix)
    #           print()
    #           result.append(infix)

    # Define a regex pattern to match Q: and A: pairs
    qa_pattern = r"Q: (.*)\nA: (.*)\n"

    # Read in the file
    with open("data/prompts_batch1_noperm/CoT_prompt_nexamples1_symbolic_batch1_noperm_ltl_formula_4_123_fold4.txt", "r") as f:
        text = f.read()

    # Find all the Q: and A: pairs in the text
    qa_pairs = re.findall(qa_pattern, text)

    # Loop through each pair and replace the A: part with the extracted formula
    for q, a in qa_pairs:
        formula_match = re.search(r"The answer is \"(.*)\"\.", a)
        if formula_match:
            formula = formula_match.group(1)
            new_a = formula
            text = text.replace(a, new_a)

    # Write the updated text back to the file
    with open("data/prompts_batch1_noperm/prompt_nexamples1_symbolic_batch1_noperm_ltl_formula_4_123_fold4.txt", "w") as f:
        f.write(text)


    # with open('data/prompts_batch1_noperm/prompt_nexamples1_symbolic_batch1_noperm_utt.txt', "r") as f:
    #     lines = f.readlines()

    # # process each line and write to output file
    # with open('data/prompts_batch1_noperm/prompt_nexamples1_symbolic_batch1_noperm_utt_conv.txt', "w") as f:
    #     for line in lines:
    #         if line.startswith("A:"):
    #             ltl_formula = line.split(": ")[1].strip()
    #             infix_formula = prefix_to_infix(ltl_formula)
    #             f.write("A: " + infix_formula + "\n")
    #         else:
    #             f.write(line)
              