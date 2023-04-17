import numpy as np
import spot
import logging

def evaluate_from_file(output_file):
    with open(output_file, 'r') as file:
      # read the lines of the file into a list of strings
      lines = file.readlines()
      # loop through the lines and split each line on a specific character to create tuples
      outputs = [tuple(line.strip().split(',')) for line in lines]

    print(outputs)
    incorrects = []
    accs = []
    for out_ltl, true_ltl in outputs:
        if true_ltl == out_ltl:  # Spot cannot handle long but correct LTL formula, e.g. F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F 62_on_the_park
            is_correct = "True"
        else:
            try:  # output LTL formula may have syntax error
                spot_correct = spot.are_equivalent(spot.formula(true_ltl), spot.formula(out_ltl))
                is_correct = "True" if spot_correct else "False"
            except SyntaxError:
                is_correct = "Syntax Error"
                logging.info(f"Syntax error:\n{true_ltl}\n{out_ltl}\n")
            except TypeError:
                logging.info(f"Type error:\n{true_ltl}\n{out_ltl}\n")
                breakpoint()

        if is_correct == "False":
            incorrects.append((out_ltl, true_ltl))

        accs.append(is_correct)
    acc = np.mean([True if acc == "True" else False for acc in accs])
    return accs, acc, incorrects

if __name__ == '__main__':
    output_path = "results/text-davinci-003/1/utt/validation30/CoT/prompts22_parsed_output_0.2_484.txt"
    _, acc, incorrects = evaluate_from_file(output_path)
    print(incorrects)

    # open the file in write mode
    with open('results/text-davinci-003/1/utt/validation30/CoT/prompts22_incorrects_0.2_484.txt', 'w') as file:
        # iterate over the list of tuples and write each tuple as a line in the file
        for tuple_item in incorrects:
            line = ','.join(str(item) for item in tuple_item) + '\n'
            file.write(line)
        file.write(f"Accuracy: {acc}")
