import csv
import random

if __name__ == '__main__':
  random.seed(484)

  num_datapoints = 50

  # extract 25 datapoints from the validation set to be used as the test input for GPT-3
  with open('utt_valid_iter.csv', 'r') as file:
      # read the CSV file
      reader = csv.reader(file)

      # skip the header row
      next(reader)

      # create a list of all the data points
      data = list(reader)

  # use the random.sample function to randomly select 25 data points
  sample = random.sample(data, num_datapoints)

  with open(f"test_inputs_{num_datapoints}.csv", "w", newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["instruction","ltl"])
      writer.writerows(sample)


