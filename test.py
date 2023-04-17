import pickle

if __name__ == "__main__":
  # Load the array from the file
  with open('my_array.pkl', 'rb') as f:
      loaded_arr = pickle.load(f)

  # Print the loaded array
  print(loaded_arr)