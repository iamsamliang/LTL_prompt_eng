import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


model = "text-davinci-003"
dataset_size = "30"


### instruction holdout
holdout_type = "utt"
# Define the directory that contains the two subdirectories
main_dir = f"results/{model}/1/{holdout_type}/validation{dataset_size}"

utt_cot_accs = []
utt_plain_accs = []

# Loop through the two subdirectories and grab each JSON file within each
for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            if filename.endswith('.json'):
                json_file_path = os.path.join(subdir_path, filename)
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                accuracy = data['Accuracy']
                if subdir == 'CoT':
                    utt_cot_accs.append(accuracy)
                else:
                    utt_plain_accs.append(accuracy)

print(utt_cot_accs)
print(utt_plain_accs)
utt_cot_avg_acc = np.mean(utt_cot_accs)
utt_plain_avg_acc = np.mean(utt_plain_accs)
utt_cot_std_dev = np.std(utt_cot_accs)
utt_plain_std_dev = np.std(utt_plain_accs)


### formula holdout
holdout_type = "ltl_formula"
# Define the directory that contains the two subdirectories
main_dir = f"results/{model}/1/{holdout_type}/validation{dataset_size}"

formula_cot_accs = []
formula_plain_accs = []

# Loop through the two subdirectories and grab each JSON file within each
for fold in os.listdir(main_dir):
    fold_path = os.path.join(main_dir, fold)
    print(fold_path)
    for subdir in os.listdir(fold_path):
        subdir_path = os.path.join(fold_path, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.json'):
                    json_file_path = os.path.join(subdir_path, filename)
                    with open(json_file_path, 'r') as f:
                        data = json.load(f)
                    accuracy = data['Accuracy']
                    if subdir == 'CoT':
                        formula_cot_accs.append(accuracy)
                    else:
                        formula_plain_accs.append(accuracy)

print(formula_cot_accs)
print(formula_plain_accs)
formula_cot_avg_acc = np.mean(formula_cot_accs)
formula_plain_avg_acc = np.mean(formula_plain_accs)
formula_cot_std_dev = np.std(formula_cot_accs)
formula_plain_std_dev = np.std(formula_plain_accs)


### type holdout
holdout_type = "ltl_type"
# Define the directory that contains the two subdirectories
main_dir = f"results/{model}/1/{holdout_type}/validation{dataset_size}"

type_cot_accs = []
type_plain_accs = []

# Loop through the two subdirectories and grab each JSON file within each
for fold in os.listdir(main_dir):
    fold_path = os.path.join(main_dir, fold)
    print(fold_path)
    for subdir in os.listdir(fold_path):
        subdir_path = os.path.join(fold_path, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.json'):
                    json_file_path = os.path.join(subdir_path, filename)
                    with open(json_file_path, 'r') as f:
                        data = json.load(f)
                    accuracy = data['Accuracy']
                    if subdir == 'CoT':
                        type_cot_accs.append(accuracy)
                    else:
                        type_plain_accs.append(accuracy)

print(type_cot_accs)
print(type_plain_accs)
type_cot_avg_acc = np.mean(type_cot_accs)
type_plain_avg_acc = np.mean(type_plain_accs)
type_cot_std_dev = np.std(type_cot_accs)
type_plain_std_dev = np.std(type_plain_accs)

# Create a Pandas DataFrame with the accuracy data
df = pd.DataFrame({
    'Plain': [utt_plain_avg_acc, formula_plain_avg_acc, type_plain_avg_acc],
    'CoT': [utt_cot_avg_acc, formula_cot_avg_acc, type_cot_avg_acc],
    'Holdout Test': ['Instruction Holdout', 'Formula Holdout', 'Type Holdout']
})

# Melt the DataFrame so that the accuracy values for Method 1 and Method 2 are in the same column
df_melted = df.melt(id_vars=['Holdout Test'], var_name='Method', value_name='Accuracy')
std_dev = df_melted.groupby('Method')['Accuracy'].apply(np.std).reset_index()

colors = ['#4169E1', '#d62728']
# Create a grouped bar chart using seaborn
sns.set_style('white')
sns.set_palette(colors)
sns.barplot(x='Holdout Test', y='Accuracy', hue='Method', data=df_melted)
# sns.despine()

ax = plt.gca()
# remove the x-axis label
ax.set_xlabel(None)
                                      
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.title('GPT-3 Accuracy on Holdout Tests')
plt.legend(title='Prompting Method')
plt.show()