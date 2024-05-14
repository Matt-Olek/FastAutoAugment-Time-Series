import subprocess
import pandas as pd

'''
This script runs the FastAA/main.py script on all datasets in the UCRArchive_2018 archive that have not been processed yet.
The results are saved in the 'data/logs/results.csv' file.
'''

res_csv = pd.read_csv("data/logs/results.csv")

datasets_names = []

try:
    with open("data/datasets_names.txt", "r") as f:
        for line in f:
            datasets_names.append(line.strip())
except FileNotFoundError:
    print("Error: datasets_names.txt file not found, you can create it by running 'python FastAA/collect_datasets_names.py' first.")
except Exception as e:
    print("An error occurred:", str(e))
        
for dataset_name in datasets_names:
    if dataset_name not in res_csv["dataset"].values:
        subprocess.run(["python", "FastAA/main.py", "--dataset", dataset_name, "--compare", "--runs", "5"])
