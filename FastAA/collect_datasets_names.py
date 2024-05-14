import os

'''
Collects the names of the datasets in the UCR Archive 2018 and saves them in a file.
'''

datasets_names = []
for dataset_name in os.listdir("data/UCRArchive_2018/"):
    datasets_names.append(dataset_name)

# Save the list in a file
with open("data/datasets_names.txt", "w") as f:
    for dataset_name in datasets_names:
        f.write(dataset_name + "\n")

    
    