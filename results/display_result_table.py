import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_path = 'data/logs/results.csv'
df = pd.read_csv(results_path)

# Group the data by dataset
grouped = df.groupby('dataset')

# Set the width of the bars
bar_width = 0.35
index = np.arange(len(grouped))

# Initialize the plot
fig, axs = plt.subplots(3, 1, figsize=(35, 15))

metrics = ['accuracy', 'f1', 'recall']
baseline_means = {'accuracy': [], 'f1': [], 'recall': []}
baseline_stds = {'accuracy': [], 'f1': [], 'recall': []}
augmented_means = {'accuracy': [], 'f1': [], 'recall': []}
augmented_stds = {'accuracy': [], 'f1': [], 'recall': []}

for name, group in grouped:
    baseline = group[group['type'] == 'baseline']
    augmented = group[group['type'] == 'augmented']
    
    for metric in metrics:
        if not baseline.empty:
            baseline_means[metric].append(baseline[f'{metric}_mean'].iloc[0])
            baseline_stds[metric].append(baseline[f'{metric}_std'].iloc[0])
        else:
            baseline_means[metric].append(np.nan)
            baseline_stds[metric].append(np.nan)
        
        if not augmented.empty:
            augmented_means[metric].append(augmented[f'{metric}_mean'].iloc[0])
            augmented_stds[metric].append(augmented[f'{metric}_std'].iloc[0])
        else:
            augmented_means[metric].append(np.nan)
            augmented_stds[metric].append(np.nan)


table_data = []
for name, group in grouped:
    baseline = group[group['type'] == 'baseline']
    augmented = group[group['type'] == 'augmented']
    
    row = [name]
    for metric in metrics:
        if not baseline.empty:
            baseline_mean = round(baseline[f'{metric}_mean'].iloc[0], 2)
            baseline_std = round(baseline[f'{metric}_std'].iloc[0], 2)
            row.append(f'{baseline_mean} ({baseline_std})')
        else:
            row.append('N/A')
        
        if not augmented.empty:
            augmented_mean = round(augmented[f'{metric}_mean'].iloc[0], 2)
            augmented_std = round(augmented[f'{metric}_std'].iloc[0], 2)
            augmentation_percentage = round(((augmented_mean - baseline_mean) / baseline_mean) * 100, 2)
            if augmentation_percentage > 0:
                row.append(f'🟢 {augmented_mean} ({augmented_std}) [+{augmentation_percentage}%] ')
            elif augmentation_percentage < 0:
                row.append(f'🔴 {augmented_mean} ({augmented_std}) [{augmentation_percentage}%] ')
            else:
                row.append(f'⚪ {augmented_mean} ({augmented_std}) [{augmentation_percentage}%] ')
        else:
            row.append('N/A')
    
    if not baseline.empty and not augmented.empty:
        baseline_accuracy = baseline['accuracy_mean'].iloc[0]
        augmented_accuracy = augmented['accuracy_mean'].iloc[0]
        successful_augmentation_accuracy = baseline_accuracy < augmented_accuracy
        
        baseline_f1 = baseline['f1_mean'].iloc[0]
        augmented_f1 = augmented['f1_mean'].iloc[0]
        successful_augmentation_f1 = baseline_f1 < augmented_f1
        
        baseline_recall = baseline['recall_mean'].iloc[0]
        augmented_recall = augmented['recall_mean'].iloc[0]
        successful_augmentation_recall = baseline_recall < augmented_recall
        
        if successful_augmentation_accuracy or successful_augmentation_f1 or successful_augmentation_recall:
            row.append('🟢')
        else :
            row.append('🔴')
    else:
        row.append('N/A')
    
    table_data.append(row)

table_columns = ['Dataset']
for metric in metrics:
    table_columns.append(f'Baseline {metric.capitalize()}')
    table_columns.append(f'Augmented {metric.capitalize()} + Aug%')
table_columns.append('Successful Augmentation')

table_df = pd.DataFrame(table_data, columns=table_columns)
table_df.to_markdown('Table.md', index=False)
