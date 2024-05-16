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

for i, metric in enumerate(metrics):
    ax = axs[i]
    
    # Plotting the bars with error bars representing std
    baseline_bars = ax.errorbar(index, baseline_means[metric], yerr=baseline_stds[metric], label='Baseline', fmt='o')
    augmented_bars = ax.errorbar(index + bar_width, augmented_means[metric], yerr=augmented_stds[metric], label='Augmented', fmt='o')
    
    # Linking the baseline and augmented means with a line
    for j in range(len(grouped)):
        ax.plot([index[j], index[j] + bar_width], [baseline_means[metric][j], augmented_means[metric][j]], color='gray', linestyle='--')
    
    # Labeling and customization
    ax.set_xlabel('Dataset')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison by Dataset')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(grouped.groups.keys())
    ax.legend()

plt.tight_layout()
# Show plot
plt.savefig('results/metrics_comparison.png')
plt.show()
