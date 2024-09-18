import matplotlib.pyplot as plt
import pandas as pd

# Prepare the data
data = {
    'Metric': ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'VGGish Categorical (Model A)': [0.230, 0.832, 0.860, 0.329, 0.812],
    'VGGish Binary (Model D)': [0.158, 0.261, 0.684, 0.819, 0.581],
    'Perch Categorical (Model B)': [0.893, 0.895, 0.261, 0.458, 0.540],
    'Perch Binary (Model E)': [0.788, 0.747, 0.175, 0.820, 0.419],
    'YAMNet Categorical (Model C)': [0.488, 0.613, 0.367, 0.637, 0.488],
    'YAMNet Binary (Model F)': [0.647, 0.686, 0.883, 0.707, 0.889]
}

df = pd.DataFrame(data)

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))

df.set_index('Metric').plot(kind='bar', ax=ax, color=['lightgreen', 'darkgreen', 'lightblue', 'darkblue', 'lightcoral', 'coral'])

# Adding labels and title
plt.xlabel('Metric', fontweight='bold')
plt.ylabel('Levene Test P-value', fontweight='bold')
plt.title('Levene Test P-values for Models A-F Across Metrics', fontweight='bold')
plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
plt.legend(title='Model')
plt.grid(axis='y')

# Set x-axis labels to horizontal
ax.set_xticklabels(df['Metric'], rotation=0)

# Show the plot
plt.show()
