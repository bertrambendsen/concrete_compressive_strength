# This code has been written using GitHub CoPilot for assisting in the completion of the project.
# The code is part of Project 1 for the course Introduction to Machine Learning and Data Mining.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xlsxwriter
import seaborn as sns

df = pd.read_excel("data\Concrete_Data.xls")
input_data = df.iloc[:, :-1] # Excluding the lastr column as it's the output
compressive_strength = df.iloc[:, -1] # Loading last column as compressive strength
scale = StandardScaler() 
input_data = scale.fit_transform(input_data) # Centering and scaling the data


# Applying PCA to our data
pca = PCA()
input_pca = pca.fit_transform(input_data)
exvar = pca.explained_variance_ratio_

# Define intervals for Low, High
Low = (2.33, 34.44) # 
High = (34.44, 80.2) # 

# Categorize compressive strength
def categorize_strength(strength):
    if strength < Low[1]:
        return 'Low'
    else:
        return 'High'

correct_col_name = 'Concrete compressive strength(MPa, megapascals)'

# Applying Principal components
pc_df = pd.DataFrame(input_pca, columns=[f'PC{i+1}' for i in range(input_pca.shape[1])])
pc_df[correct_col_name] = compressive_strength.values
pc_df['Strength Category'] = compressive_strength.apply(categorize_strength)

# Define colors for categories
palette = {'Low': 'red', 'High': 'green'}

# Plot pairwise 2D projections with color-coding
pairplot = sns.pairplot(
    pc_df,
    vars=[f'PC{i+1}' for i in range(8)],  # All 8 PCs
    hue='Strength Category',
    palette=palette,
    plot_kws={'alpha': 0.7, 's': 20, 'edgecolor': 'k'},  # Smaller points (`s`)
    diag_kind='kde',
    height=2.0,  # Default is 2.5
    aspect=0.8,  # Adjust width-to-height ratio (default 1)
)
pairplot.fig.suptitle("Pairwise 2D Projections of Principal Components", y=1.02)
plt.tight_layout()
plt.show()


# Calculating the cumulative explained variance
cumulative_exvar = np.zeros(len(exvar))
cumulative_exvar[0] = exvar[0]
for i in range(1, len(exvar)):
    cumulative_exvar[i] = cumulative_exvar[i-1] + exvar[i]

plt.figure()

# Defining the 3 graphs
plt.plot(range(1, len(exvar) + 1), exvar, "x-")
plt.plot(range(1, len(cumulative_exvar) + 1), cumulative_exvar, "o-")
plt.plot(range(1, len(exvar) + 1), np.ones(len(exvar)) * 0.9, "--")

plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "90% Explained"])

plt.grid()
plt.show()

pc_df = pd.DataFrame(input_pca, columns=[f'PC{i+1}' for i in range(8)])
pc_df['Compressive Strength'] = compressive_strength

for i, v in enumerate(exvar):
    print(f"Principal component {i+1}: {v:.4f} ({v*100:.2f}%) variance explained")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))  # 2 rows, 4 columns

