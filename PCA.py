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
#print(pca.components_)
exvar = pca.explained_variance_ratio_

# Writing to xls file
# workbook = xlsxwriter.Workbook('data\Principal_componcents.xls')
# worksheet = workbook.add_worksheet()

# row = column = 0
# for i in pca.components_:
#     row = 0 
#     worksheet.write(row, column,f'PC{column + 1}')
#     for j in i:
#         row += 1
#         worksheet.write(row, column, j)
#     column += 1
# workbook.close()

pc_df = pd.DataFrame(input_data, columns=[f'PC{i+1}' for i in range(input_pca.shape[1])])
PC_data = pd.read_excel('Principal_componcents.xls')  # Corrected spelling

# Define intervals for Low, Medium, High
Low = (2.33, 30.96)
Medium = (30.96, 39.05)
High = (39.05, 80.2)

# Categorize compressive strength
def categorize_strength(strength):
    if strength <= Low[1]:
        return 'Low'
    elif strength <= Medium[1]:
        return 'Medium'
    else:
        return 'High'

pc_df['Strength Category'] = compressive_strength.apply(categorize_strength)

# Define colors for categories
palette = {'Low': 'red', 'Medium': 'orange', 'High': 'green'}

# Plot pairwise 2D projections with color-coding
sns.set(style="whitegrid", font_scale=1.2)
pairplot = sns.pairplot(
    pc_df,
    vars=pc_df.columns[:-1],  # Exclude 'Strength Category' from axes
    hue='Strength Category',
    palette=palette,
    plot_kws={'alpha': 0.7, 's': 30, 'edgecolor': 'k'},
    diag_kind='kde'
)

pairplot.fig.suptitle("Pairwise 2D Projections of PCs (Colored by Strength)", y=1.02)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for i, ax in enumerate(axes.flat):
    sns.scatterplot(
        data=pc_df,
        x=f'PC{i+1}',
        y='Compressive Strength',
        hue='Strength Category',
        palette=palette,
        alpha=0.7,
        ax=ax
    )
    ax.set_title(f'PC{i+1} vs. Compressive Strength')
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

# Plots all principal components against compressive strength
for i, ax in enumerate(axes.flat):
    ax.scatter(pc_df[f'PC{i+1}'], compressive_strength, alpha=0.5)
    ax.set_xlabel(f'PC{i+1}')
    ax.set_ylabel('Compressive Strength')
    ax.set_title(f'PC{i+1} vs. Compressive Strength')

plt.tight_layout()
plt.show()

""" 2D projection onto principal components"""




# """ 2D projection onto principal components"""
# PC_x = 3
# PC_y = 5
# PC_z = 6

# X_2d = input_pca[:, :PC_y]  # Select the first x principal components
# color_dimension = df["Concrete compressive strength(MPa, megapascals) "] # Use concrete strength for color coding
# #color_dimension = input_pca[:, 2] # Use the third principal component for color coding

# plt.scatter(X_2d[:, PC_x-1], X_2d[:, PC_y-1], c=color_dimension, cmap='coolwarm', alpha=0.7)
# plt.xlabel(f'Principal Component {PC_x}')
# plt.ylabel(f'Principal Component {PC_y}')
# plt.title(f'2D Projection of Data onto PC{PC_x} and PC{PC_y}')
# plt.show()

# """ 3D projection onto principal components """
# # Project onto the first 3 components
# X_3d = input_pca[:, :PC_z]

# # Plot in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_3d[:, PC_x-1], X_3d[:, PC_y-1], X_3d[:, PC_z-1],c=color_dimension, cmap='coolwarm', alpha=0.7)
# ax.set_xlabel(f'Principal Component {PC_x}')
# ax.set_ylabel(f'Principal Component {PC_y}')
# ax.set_zlabel(f'Principal Component {PC_z}')
# plt.title(f'3D Projection of Data onto PC{PC_x}, PC{PC_y} and PC{PC_z}')
# plt.show()
