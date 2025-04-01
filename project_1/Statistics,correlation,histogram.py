import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('/Users/ehabal-saoudi/Downloads/concrete+compressive+strength/Concrete_Data.xls')

#Summary statistics
data_summary = {
    "Attribut": [
        "Blast Furnace Slag (component 2) (kg in a m^3 mixture)",
        "Cement (component 1) (kg in a m^3 mixture)",
        "Fly Ash (component 3) (kg in a m^3 mixture)",
        "Water (component 4) (kg in a m^3 mixture)",
        "Superplasticizer (component 5) (kg in a m^3 mixture)",
        "Coarse Aggregate (component 6) (kg in a m^3 mixture)",
        "Fine Aggregate (component 7) (kg in a m^3 mixture)",
        "Age (day)"
    ],
    "Mean": [
        data["Blast Furnace Slag (component 2)(kg in a m^3 mixture)"].mean(),
        data["Cement (component 1)(kg in a m^3 mixture)"].mean(),
        data["Fly Ash (component 3)(kg in a m^3 mixture)"].mean(),
        data["Water  (component 4)(kg in a m^3 mixture)"].mean(),
        data["Superplasticizer (component 5)(kg in a m^3 mixture)"].mean(),
        data["Coarse Aggregate  (component 6)(kg in a m^3 mixture)"].mean(),
        data["Fine Aggregate (component 7)(kg in a m^3 mixture)"].mean(),
        data["Age (day)"].mean()
    ],
    "Median": [
        data["Blast Furnace Slag (component 2)(kg in a m^3 mixture)"].median(),
        data["Cement (component 1)(kg in a m^3 mixture)"].median(),
        data["Fly Ash (component 3)(kg in a m^3 mixture)"].median(),
        data["Water  (component 4)(kg in a m^3 mixture)"].median(),
        data["Superplasticizer (component 5)(kg in a m^3 mixture)"].median(),
        data["Coarse Aggregate  (component 6)(kg in a m^3 mixture)"].median(),
        data["Fine Aggregate (component 7)(kg in a m^3 mixture)"].median(),
        data["Age (day)"].median()
    ],
    "Standard Deviation": [
        data["Blast Furnace Slag (component 2)(kg in a m^3 mixture)"].std(),
        data["Cement (component 1)(kg in a m^3 mixture)"].std(),
        data["Fly Ash (component 3)(kg in a m^3 mixture)"].std(),
        data["Water  (component 4)(kg in a m^3 mixture)"].std(),
        data["Superplasticizer (component 5)(kg in a m^3 mixture)"].std(),
        data["Coarse Aggregate  (component 6)(kg in a m^3 mixture)"].std(),
        data["Fine Aggregate (component 7)(kg in a m^3 mixture)"].std(),
        data["Age (day)"].std()
    ],
    "Minimum": [
        data["Blast Furnace Slag (component 2)(kg in a m^3 mixture)"].min(),
        data["Cement (component 1)(kg in a m^3 mixture)"].min(),
        data["Fly Ash (component 3)(kg in a m^3 mixture)"].min(),
        data["Water  (component 4)(kg in a m^3 mixture)"].min(),
        data["Superplasticizer (component 5)(kg in a m^3 mixture)"].min(),
        data["Coarse Aggregate  (component 6)(kg in a m^3 mixture)"].min(),
        data["Fine Aggregate (component 7)(kg in a m^3 mixture)"].min(),
        data["Age (day)"].min()
    ],
    "Maximum": [
        data["Blast Furnace Slag (component 2)(kg in a m^3 mixture)"].max(),
        data["Cement (component 1)(kg in a m^3 mixture)"].max(),
        data["Fly Ash (component 3)(kg in a m^3 mixture)"].max(),
        data["Water  (component 4)(kg in a m^3 mixture)"].max(),
        data["Superplasticizer (component 5)(kg in a m^3 mixture)"].max(),
        data["Coarse Aggregate  (component 6)(kg in a m^3 mixture)"].max(),
        data["Fine Aggregate (component 7)(kg in a m^3 mixture)"].max(),
        data["Age (day)"].max()
    ],
    "Count": [
        data["Blast Furnace Slag (component 2)(kg in a m^3 mixture)"].count(),
        data["Cement (component 1)(kg in a m^3 mixture)"].count(),
        data["Fly Ash (component 3)(kg in a m^3 mixture)"].count(),
        data["Water  (component 4)(kg in a m^3 mixture)"].count(),
        data["Superplasticizer (component 5)(kg in a m^3 mixture)"].count(),
        data["Coarse Aggregate  (component 6)(kg in a m^3 mixture)"].count(),
        data["Fine Aggregate (component 7)(kg in a m^3 mixture)"].count(),
        data["Age (day)"].count()
    ]
}

df_summary = pd.DataFrame(data_summary)
print(df_summary)
print(data["Concrete compressive strength(MPa, megapascals) "].std())
print(data["Concrete compressive strength(MPa, megapascals) "].mean())
print(data["Concrete compressive strength(MPa, megapascals) "].median())
print(data["Concrete compressive strength(MPa, megapascals) "].min())
print(data["Concrete compressive strength(MPa, megapascals) "].max())

# Korrelationsmatrixen med et heatmap
correlation_matrix = data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Korrelationsmatrix af attributter')
plt.show()

#Scatterplot matrix
data.columns = data.columns.str.replace(r'\s*\(.*\)', '', regex=True) #Fjerne enheder fra navnet

scaler = StandardScaler()
standardized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

sns.set(style="whitegrid", rc={"axes.labelsize": 16, "axes.titlesize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14})

pairplot = sns.pairplot(standardized_data, 
                         diag_kind="kde", 
                         plot_kws={'alpha': 0.5, 's': 10, 'edgecolor': 'none'},  
                         diag_kws={'color': 'navy'})  

plt.show()

# Standardiser dataene
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Boxplots
plt.figure(figsize=(12, 8))
sns.boxplot(data=data_standardized)

plt.title('Boxplot for hver standardiseret attribut')
plt.xlabel('Attributter')
plt.ylabel('Standardiserede værdier')

plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()

#Hisogrammer
data = pd.read_excel('/Users/ehabal-saoudi/Downloads/concrete+compressive+strength/Concrete_Data.xls')

attributes = [
    "Cement (component 1)(kg in a m^3 mixture)",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
    "Fly Ash (component 3)(kg in a m^3 mixture)",
    "Water  (component 4)(kg in a m^3 mixture)",
    "Superplasticizer (component 5)(kg in a m^3 mixture)",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)",
    "Age (day)"
]

fig, axes = plt.subplots(2, 4, figsize=(15, 8))  
axes = axes.flatten()  

for i, attr in enumerate(attributes):
    axes[i].hist(data[attr], bins=30, edgecolor='black')
    
    short_name = attr.split(" (")[0]
    
    axes[i].set_xlabel(short_name)  
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()