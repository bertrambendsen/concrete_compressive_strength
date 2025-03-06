import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('/Users/ehabal-saoudi/Downloads/concrete+compressive+strength/Concrete_Data.xls')

# Opret en dictionary med attributterne og deres summary statistics
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

# Konverter til en DataFrame og print den
df_summary = pd.DataFrame(data_summary)
print(df_summary)
print(data["Concrete compressive strength(MPa, megapascals) "].std())
print(data["Concrete compressive strength(MPa, megapascals) "].mean())
print(data["Concrete compressive strength(MPa, megapascals) "].median())
print(data["Concrete compressive strength(MPa, megapascals) "].min())
print(data["Concrete compressive strength(MPa, megapascals) "].max())

# Beregn korrelationen og kovariansen for hvert attribut med "Concrete compressive strength(MPa, megapascals) "
correlations = {
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
    "Correlation with Concrete Strength": [
        data["Blast Furnace Slag (component 2)(kg in a m^3 mixture)"].corr(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Cement (component 1)(kg in a m^3 mixture)"].corr(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Fly Ash (component 3)(kg in a m^3 mixture)"].corr(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Water  (component 4)(kg in a m^3 mixture)"].corr(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Superplasticizer (component 5)(kg in a m^3 mixture)"].corr(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Coarse Aggregate  (component 6)(kg in a m^3 mixture)"].corr(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Fine Aggregate (component 7)(kg in a m^3 mixture)"].corr(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Age (day)"].corr(data["Concrete compressive strength(MPa, megapascals) "])
    ],
    "Covariance with Concrete Strength": [
        data["Blast Furnace Slag (component 2)(kg in a m^3 mixture)"].cov(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Cement (component 1)(kg in a m^3 mixture)"].cov(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Fly Ash (component 3)(kg in a m^3 mixture)"].cov(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Water  (component 4)(kg in a m^3 mixture)"].cov(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Superplasticizer (component 5)(kg in a m^3 mixture)"].cov(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Coarse Aggregate  (component 6)(kg in a m^3 mixture)"].cov(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Fine Aggregate (component 7)(kg in a m^3 mixture)"].cov(data["Concrete compressive strength(MPa, megapascals) "]),
        data["Age (day)"].cov(data["Concrete compressive strength(MPa, megapascals) "])
    ]
}

# Konverter til en DataFrame og print den
df_correlation_covariance = pd.DataFrame(correlations)
print(df_correlation_covariance)


# Tjek de første par rækker i datasættet for at forstå strukturen
print(data.head())

plt.figure(figsize=(12, 8))
sns.boxplot(data=data)

plt.title('Boxplot for hver attribut')
plt.xlabel('Attributter')
plt.ylabel('Værdier')

plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()

outliers = {}

for column in data.columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers[column] = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

for column, outlier_data in outliers.items():
    if not outlier_data.empty:
        print(f"Ekstreme værdier for {column}:")
        print(outlier_data)
    else:
        print(f"Der er ingen ekstreme værdier for {column}.")



