import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_excel("data/Concrete_Data.xls")

# Centering the concrete compressive strength data
data["Concrete compressive strength centered"] = (
    data["Concrete compressive strength(MPa, megapascals) "]
    - data["Concrete compressive strength(MPa, megapascals) "].mean()
)

# Define figure and subplots (2 rows, 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

# List of components and their corresponding labels
components = [
    ("Cement (component 1)(kg in a m^3 mixture)", "Cement"),
    ("Blast Furnace Slag (component 2)(kg in a m^3 mixture)", "Blast Furnace Slag"),
    ("Fly Ash (component 3)(kg in a m^3 mixture)", "Fly Ash"),
    ("Water  (component 4)(kg in a m^3 mixture)", "Water"),
    ("Superplasticizer (component 5)(kg in a m^3 mixture)", "Superplasticizer"),
    ("Coarse Aggregate  (component 6)(kg in a m^3 mixture)", "Coarse Aggregate"),
    ("Fine Aggregate (component 7)(kg in a m^3 mixture)", "Fine Aggregate"),
    ("Age (day)", "Age"),
]

# Loop over components and plot in subplots
for ax, (column, label) in zip(axes.flat, components):
    ax.scatter(data[column], data["Concrete compressive strength centered"], alpha=0.5)
    ax.set_xlabel(label)
    ax.set_ylabel("Concrete compressive strength centered")
    ax.set_title(f"{label} vs Strength")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
