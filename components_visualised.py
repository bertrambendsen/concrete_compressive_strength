import pandas as pd
import matplotlib.pyplot as plt

# The principal directions of the considered components. 

data = pd.read_excel("data/Concrete_Data.xls") # Extracting data using panda
data["Concrete compressive strength centered"] = data["Concrete compressive strength(MPa, megapascals) "]-data["Concrete compressive strength(MPa, megapascals) "].mean() # Centering the data

# Plotting the components
plt.scatter(data["Cement (component 1)(kg in a m^3 mixture)"], data["Concrete compressive strength centered"])
plt.xlabel("Cement (component 1)(kg in a m^3 mixture)")
plt.ylabel("Concrete compressive strength centered")
plt.show()

plt.scatter(data["Blast Furnace Slag (component 2)(kg in a m^3 mixture)"], data["Concrete compressive strength centered"])
plt.xlabel("Blast Furnace Slag (component 2)(kg in a m^3 mixture)")
plt.ylabel("Concrete compressive strength centered")
plt.show()

plt.scatter(data["Fly Ash (component 3)(kg in a m^3 mixture)"], data["Concrete compressive strength centered"])
plt.xlabel("Fly Ash (component 3)(kg in a m^3 mixture)")
plt.ylabel("Concrete compressive strength centered")
plt.show()

plt.scatter(data["Water  (component 4)(kg in a m^3 mixture)"], data["Concrete compressive strength centered"])
plt.xlabel("Water  (component 4)(kg in a m^3 mixture)")
plt.ylabel("Concrete compressive strength centered")
plt.show()

plt.scatter(data["Superplasticizer (component 5)(kg in a m^3 mixture)"], data["Concrete compressive strength centered"])
plt.xlabel("Superplasticizer (component 5)(kg in a m^3 mixture)")
plt.ylabel("Concrete compressive strength centered")
plt.show()

plt.scatter(data["Coarse Aggregate  (component 6)(kg in a m^3 mixture)"], data["Concrete compressive strength centered"])
plt.xlabel("Coarse Aggregate  (component 6)(kg in a m^3 mixture)")
plt.ylabel("Concrete compressive strength centered")
plt.show()

plt.scatter(data["Fine Aggregate (component 7)(kg in a m^3 mixture)"], data["Concrete compressive strength centered"])
plt.xlabel("Fine Aggregate (component 7)(kg in a m^3 mixture)")
plt.ylabel("Concrete compressive strength centered")
plt.show()

plt.scatter(data["Age (day)"], data["Concrete compressive strength centered"])
plt.xlabel("Age (day)")
plt.ylabel("Concrete compressive strength centered")
plt.show()