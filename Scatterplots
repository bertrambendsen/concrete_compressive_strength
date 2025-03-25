import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


data = pd.read_excel('/Users/ehabal-saoudi/Downloads/concrete+compressive+strength/Concrete_Data.xls')

data.columns = data.columns.str.replace(r'\s*\(.*\)', '', regex=True)

scaler = StandardScaler()
standardized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

sns.set(style="whitegrid", rc={"axes.labelsize": 16, "axes.titlesize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14})

pairplot = sns.pairplot(standardized_data, 
                         diag_kind="kde", 
                         plot_kws={'alpha': 0.5, 's': 10, 'edgecolor': 'none'},  
                         diag_kws={'color': 'navy'})  

plt.show()
