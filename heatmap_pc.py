import matplotlib.pyplot as plt
import seaborn as sns

principal_components = [[0.09827295,  0.17725317, -0.39464178,  0.54705427, -0.50591697,  0.03805569, -0.40190575,  0.29152151], 
     [-0.11181022,  0.68562442, -0.14379962, 0.0529213, 0.28360405, -0.63034067, -0.01956876, -0.12567848], 
     [0.81449534, -0.17340093, -0.40777505, -0.21308433,  0.23419128, -0.17256392, -0.00484576,  0.10097873], 
     [0.05437612,  0.36269994, -0.22654071, -0.29601729,  0.03741495,  0.5457468, -0.38554226, -0.5278852], 
     [-0.14788131,  0.02121136, -0.5499439,  -0.07046483, -0.35441099,  0.03310011, 0.7011056,  -0.22809163,], 
     [-0.20312941,  0.30495397, -0.18309239, -0.36612798,  0.19324298,  0.31451971, 0.09236092,  0.74389043], 
     [-0.22208449, -0.22837173, -0.35236521,  0.52417861,  0.66463655,  0.22701428, 0.03908382, -0.06925024], 
     [0.44612725,  0.43735666,  0.38191098,  0.38874361,  0.05176469, 0.34935768, 0.43337671,  0.01289534]]

x_labels = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8']
y_labels = ['Cement','Blast Furnace Slag', 'Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age']

# Create heatmap with labels
plt.figure(figsize=(10, 8))
ax = sns.heatmap(principal_components, annot=True, 
                 xticklabels=x_labels, yticklabels=y_labels)

# You can also customize the labels further if needed
ax.set_xlabel('Principal Components')
ax.set_ylabel('Features')

# Rotate x-axis labels for better readability if needed
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()