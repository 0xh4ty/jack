from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch the dataset from the UCI repository
online_retail = fetch_ucirepo(id=352)

# Extract features and targets
X = online_retail.data.features
y = online_retail.data.targets

# Display the metadata and variable information
print(online_retail.metadata)
print(online_retail.variables)

# Data Preprocessing
# Drop rows with missing values
X_clean = X.dropna()

# Select numeric features for clustering
# For example, using 'Quantity' and 'UnitPrice' for clustering
X_cluster = X_clean[['Quantity', 'UnitPrice']].values

# Apply K-Means clustering
# Let's choose 3 clusters for this example
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_cluster)
y_kmeans = kmeans.predict(X_cluster)

# Add the cluster labels to the original dataframe using .loc
X_clean = X_clean.copy()
X_clean.loc[:, 'Cluster'] = y_kmeans

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quantity', y='UnitPrice', hue='Cluster', data=X_clean, palette='viridis')
plt.title('K-Means Clustering of Online Retail Data')
plt.xlabel('Quantity')
plt.ylabel('Unit Price')
plt.legend(title='Cluster')
plt.show()
