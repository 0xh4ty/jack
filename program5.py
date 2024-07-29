import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Fetch dataset
wine_quality = fetch_ucirepo(id=186)

# Data (as pandas dataframes)
data = wine_quality.data.features
target = wine_quality.data.targets

# Add target to the dataset for visualization
data['quality'] = target

# Display basic information
print("Dataset Information:")
print(data.info())

print("\nDataset Description:")
print(data.describe())

# Scatter plot to visualize relationships between numerical features
def plot_scatter(df, x_feature, y_feature, hue_feature=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_feature, y=y_feature, hue=hue_feature, palette='viridis', alpha=0.7)
    plt.title(f'Scatter Plot: {x_feature} vs {y_feature}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.show()

# Bar chart to visualize distributions of categorical features
def plot_bar(df, feature):
    plt.figure(figsize=(10, 6))
    df[feature].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f'Bar Chart: Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

# Example scatter plots
plot_scatter(data, 'alcohol', 'quality', 'quality')
plot_scatter(data, 'residual sugar', 'quality')

# Example bar chart
plot_bar(data, 'quality')

# Additional plots based on the dataset features
plot_scatter(data, 'pH', 'quality')
plot_scatter(data, 'citric acid', 'quality')
plot_bar(data, 'quality')
