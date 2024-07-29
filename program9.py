from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt

# Fetch the Iris dataset from the UCI repository
iris = fetch_ucirepo(id=53)

# Extract features and targets
X = iris.data.features
y = iris.data.targets

# Display metadata and variable information
print(iris.metadata)
print(iris.variables)

# Define the target names manually
target_names = ['setosa', 'versicolor', 'virginica']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Visualize the decision tree
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=X.columns,  
                           class_names=target_names,  
                           filled=True, rounded=True,  
                           special_characters=True)  

# Render the decision tree using graphviz
graph = graphviz.Source(dot_data)  
graph.render("iris_decision_tree")  # Save the tree as a file
graph.view()  # Open the tree visualization

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(X.columns, clf.feature_importances_)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in the Decision Tree")
plt.show()
