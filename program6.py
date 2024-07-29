import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Fetch dataset
wine_quality = fetch_ucirepo(id=186)

# Data (as pandas dataframes)
X = wine_quality.data.features
y = wine_quality.data.targets

# Display metadata and variable information
print("Metadata:")
print(wine_quality.metadata)
print("\nVariable Information:")
print(wine_quality.variables)

# 1. Handle Missing Data
# Check for missing values
print("\nMissing Values Before Imputation:")
print(X.isnull().sum())

# Impute missing values using mean for numerical columns
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Verify that missing values have been handled
print("\nMissing Values After Imputation:")
print(X_imputed.isnull().sum())

# 2. Encode Categorical Variables
# If there were categorical variables, we would encode them here
# For this dataset, all features are numerical, but if you had categorical variables, use:
# encoder = LabelEncoder()
# X['categorical_feature'] = encoder.fit_transform(X['categorical_feature'])

# 3. Perform Feature Scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# Ensure y is a 1D array
y = y.values.ravel()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a simple model for demonstration
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy}")
