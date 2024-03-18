import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('./selected_features.csv')  # Replace with your dataset file path

# Drop irrelevant columns
columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP']
data = data.drop(columns_to_drop, axis=1)

# Handle Infinity and Large Values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Separate features (X) and target (y)
target_column_index = -1
X = data_imputed[:, :-1]
y = data_imputed[:, target_column_index]

# Convert target column to binary classes
threshold = 0.5
y_binary = (y > threshold).astype(int)

# Preprocess categorical data (one-hot encoding)
categorical_features = [2]
numeric_features = list(range(X.shape[1]))
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_features)],
    remainder='passthrough'
)

# Normalize the data to [0, 1] using MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Select the best features using chi-squared
selector = SelectKBest(score_func=chi2)
X_selected_chi2 = selector.fit_transform(X_normalized, y_binary)

# Retrieve selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Retrieve selected feature names
selected_features = []
for idx in selected_feature_indices:
    if idx < len(categorical_features):
        selected_features.append(f'cat_{idx}')
    else:
        selected_features.append(data.columns[idx - len(categorical_features)])

# Print selected features using chi-squared
print("Selected Features (chi-squared):")
print(selected_features)

# Create a DataFrame with selected features and write it to a CSV file
selected_features_df = data.iloc[:, selected_feature_indices]
selected_features_df.to_csv('chisquare_selected_features.csv', index=False)

# Split the data into training and testing sets
X_train_chi2, X_test_chi2, y_train, y_test = train_test_split(X_selected_chi2, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# Train a model (replace with your desired model)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_chi2, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_chi2)

# Evaluate the model for each class
print("Classification Report for Normal Requests:")
print(classification_report(y_test[y_test == 0], y_pred[y_test == 0], zero_division=1))
print("Classification Report for Abnormal Requests:")
print(classification_report(y_test[y_test == 1], y_pred[y_test == 1], zero_division=1))
