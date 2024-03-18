import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('./selected_features_final.csv') 

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

# Define the pipeline for feature selection using f_classif and SelectPercentile
pipeline_f_classif = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('feature_selector', SelectPercentile(score_func=f_classif, percentile=50))
])

# Fit and transform the data using the pipeline
X_selected = pipeline_f_classif.fit_transform(X, y_binary)
selected_feature_indices = pipeline_f_classif.named_steps['feature_selector'].get_support(indices=True)
selected_features = []

# Retrieve selected feature names
for idx in selected_feature_indices:
    if idx < len(categorical_features):
        selected_features.append(f'cat_{idx}')
    else:
        selected_features.append(data.columns[idx - len(categorical_features)])

# Print selected features using f_classif
print("Selected Features (f_classif):")
print(selected_features)

# Create a DataFrame with selected features and data
selected_features_data = pd.DataFrame(X_selected, columns=selected_features)

# Save selected features and data to a CSV file
selected_features_data.to_csv('f_classif_selected_features.csv', index=False)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# Train a model (replace with your desired model)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model for each class
print("Classification Report for Normal Requests:")
print(classification_report(y_test[y_test == 0], y_pred[y_test == 0], zero_division=1))
print("Classification Report for Abnormal Requests:")
print(classification_report(y_test[y_test == 1], y_pred[y_test == 1], zero_division=1))
