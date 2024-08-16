import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

no_show = pd.read_csv('no_show.csv')
clinics = pd.read_csv('clinics.csv')
planning_neighborhoods = pd.read_csv('planning_neighborhoods.csv')
no_show_historical = pd.read_csv('no_show_historical.csv')

print("No Show Columns:", no_show.columns)
print("Clinics Columns:", clinics.columns)
print("Planning Neighborhoods Columns:", planning_neighborhoods.columns)
print("No Show Historical Columns:", no_show_historical.columns)

if 'Clinic Location' in no_show.columns and 'clinic' in clinics.columns:
    data = pd.merge(no_show, clinics, left_on='Clinic Location', right_on='clinic')
    print('Successfully merged now show and clinic')
    print('Data Columns :',data.columns)
else:
    print("Clinic Location column not found in one of the dataframes.")
    data = no_show

if 'Neighborhood' in data.columns and 'neighborho' in planning_neighborhoods.columns:
    data = pd.merge(data, planning_neighborhoods, left_on='Neighborhood', right_on='neighborho')
    print('Successfully merged data and planning_neighborhoods ')
    print('data columns :',data.columns)
else:
    print("Neighborhood column not found in one of the dataframes.")


if 'Patient ID' in data.columns and 'Patient ID' in no_show_historical.columns:
    data = pd.merge(data, no_show_historical, on='Patient ID')
    print('Successfully merged data and no show historical: ')
    print('data colums: ',data.columns)
else:
    print("Patient ID column not found in one of the dataframes.")

# Print columns after merge
print("Columns after merging:", data.columns)

# Combine no_show_x and no_show_y into a single column
if 'no_show_x' in data.columns and 'no_show_y' in data.columns:
    data['target_no_show'] = data['no_show_x'].fillna(data['no_show_y'])
    print(data)
else:
    raise KeyError("One or both of 'no_show_x' and 'no_show_y' columns not found in the data.")

# Drop the original no_show columns
data.drop(['no_show_x', 'no_show_y'], axis=1, inplace=True)
print(data)

print("Columns after creating target_no_show:", data.columns)
print("Sample rows of target_no_show:\n", data[['target_no_show']].head())

print(data)

# Preprocess data
data['Gender'] = data['Gender'].map({'M': 0, 'F': 1})
mapping_dict = {
    '0/week': 0,
    '1/week': 1,
    '5/week': 2,
    '10/week': 3,
    '> 14/week': 4
}
# data['Alcohol Consumption'] = data['Alcohol Consumption'].map({'0/week': 0, '5/week': 1, '> 14/week': 2})
data['Alcohol Consumption'] = data['Alcohol Consumption'].map(mapping_dict)
data['target_no_show'] = data['target_no_show'].astype(int)
data['Hypertension'] = data['Hypertension'].astype(int)
data['Diabetes'] = data['Diabetes'].astype(int)


print(data['Alcohol Consumption'].head())

#  Create age groups
data['age_group'] = pd.cut(data['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '>60'])
data = pd.get_dummies(data, columns=['age_group'], drop_first=True)

data.head()

# Drop any remaining non-numeric columns
numeric_data = data.select_dtypes(include=[np.number])
numeric_data.head()

# Print a few rows before splitting
# print("Data before splitting:\n", data.head())
print("Data columns before splitting:", numeric_data.columns)

# Define features and target
if 'target_no_show' in numeric_data.columns:
    X = numeric_data.drop(['target_no_show'], axis=1)
    y = numeric_data['target_no_show']
else:
    raise KeyError("Column 'target_no_show' not found in the data.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

 
# Use stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and score
print(grid_search.best_params_)
print(grid_search.best_score_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Re-train the model with resampled data
best_model.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = best_model.predict(X_test)

print(confusion_matrix(y_test, y_pred_resampled))
print(classification_report(y_test, y_pred_resampled))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
import joblib

# Save the model
joblib.dump(best_model, 'best_model.pkl')

# Load the model
loaded_model = joblib.load('best_model.pkl')

import joblib

# Load the model
loaded_model = joblib.load('best_model.pkl')
