# Appendix

### A. Dataset Description
- Source: UCI Machine Learning Repository – Cleveland Heart Disease dataset
- Total Records: 303
- Features: 13 attributes + 1 target
- Target Variable: `condition` (0 = no disease, 1 = presence of heart disease)
- Feature List:
  - age, sex, cp (chest pain type), trestbps (resting blood pressure), etc.

### B. Quick Insight

```python
import pandas as pd

# Load the dataset with the correct filename
df = pd.read_csv('heart_cleveland_upload.csv')

# Display the first five rows of the dataset
df.head()
```
![TabNet Accuracy Output](../images/quick_insight.PNG)
### C. Code for Data Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features (X) and target variable (y)
X = df.drop('condition', axis=1)
y = df['condition']

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### D. Code for Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train the model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predict
y_pred = lr_model.predict(X_test_scaled)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```
![TabNet Accuracy Output](../images/logistic_regression.PNG)
### E. Code for Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
```
![TabNet Accuracy Output](../images/random_forest.PNG)
### F. Code for Gradient Boosting Classifier

```python
from sklearn.ensemble import GradientBoostingClassifier

# Train model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Predict
y_pred_gb = gb_model.predict(X_test_scaled)

# Evaluate
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Classification Report:\n", classification_report(y_test, y_pred_gb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))
```
![TabNet Accuracy Output](../images/gradient_boosting.PNG)
### G. TabNet Setup and Code

```python
!pip install pytorch-tabnet

import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

# Load dataset
df = pd.read_csv("heart_cleveland_upload.csv")

# Separate features and target
X = df.drop("condition", axis=1).values
y = df["condition"].values

# Encode target labels (if not already 0 and 1)
y = LabelEncoder().fit_transform(y)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = TabNetClassifier()

clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_name=['test'],
    eval_metric=['accuracy'],
    max_epochs=100,
    patience=10,
    batch_size=32,
    virtual_batch_size=16,
    num_workers=0,
    drop_last=False
)
```
![TabNet Accuracy Output](../images/clf.PNG)
```python
# Make predictions
y_pred = clf.predict(X_test)

# Evaluation
print("TabNet Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```
![TabNet Accuracy Output](../images/tabnet.PNG)
