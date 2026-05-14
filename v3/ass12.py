# === ASSIGNMENT 12: SVM Classification on Weather Data ===
# Dataset: weatherAUS.csv
# Objective: Predict RainTomorrow using SVM with linear and polynomial kernels

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === Section 1: Load Dataset ===
df = pd.read_csv("weatherAUS.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("First 5 rows:\n", df.head())
print("Missing values:\n", df.isnull().sum())

# === Section 2: Handle Missing Values ===
df = df.dropna()

print("Missing values after dropna:\n", df.isnull().sum())

# === Section 3: Drop Unnecessary Columns ===
df = df.drop(['Date', 'Location', 'WindDir9am', 'WindDir3pm'], axis=1)

# === Section 4: Label Encoding ===
le = LabelEncoder()

df['WindGustDir'] = le.fit_transform(df['WindGustDir'])
df['RainToday'] = le.fit_transform(df['RainToday'])
df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])

print("Data after encoding:\n", df.head())

# === Section 5: Correlation Matrix ===
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# === Section 6: Train-Test Split ===
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

# === Section 7: Feature Scaling ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Section 8: SVM with Linear Kernel ===
model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# === Section 9: SVM with Polynomial Kernel ===
model_poly = SVC(kernel='poly')
model_poly.fit(X_train, y_train)

y_pred_poly = model_poly.predict(X_test)

print("Polynomial SVM Accuracy:", accuracy_score(y_test, y_pred_poly))
