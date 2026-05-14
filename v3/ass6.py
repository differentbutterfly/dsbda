# === ASSIGNMENT 6: Data Analytics III - Naive Bayes Classification ===
# Dataset: Iris.csv
# Objective: Classify iris species using Gaussian and Bernoulli Naive Bayes

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

# === Section 1: Helper Functions ===
def remove_outlier(df, col):
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1

    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR

    print("\nColumn:", col)
    print("Lowest allowed:", low)
    print("Highest allowed:", high)
    print("Outliers:", df[(df[col] < low) | (df[col] > high)][col].count())

    return df[(df[col] >= low) & (df[col] <= high)]


def draw_boxplots(df, cols, title):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title)

    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(col)

    plt.tight_layout()
    plt.show()


def show_result(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:\n", cm)

    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("Error Rate:", 1 - acc)
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("Classification Report:\n", classification_report(y_test, y_pred))


# === Section 2: Load Dataset ===
df = pd.read_csv("Iris.csv")
print("Iris Dataset Loaded Successfully")

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# === Section 3: Display Dataset Information ===
print("First 5 Rows:\n", df.head())
print("Columns:\n", df.columns)
print("Shape:", df.shape)
print("Data Types:\n", df.dtypes)

# === Section 4: Find Missing Values ===
print("Missing Values:\n", df.isnull().sum())

# === Section 5: Detect and Remove Outliers ===
draw_boxplots(df, features, "Before Removing Outliers")

for col in features:
    df = remove_outlier(df, col)

draw_boxplots(df, features, "After Removing Outliers")

# === Section 6: Label Encoding ===
if df['Species'].dtype == 'object':
    encoder = LabelEncoder()
    df['Species'] = encoder.fit_transform(df['Species'])

    print("Classes:", list(encoder.classes_))
    print("Encoded Values:", df['Species'].unique())
    print("Encoded Dataset:\n", df.head())

# === Section 7: Correlation Matrix ===
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Matrix")
plt.show()

# === Section 8: Bernoulli Naive Bayes ===
print("\n========== Bernoulli Naive Bayes ==========")
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_bnb = BernoulliNB()
model_bnb.fit(X_train, y_train)

y_pred_bnb = model_bnb.predict(X_test)

print("\nPredicted Values:\n", y_pred_bnb)
show_result(y_test, y_pred_bnb)

# === Section 9: Gaussian Naive Bayes ===
print("\n========== Gaussian Naive Bayes ==========")
model_gnb = GaussianNB()
model_gnb.fit(X_train, y_train)

y_pred_gnb = model_gnb.predict(X_test)

print("\nPredicted Values:\n", y_pred_gnb)
show_result(y_test, y_pred_gnb)

# === Section 10: Prediction on Sample from Test Set ===
print("\n========== Sample Predictions ==========")
sample_idx = 0
sample_input = X_test[sample_idx].reshape(1, -1)
pred = model_gnb.predict(sample_input)[0]
species_name = encoder.inverse_transform([pred])[0]
print(f"Sample index {sample_idx}: Predicted Species = {species_name}")
print(f"Actual Species = {encoder.inverse_transform([y_test.iloc[sample_idx]])[0]}")
