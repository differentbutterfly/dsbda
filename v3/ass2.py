# === ASSIGNMENT 2: Data Wrangling II ===
# Dataset: student_dataset.csv
# Operations: Missing values, outliers, transformations, visualizations

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# === Section 1: Load Dataset ===
df = pd.read_csv("student_dataset.csv")
print("dataset loaded successfully........")

# === Section 2: Helper Functions ===
def detectOutliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    IQR = q3 - q1

    high = q3 + 1.5 * IQR
    low = q1 - 1.5 * IQR

    df = df[(df[col] >= low) & (df[col] <= high)]
    print('outliers removed in ', col)

    return df

# === Section 3: Display Info and Stats ===
print("shape: \n", df.shape)
print("__________________________________")
print("columns: \n", df.columns)
print("__________________________________")
print("1st 5:\n", df.head().T)
print("last 5:\n", df.tail().T)
print("sample 5:\n", df.sample(5).T)
print("__________________________________")
print("statistical information: \n", df.describe())
print("__________________________________")

# === Section 4: Handle Missing Values ===
print("Null values before:")
print(df.isna().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

print("Null values after:")
print(df.isna().sum())
print("__________________________________")

# === Section 5: Detect and Remove Outliers ===
numcolumns = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]

fig, axes = plt.subplots(2, 2)
fig.suptitle("before removing Outliers")
sns.boxplot(data=df, x="raisedhands", ax=axes[0, 0])
sns.boxplot(data=df, x="VisITedResources", ax=axes[0, 1])
sns.boxplot(data=df, x="AnnouncementsView", ax=axes[1, 0])
sns.boxplot(data=df, x="Discussion", ax=axes[1, 1])
fig.tight_layout()
plt.show()

for col in numcolumns:
    df = detectOutliers(df, col)

fig, axes = plt.subplots(2, 2)
fig.suptitle("after removing Outliers")
sns.boxplot(data=df, x="raisedhands", ax=axes[0, 0])
sns.boxplot(data=df, x="VisITedResources", ax=axes[0, 1])
sns.boxplot(data=df, x="AnnouncementsView", ax=axes[1, 0])
sns.boxplot(data=df, x="Discussion", ax=axes[1, 1])
fig.tight_layout()
plt.show()
print("__________________________________")

# === Section 6: Data Transformation (Categorical to Quantitative) ===
df['gender'] = df['gender'].astype('category')
df['gender'] = df['gender'].cat.codes

print('category changed: ', df.dtypes['gender'])
print('gender values: ', df['gender'].unique())
print("__________________________________")

# === Section 7: Boxplot - Gender vs RaisedHands ===
sns.boxplot(data=df, x='gender', y='raisedhands', hue='gender')
plt.title("Boxplot with 2 variables gender and raisedhands")
plt.show()
print("__________________________________")

# === Section 8: Boxplot - Gender, Nationality, Discussion ===
sns.boxplot(data=df, x='Discussion', y='NationalITy', hue='gender')
plt.title('Boxplot with 3 variables gender, nationality, discussion')
plt.show()
print("__________________________________")

# === Section 9: Scatterplot - RaisedHands vs VisitedResources ===
sns.scatterplot(data=df, x='raisedhands', y='VisITedResources', hue='gender')
plt.title("Scatterplot for raisedhands and VisITedResources")
plt.show()
