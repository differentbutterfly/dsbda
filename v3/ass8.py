# === ASSIGNMENT 8: Data Visualization I ===
# Dataset: titanic_data.csv
# Objective: Visualize patterns in Titanic data using Seaborn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Section 1: Load Dataset ===
df = pd.read_csv("titanic_data.csv")

print("Titanic Dataset Loaded")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("First 5 rows:\n", df.head())
print("Missing values:\n", df.isnull().sum())
print("Statistics:\n", df.describe())

# === Section 2: Handle Missing Values ===
df['Age'] = df['Age'].fillna(df['Age'].mean())

print("Missing values after filling Age:\n", df.isnull().sum())

# === Section 3: Age Distribution ===
sns.histplot(data=df, x='Age', kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# === Section 4: Fare Distribution ===
sns.histplot(data=df, x='Fare', kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

# === Section 5: Age Distribution by Sex ===
sns.histplot(data=df, x='Age', hue='Sex', multiple='dodge')
plt.title("Age Distribution by Sex")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# === Section 6: Age Distribution by Passenger Class ===
sns.histplot(data=df, x='Age', hue='Pclass', multiple='dodge')
plt.title("Age Distribution by Passenger Class")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# === Section 7: Survival Count ===
sns.countplot(data=df, x='Survived')
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# === Section 8: Survival by Sex ===
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Survival by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

# === Section 9: Survival by Passenger Class ===
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

# === Section 10: Fare Distribution by Passenger Class ===
sns.boxplot(data=df, x='Pclass', y='Fare')
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()

# === Section 11: Age vs Fare by Survival ===
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived')
plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()
