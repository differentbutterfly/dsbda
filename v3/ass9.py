# === ASSIGNMENT 9: Data Visualization II ===
# Dataset: titanic_data.csv
# Objective: Boxplot analysis of Age, Fare, Sex, Survived

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# === Section 1: Load Dataset ===
df = pd.read_csv("titanic_data.csv")
print('Titanic Dataset is successfully loaded....')

# === Section 2: Display Information ===
df.info()
print('Shape of Dataset (row x column):', df.shape)
print('Columns Name:', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n', df.tail().T)
print('Any 5 rows:\n', df.sample(5).T)

# === Section 3: Find Missing Values ===
print('Total Number of Null Values in Dataset:\n', df.isna().sum())

# === Section 4: Fill Missing Values ===
df['Age'] = df['Age'].fillna(df['Age'].median())
print('Null values after filling:\n', df.isna().sum())

# === Section 5: Boxplot - 1 Variable (Age & Fare) ===
fig, axes = plt.subplots(1, 2)
fig.suptitle('Boxplot of 1-variable (Age & Fare)')
sns.boxplot(data=df, x='Age', ax=axes[0])
sns.boxplot(data=df, x='Fare', ax=axes[1])
fig.tight_layout()
plt.show()

# === Section 6: Boxplot - 2 Variables ===
fig, axes = plt.subplots(2, 2)
fig.suptitle('Boxplot of 2-variables')
sns.boxplot(data=df, x='Embarked', y='Age', ax=axes[0, 0])
sns.boxplot(data=df, x='Embarked', y='Fare', ax=axes[0, 1])
sns.boxplot(data=df, x='Sex', y='Age', ax=axes[1, 0])
sns.boxplot(data=df, x='Sex', y='Fare', ax=axes[1, 1])
fig.tight_layout()
plt.show()

# === Section 7: Boxplot - 3 Variables (Age vs Sex vs Survived) ===
fig, axes = plt.subplots(1, 2)
fig.suptitle('Boxplot of 3-variables (Age vs Sex vs Survived)')
sns.boxplot(data=df, x='Sex', y='Age', hue='Survived', ax=axes[0])
sns.boxplot(data=df, x='Sex', y='Fare', hue='Survived', ax=axes[1])
fig.tight_layout()
plt.show()
