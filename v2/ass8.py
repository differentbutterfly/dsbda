import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set(style="darkgrid")

dataset = sns.load_dataset('titanic')

print("Titanic Dataset Loaded Successfully")
print("First 5 Rows:\n", dataset.head())
print("Shape:", dataset.shape)
print("Columns:", dataset.columns)
print("Data Types:\n", dataset.dtypes)
print("Missing Values:\n", dataset.isnull().sum())

dataset['age'] = dataset['age'].fillna(dataset['age'].median())

print("Missing Values After Filling:\n", dataset.isnull().sum())
print("Statistical Information:\n", dataset.describe())

# 1. Correlation heatmap separately because it needs more space
plt.figure(figsize=(10, 6))
sns.heatmap(dataset.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 2. First 3 diagrams in one box
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fare and Age Distributions")

sns.histplot(dataset['fare'], kde=True, ax=axes[0])
axes[0].set_title("Fare with KDE")
axes[0].set_xlabel("Fare")
axes[0].set_ylabel("Count")

sns.histplot(dataset['fare'], kde=False, ax=axes[1])
axes[1].set_title("Fare without KDE")
axes[1].set_xlabel("Fare")
axes[1].set_ylabel("Count")

sns.histplot(dataset['fare'], kde=False, bins=10, ax=axes[2])
axes[2].set_title("Fare with 10 Bins")
axes[2].set_xlabel("Fare")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.show()

# 3. Next 3 diagrams in one box
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Titanic Dataset Patterns")

sns.histplot(dataset['age'], kde=True, bins=20, ax=axes[0])
axes[0].set_title("Age Distribution")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")

sns.histplot(data=dataset, x='fare', hue='class', multiple='stack', ax=axes[1])
axes[1].set_title("Fare by Passenger Class")
axes[1].set_xlabel("Fare")
axes[1].set_ylabel("Count")

sns.countplot(data=dataset, x='survived', ax=axes[2])
axes[2].set_title("Survival Count")
axes[2].set_xlabel("Survived")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.show()

# 4. Remaining diagram
plt.figure(figsize=(6, 5))
sns.countplot(data=dataset, x='sex')
plt.title("Gender Count")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

print("\nVisualization Completed Successfully")