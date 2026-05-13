import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("titanic_data.csv")

print("Titanic Dataset Loaded Successfully")
print("First 5 Rows:\n", df.head())
print("Dataset Shape:", df.shape)
print("Column Names:\n", df.columns)
print("Missing Values:\n", df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())

print("Missing Values After Filling:\n", df.isnull().sum())

# ---------- First 3 plots ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Titanic Basic Visualizations")

sns.countplot(x='Survived', data=df, ax=axes[0])
axes[0].set_title("Count Plot of Survival")

df['Sex'].value_counts().plot(
    kind='pie',
    autopct='%1.2f%%',
    ax=axes[1]
)
axes[1].set_title("Male and Female Passengers")
axes[1].set_ylabel("")

axes[2].hist(df['Age'], bins=10)
axes[2].set_title("Histogram of Age")
axes[2].set_xlabel("Age")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.show()

# ---------- Next 3 plots ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Titanic Relationship Visualizations")

sns.histplot(df['Age'], kde=True, ax=axes[0])
axes[0].set_title("Distribution Plot of Age")

sns.scatterplot(x='Age', y='Fare', hue='Sex', data=df, ax=axes[1])
axes[1].set_title("Age vs Fare")

sns.barplot(x='Pclass', y='Fare', hue='Sex', data=df, ax=axes[2])
axes[2].set_title("Passenger Class and Fare")

plt.tight_layout()
plt.show()

# ---------- Last 2 plots ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Titanic Survival Analysis")

sns.boxplot(x='Sex', y='Age', hue='Survived', data=df, ax=axes[0])
axes[0].set_title("Age by Gender and Survival")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Age")

ct = pd.crosstab(df['Pclass'], df['Survived'])
sns.heatmap(ct, annot=True, fmt='d', ax=axes[1])
axes[1].set_title("Pclass vs Survived")

plt.tight_layout()
plt.show()

print("""
----------- OBSERVATIONS -----------

1. Female passengers survived more compared to male passengers.
2. Most passengers were between age 20 to 40 years.
3. Children had higher survival chances.
4. Male passengers show more age outliers.
5. First class passengers paid higher fare.
6. Heatmap shows survival count according to passenger class.
7. Boxplot helps identify median, quartiles and outliers.
""")