import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("titanic_data.csv")

print("Titanic Dataset Loaded")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("First 5 rows:\n", df.head())
print("Missing values:\n", df.isnull().sum())
print("Statistics:\n", df.describe())

df['Age'] = df['Age'].fillna(df['Age'].mean())

print("Missing values after filling Age:\n", df.isnull().sum())

sns.histplot(data=df, x='Age', kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

sns.histplot(data=df, x='Fare', kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

sns.histplot(data=df, x='Age', hue='Sex', multiple='dodge')
plt.title("Age Distribution by Sex")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

sns.histplot(data=df, x='Age', hue='Pclass', multiple='dodge')
plt.title("Age Distribution by Passenger Class")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

sns.countplot(data=df, x='Survived')
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Survival by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

sns.boxplot(data=df, x='Pclass', y='Fare')
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.show()

sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived')
plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()
