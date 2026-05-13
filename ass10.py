import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')
print(df.head())
print(df.describe())
print(df.isnull().sum())

column_name = ['Id', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df.columns = column_name

print("\nFeatures in the dataset:")
print("1. Sepal length :", df['sepal_length'].dtype, "\u2192 Numeric")
print("2. Sepal width  :", df['sepal_width'].dtype,  "\u2192 Numeric")
print("3. Petal length :", df['petal_length'].dtype, "\u2192 Numeric")
print("4. Petal width  :", df['petal_width'].dtype,  "\u2192 Numeric")
print("5. Species      :", df['species'].dtype,      "\u2192 Nominal")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Histogram - Sepal')
sns.histplot(x=df['sepal_length'], kde=True, ax=axes[0])
sns.histplot(x=df['sepal_width'],  kde=True, ax=axes[1])
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Histogram - Petal')
sns.histplot(x=df['petal_length'], kde=True, ax=axes[0])
sns.histplot(x=df['petal_width'],  kde=True, ax=axes[1])
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Boxplot - Sepal')
sns.boxplot(x=df['sepal_length'], ax=axes[0])
sns.boxplot(x=df['sepal_width'],  ax=axes[1])
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Boxplot - Petal')
sns.boxplot(x=df['petal_length'], ax=axes[0])
sns.boxplot(x=df['petal_width'],  ax=axes[1])
fig.tight_layout()
plt.show()

print("\nOutlier count per feature:")
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for feature in features:
    Q1   = df[feature].quantile(0.25)
    Q3   = df[feature].quantile(0.75)
    IQR  = Q3 - Q1
    low  = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < low) | (df[feature] > high)][feature].count()
    print(f"{feature}: {outliers} outliers")
