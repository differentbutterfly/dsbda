import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Iris.csv")

print("Iris Dataset Loaded Successfully")
print("First 5 Rows:\n", df.head())
print("Dataset Information:\n")
df.info()
print("Statistical Summary:\n", df.describe())
print("Missing Values:\n", df.isnull().sum())

df.columns = [
    'Id',
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'species'
]

print("""
----------- FEATURES AND TYPES -----------

1. sepal_length  : Numeric
2. sepal_width   : Numeric
3. petal_length  : Numeric
4. petal_width   : Numeric
5. species       : Nominal / Categorical
""")

print("Datatype of Columns:\n", df.dtypes)

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# ---------- Histograms: 2 diagrams in one box ----------
for i in range(0, len(features), 2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Feature Histograms")

    sns.histplot(df[features[i]], kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram of {features[i]}")
    axes[0].set_xlabel(features[i])
    axes[0].set_ylabel("Count")

    sns.histplot(df[features[i + 1]], kde=True, ax=axes[1])
    axes[1].set_title(f"Histogram of {features[i + 1]}")
    axes[1].set_xlabel(features[i + 1])
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

# ---------- Boxplots: 2 diagrams in one box ----------
for i in range(0, len(features), 2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Feature Boxplots")

    sns.boxplot(x=df[features[i]], ax=axes[0])
    axes[0].set_title(f"Boxplot of {features[i]}")

    sns.boxplot(x=df[features[i + 1]], ax=axes[1])
    axes[1].set_title(f"Boxplot of {features[i + 1]}")

    plt.tight_layout()
    plt.show()

# ---------- Groupwise Boxplots: 2 diagrams in one box ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Groupwise Boxplots by Species")

sns.boxplot(x='species', y='sepal_length', data=df, ax=axes[0])
axes[0].set_title("Sepal Length by Species")

sns.boxplot(x='species', y='petal_length', data=df, ax=axes[1])
axes[1].set_title("Petal Length by Species")

plt.tight_layout()
plt.show()

print("""
----------- OBSERVATIONS -----------

1. All feature columns are numeric except species.
2. Species column is categorical/nominal.
3. Petal length and petal width show clear differences among species.
4. Iris-setosa has smaller petal dimensions.
5. Some outliers are present in sepal width.
6. Histogram helps understand feature distribution.
7. Boxplot helps identify median, quartiles and outliers.
""")