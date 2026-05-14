# === ASSIGNMENT 3: Descriptive Statistics - Central Tendency & Variability ===
# Datasets: Employee.csv, Iris.csv
# Operations: Summary statistics grouped by categorical variables

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# === Section 1: Load Datasets ===
df = pd.read_csv("Employee.csv")
print("dataset Employee loaded successfully........")

df2 = pd.read_csv("Iris.csv")
print("dataset Iris loaded successfully........")

# ===== EMPLOYEE DATASET =====

# === Section 2: Employee - Display Info ===
print("\n========== EMPLOYEE DATASET ==========")
df.info()
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset: ', df.size)
print('Datatypes of attributes (columns): ', df.dtypes)
print('First 5 rows:\n', df.head(5))
print('Last 5 rows:\n', df.tail(5))
print('Any 5 rows:\n', df.sample(5))

# === Section 3: Employee - Statistical Info ===
print("\n--- Employee Statistical Info ---")
print("statistical info employee:\n", df.describe())

# === Section 4: Employee - Grouped by Gender ===
print("\n--- Employee Grouped by Gender ---")
columns = ['ExperienceInCurrentDomain', 'Age', 'JoiningYear']
for col in columns:
    print("_________________________")
    print(df.groupby('Gender')[col].describe())

# === Section 5: Employee - Barplots ===
print("\n--- Employee Grouped Barplots ---")
features = ['ExperienceInCurrentDomain', 'Age', 'JoiningYear']
for var in features:
    df_stat = df.groupby('Gender')[var].describe()
    df_stat = df_stat[['min', 'max', 'mean', '50%', 'std']]
    df_stat.columns = ['MIN', 'MAX', 'MEAN', 'MEDIAN', 'STD']

    df_stat.plot(kind='bar', figsize=(10, 6))
    plt.xlabel("Gender")
    plt.ylabel(var)
    plt.title(f'Groupwise statistical information of {var} by Gender')
    plt.legend(title='Gender')
    plt.show()

# ===== IRIS DATASET =====

# === Section 6: Iris - Display Info ===
print("\n========== IRIS DATASET ==========")
df2.info()
print('Shape of Dataset (row x column): ', df2.shape)
print('Columns Name: ', df2.columns)
print('Total elements in dataset: ', df2.size)
print('Datatypes of attributes (columns): ', df2.dtypes)
print('First 5 rows:\n', df2.head(5))
print('Last 5 rows:\n', df2.tail(5))
print('Any 5 rows:\n', df2.sample(5))

# === Section 7: Iris - Statistical Info ===
print("\n--- Iris Statistical Info ---")
print("statistical info iris :\n", df2.describe())

# === Section 8: Iris - Grouped by Species ===
print("\n--- Iris Grouped by Species ---")
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for col in columns:
    print("_________________________")
    print(df2.groupby('Species')[col].describe())

# === Section 9: Iris - Barplots by Species ===
print("\n--- Iris Grouped Barplots ---")
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for var in features:
    fig, ax = plt.subplots()
    grouped_stats = df2.groupby('Species')[var].agg(['min', 'max', 'mean', 'median', 'std'])
    grouped_stats.plot(kind='bar', ax=ax, width=0.25)
    ax.set_xlabel('Statistical Information')
    ax.set_ylabel(var)
    ax.set_title(f'Groupwise statistical information of {var} by Species')
    plt.legend(title='Species')
    plt.show()
