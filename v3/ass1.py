# === ASSIGNMENT 1: Data Wrangling I ===
# Dataset: Placement_data_full_class.csv
# Operations: Import, load, inspect, clean, type conversion, encoding

import pandas as pd

# === Section 1: Load Dataset ===
df = pd.read_csv("Placement_data_full_class.csv")
print("Dataset loaded successfully")

# === Section 2: Basic Inspection ===
print("columns:", df.columns)
print("_________________________________________________________________________")
print("first 5 rows\n", df.head(5).T)
print("_________________________________________________________________________")
print("last 5 rows:\n", df.tail(5).T)
print("_________________________________________________________________________")
print("random sample:\n", df.sample(5))
print("_________________________________________________________________________")
print("shape of the dataset:", df.shape)
print("_________________________________________________________________________")

# === Section 3: Statistical Summary ===
print("describe():\n", df.describe())
print("_________________________________________________________________________")

# === Section 4: Missing Values ===
print("null values:\n", df.isnull().sum())
print("_________________________________________________________________________")

# === Section 5: Data Types ===
print("data types of the data:\n")
dataTypes = df.dtypes
print(dataTypes)
print("_________________________________________________________________________")

# === Section 6: Missing Values (detailed) ===
print("Find missing values")
missing_val = df.isna().sum()
print(missing_val)
print("_________________________________________________________________________")

# === Section 7: Data Type Conversion ===
print("data type conversion:")
df['sl_no'] = df['sl_no'].astype(float)
print(df['sl_no'])

# === Section 8: Categorical to Numeric Encoding ===
print("convert categorical data into numeric data")
df = pd.get_dummies(df, columns=['gender'])
print(df.dtypes)
print("_________________________________________________________________________")

# === Section 9: Final Sample ===
print("random sample:\n", df.sample(5))
