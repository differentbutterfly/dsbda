import pandas as pd

df = pd.read_csv("Placement_data_full_class.csv")

print("columns: ",df.columns)
print("_________________________________________________________________________")
print("first 5 rows\n",df.head(5).T)
print("_________________________________________________________________________")
print("last 5 rows: \n",df.tail(5).T)
print("_________________________________________________________________________")
print("random sample:\n ",df.sample(5))
print("_________________________________________________________________________")
print("shape of the dataset: ",df.shape)
print("_________________________________________________________________________")
print("describe():\n",df.describe())
print("_________________________________________________________________________")
print("null values:\n",df.isnull().sum())
print("_________________________________________________________________________")
print("data types of the data:\n")
dataTypes = df.dtypes
print(dataTypes)
print("_________________________________________________________________________")
print("Find missing values")
missing_val = df.isna().sum()
print(missing_val)
print("_________________________________________________________________________")

print("data type conversion: ")
df['sl_no'] = df['sl_no'].astype(float)
print(df['sl_no'])

print('convert categorical data into numeric data')
df = pd.get_dummies(df,columns=['gender'])
print(df.dtypes)
print("_________________________________________________________________________")

print("random sample:\n ",df.sample(5))
