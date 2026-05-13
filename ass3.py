import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Employee.csv")
print("dataset Employee loaded successfully........")

df2 = pd.read_csv("Iris.csv")
print("dataset Iris loaded successfully........")

ch = 1
while ch != 9:
    print(" ")
    print("1.display employee info")
    print("2.stat info of numerical data")
    print("3.categorical data")
    print("4.barplot for employee")
    print("--iris--")
    print("5.display iris info")
    print("6.stat info of numerical data")
    print("7.categorical data")
    print("8.barplot for iris")
    print("9.exit")
    print(" ")


    ch = int(input("Enter choice: "))

    if ch == 1:
        df.info()
        print('Shape of Dataset (row x column): ', df.shape)
        print('Columns Name: ', df.columns)
        print('Total elements in dataset: ', df.size)
        print('Datatypes of attributes (columns): ', df.dtypes)
        print('First 5 rows:\n', df.head(5))
        print('Last 5 rows:\n', df.tail(5))
        print('Any 5 rows:\n', df.sample(5))
    elif ch == 2:
        print("statistical info employee:\n",df.describe())
    elif ch == 3:
        columns = ['ExperienceInCurrentDomain', 'Age', 'JoiningYear']
        for col in columns:
            print("_________________________")
            print(df.groupby('Gender')[col].describe())

    elif ch == 4:
        features = ['ExperienceInCurrentDomain', 'Age', 'JoiningYear']
        for var in features:
            df_stat = df.groupby('Gender')[var].describe()
            df_stat = df_stat[['min','max','mean','50%','std']]
            df_stat.columns = ['MIN','MAX','MEAN','MEDIAN','STD']

            df_stat.plot(kind='bar',figsize=(10,6))
            plt.xlabel("Gender")
            plt.ylabel(var)
            plt.title(f'Groupwise statistical information of {var} by Gender')
            plt.legend(title='Gender')
            plt.show()
    elif ch == 5:
        df2.info()
        print('Shape of Dataset (row x column): ', df2.shape)
        print('Columns Name: ', df2.columns)
        print('Total elements in dataset: ', df2.size)
        print('Datatypes of attributes (columns): ', df2.dtypes)
        print('First 5 rows:\n', df2.head(5))
        print('Last 5 rows:\n', df2.tail(5))
        print('Any 5 rows:\n', df2.sample(5))
    elif ch == 6:
        print("statistical info iris :\n",df2.describe())
    elif ch == 7:
        columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        for col in columns:
            print("_________________________")
            print(df2.groupby('Species')[col].describe())
    elif ch == 8:
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

    elif ch == 9:
        print("quitting ......")
        break
