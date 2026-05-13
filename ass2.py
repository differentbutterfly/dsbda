import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("student_dataset.csv")
print("dataset loaded successfully........")

def detectOutliers(df,col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    IQR = q3-q1

    high = q3 + 1.5*IQR
    low = q1 - 1.5 *IQR

    df = df[(df[col] >= low) & (df[col] <= high)]
    print('outliers removed in ',col)

    return df
ch=0
while ch < 9:
    print("1.display info and stats")
    print("2.handle null value")
    print("3.detect outliers")
    print("4.data transformations: category to quantitative")
    print("5.2 var boxplot:gender and raisedhands")
    print("6.3 var box plot gender,nationality,discussion")
    print("7.scatterplot raisedhands and visitedresources")
    print("8.exit")
    ch = int(input("Enter choice: "))

    if ch == 1:
        print("shape: \n",df.shape)
        print("__________________________________")
        print("columns: \n",df.columns)
        print("__________________________________")
        print("1st 5:\n",df.head().T)
        print("last 5:\n",df.tail().T)
        print("sample 5:\n",df.sample(5).T)

        print("statistical information: \n",df.describe())

    elif ch ==2:
        print("Null values before:")
        print(df.isna().sum())

        df.fillna(df.mean(numeric_only=True), inplace=True)
        df.fillna(df.mode().iloc[0], inplace=True)

        print("Null values after:")
        print(df.isna().sum())
    elif ch == 3:
        numcolumns = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]

        fig, axes = plt.subplots(2, 2)
        fig.suptitle("before removing Outliers")
        sns.boxplot(data=df,x="raisedhands",ax=axes[0,0])
        sns.boxplot(data=df,x="VisITedResources",ax=axes[0,1])
        sns.boxplot(data=df,x="AnnouncementsView",ax=axes[1,0])
        sns.boxplot(data=df,x="Discussion",ax=axes[1,1])
        fig.tight_layout()
        plt.show()

        for col in numcolumns:
            df = detectOutliers(df,col)

        fig, axes = plt.subplots(2, 2)
        fig.suptitle("after removing Outliers")
        sns.boxplot(data=df,x="raisedhands",ax=axes[0,0])
        sns.boxplot(data=df,x="VisITedResources",ax=axes[0,1])
        sns.boxplot(data=df,x="AnnouncementsView",ax=axes[1,0])
        sns.boxplot(data=df,x="Discussion",ax=axes[1,1])
        fig.tight_layout()
        plt.show()

    elif ch == 4:
        df['gender'] = df['gender'].astype('category')
        df['gender'] = df['gender'].cat.codes

        print('category changed: ' ,df.dtypes['gender'])
        print('gender values: ',df['gender'].unique())
    elif ch == 5:
        sns.boxplot(data = df,x = 'gender',y='raisedhands',hue='gender')
        plt.title("Boxplot with 2 variables gender and raisedhands")
        plt.show()
    elif ch == 6:
        sns.boxplot(data = df,x = 'Discussion',y='NationalITy',hue='gender')
        plt.title('Boxplot with 3 variables gender, nationality, discussion')
        plt.show()
    elif ch == 7:
        sns.scatterplot(data = df,x = 'raisedhands',y='VisITedResources',hue='gender')
        plt.title("Scatterplot for raisedhands and VisITedResources")
        plt.show()
    elif ch == 8:
        break
    else:
         print("Invalid choice. Please enter a number between 1 and 9.")
