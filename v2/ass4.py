import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def remove_outlier(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR

    print(f"\n{col}")
    print("Lowest allowed:", low)
    print("Highest allowed:", high)
    print("Outliers:", df[(df[col] < low) | (df[col] > high)][col].count())

    return df[(df[col] >= low) & (df[col] <= high)]


def boxplot(df, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)

    sns.boxplot(data=df, x='rm', ax=ax[0])
    sns.boxplot(data=df, x='lstat', ax=ax[1])

    plt.tight_layout()
    plt.show()


def train_model(df):
    X = df[['rm', 'lstat']]
    y = df['medv']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Model Score:", model.score(X_test, y_test))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    return model


df = pd.read_csv("BostonHousing.csv")
print("Boston Housing Dataset Loaded Successfully")

choice = 1

while choice != 9:
    print("""
----------- MENU -----------
1. Display Dataset Information
2. Find Missing Values
3. Detect and Remove Outliers
4. Find Correlation Matrix
5. Train and Test Linear Regression Model
6. Predict Housing Price
7. Reload Dataset
8. Display Statistical Information
9. Exit
""")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        print("First 5 Rows:\n", df.head())
        print("Columns:\n", df.columns)
        print("Shape:", df.shape)
        print("Data Types:\n", df.dtypes)

    elif choice == 2:
        print("Missing Values:\n", df.isnull().sum())

    elif choice == 3:
        boxplot(df, "Before Removing Outliers")

        for col in ['rm', 'lstat']:
            df = remove_outlier(df, col)

        boxplot(df, "After Removing Outliers")

    elif choice == 4:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True)
        plt.title("Correlation Matrix")
        plt.show()

    elif choice == 5:
        model = train_model(df)

    elif choice == 6:
        model = train_model(df)

        rm = float(input("Enter average number of rooms: "))
        lstat = float(input("Enter lower status population percentage: "))

        features = pd.DataFrame({'rm': [rm], 'lstat': [lstat]})
        prediction = model.predict(features)

        print("Predicted House Price:", prediction[0])

    elif choice == 7:
        df = pd.read_csv("BostonHousing.csv")
        print("Dataset Reloaded Successfully")

    elif choice == 8:
        print("Statistical Information:\n", df.describe())

    elif choice == 9:
        print("Program Ended Successfully")

    else:
        print("Invalid Choice")