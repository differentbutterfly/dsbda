import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("weatherAUS.csv")

print("Dataset loaded")

# dataset is too big so taking small sample otherwise svm takes forever
df = df.sample(n=5000, random_state=42)
print("taking sample of 5000 rows for faster processing")

print(df.head())
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())
print("Null values before removing:")
print(df.isnull().sum())

df = df.dropna()

print("Null values after removing:")
print(df.isnull().sum())

le = LabelEncoder()

cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']

for col in cols:
    df[col] = le.fit_transform(df[col])

def svm_model(kernel_name):
    print("training SVM with", kernel_name, "kernel please wait...")
    
    X = df.drop(['RainTomorrow', 'Date', 'RISK_MM'], axis=1)
    y = df['RainTomorrow']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = SVC(kernel=kernel_name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(kernel_name, "kernel result")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))

choice = 0

while choice != 14:
    print("\n1. Display dataset")
    print("2. Display columns")
    print("3. Display first 5 rows")
    print("4. Display shape")
    print("5. Display information")
    print("6. Display statistical description")
    print("7. Check null values")
    print("8. Average temperature")
    print("9. Find Average Humidity")
    print("10. Average wind speed")
    print("11. Correlation heatmap")
    print("12. SVM with linear kernel")
    print("13. SVM with polynomial kernel")
    print("14. Exit")

    choice = int(input("Enter choice: "))

    if choice == 1:
        print(df)

    elif choice == 2:
        print(df.columns)

    elif choice == 3:
        print(df.head())

    elif choice == 4:
        print(df.shape)

    elif choice == 5:
        df.info()

    elif choice == 6:
        print(df.describe())

    elif choice == 7:
        print(df.isnull().sum())

    elif choice == 8:
        print("Average temperature:", df['Temp9am'].mean())

    elif choice == 9:
        print("Average humidity:", df['Humidity9am'].mean())

    elif choice == 10:
        print("Average wind speed:", df['WindSpeed9am'].mean())

    elif choice == 11:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True)
        plt.title("Correlation Heatmap")
        plt.show()

    elif choice == 12:
        svm_model('linear')

    elif choice == 13:
        svm_model('poly')

    elif choice == 14:
        print("Program ended")

    else:
        print("Invalid choice")