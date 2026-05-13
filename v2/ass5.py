import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report


def load_data():
    df = pd.read_csv("Social_Network_Ads.csv")
    return df.drop(columns=['User ID'])


def remove_outlier(df, col):
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    print(f"\n{col}")
    print("Lowest allowed:", low)
    print("Highest allowed:", high)
    print("Outliers:", df[(df[col] < low) | (df[col] > high)][col].count())

    return df[(df[col] >= low) & (df[col] <= high)]


def draw_boxplot(df, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)

    sns.boxplot(data=df, x='Age', ax=ax[0])
    sns.boxplot(data=df, x='EstimatedSalary', ax=ax[1])

    plt.tight_layout()
    plt.show()


def train_logistic(df, scale=False):
    X = df[['Age', 'EstimatedSalary']]
    y = df['Purchased']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    scaler = None

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nActual Values:\n", y_test.values)
    print("\nPredicted Values:\n", y_pred)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    if scale:
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()

        print("\nConfusion Matrix:\n", cm)

        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        print("TP:", TP)
        print("TN:", TN)
        print("FP:", FP)
        print("FN:", FN)
        print("Error Rate:", 1 - accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, scaler


df = load_data()
model1 = None
scaler = None

print("Social Network Ads Dataset Loaded Successfully")

choice = 1

while choice != 10:
    print("""
------------- MENU -------------
1. Display Dataset Information
2. Find Missing Values
3. Detect and Remove Outliers
4. Encoding using Label Encoder
5. Find Correlation Matrix
6. Apply Logistic Regression
7. Normalize Data and Apply Logistic Regression
8. Predict Purchased or Not
9. Reload Dataset
10. Exit
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
        draw_boxplot(df, "Before Removing Outliers")

        for col in ['Age', 'EstimatedSalary']:
            df = remove_outlier(df, col)

        draw_boxplot(df, "After Removing Outliers")

    elif choice == 4:
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
        print("Encoded Dataset:\n", df.head())

    elif choice == 5:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True)
        plt.title("Correlation Matrix")
        plt.show()

    elif choice == 6:
        train_logistic(df, scale=False)

    elif choice == 7:
        model1, scaler = train_logistic(df, scale=True)

    elif choice == 8:
        if model1 is None or scaler is None:
            model1, scaler = train_logistic(df, scale=True)

        age = int(input("Enter Age: "))
        salary = int(input("Enter Estimated Salary: "))

        user_data = scaler.transform(pd.DataFrame({
            'Age': [age],
            'EstimatedSalary': [salary]
        }))

        prediction = model1.predict(user_data)

        if prediction[0] == 1:
            print("Customer Will Purchase Product")
        else:
            print("Customer Will NOT Purchase Product")

    elif choice == 9:
        df = load_data()
        model1 = None
        scaler = None
        print("Dataset Reloaded Successfully")

    elif choice == 10:
        print("Program Ended Successfully")

    else:
        print("Invalid Choice")