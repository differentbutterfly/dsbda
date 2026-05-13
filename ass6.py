import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

model = None
scaler = None
encoder = None


def remove_outlier(df, col):
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1

    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR

    print("\nColumn:", col)
    print("Lowest allowed:", low)
    print("Highest allowed:", high)
    print("Outliers:", df[(df[col] < low) | (df[col] > high)][col].count())

    return df[(df[col] >= low) & (df[col] <= high)]


def draw_boxplots(df, cols, title):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title)

    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(col)

    plt.tight_layout()
    plt.show()


def show_result(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:\n", cm)

    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("Error Rate:", 1 - acc)
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("Classification Report:\n", classification_report(y_test, y_pred))


def encode_species():
    global df, encoder

    if df['Species'].dtype == 'object':
        encoder = LabelEncoder()
        df['Species'] = encoder.fit_transform(df['Species'])

        print("Classes:", list(encoder.classes_))
        print("Encoded Values:", df['Species'].unique())
        print("Encoded Dataset:\n", df.head())
    else:
        print("Species column is already encoded")


def train_nb(nb_type):
    global model, scaler, encoder, df

    if encoder is None and df['Species'].dtype == 'object':
        encoder = LabelEncoder()
        df['Species'] = encoder.fit_transform(df['Species'])

    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GaussianNB() if nb_type == "gaussian" else BernoulliNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nPredicted Values:\n", y_pred)
    show_result(y_test, y_pred)


df = pd.read_csv("Iris.csv")
print("Iris Dataset Loaded Successfully")

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

choice = 1

while choice != 9:
    print("""
------------- MENU -------------
1. Display Dataset Information
2. Find Missing Values
3. Detect and Remove Outliers
4. Encoding using Label Encoder
5. Find Correlation Matrix
6. Apply Bernoulli Naive Bayes
7. Apply Gaussian Naive Bayes
8. Prediction on User Input
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
        draw_boxplots(df, features, "Before Removing Outliers")

        for col in features:
            df = remove_outlier(df, col)

        draw_boxplots(df, features, "After Removing Outliers")

    elif choice == 4:
        encode_species()

    elif choice == 5:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True)
        plt.title("Correlation Matrix")
        plt.show()

    elif choice == 6:
        train_nb("bernoulli")

    elif choice == 7:
        train_nb("gaussian")

    elif choice == 8:
        if model is None or scaler is None:
            print("Please run option 6 or 7 first to train a model.")
        else:
            values = [[
                float(input("Enter Sepal Length: ")),
                float(input("Enter Sepal Width: ")),
                float(input("Enter Petal Length: ")),
                float(input("Enter Petal Width: "))
            ]]

            values = scaler.transform(values)

            pred = model.predict(values)[0]

            print("Predicted Species Code:", pred)

            if encoder is not None:
                species_name = encoder.inverse_transform([pred])[0]
                print("Predicted Species Name:", species_name)
            else:
                print("Encoder not found, cannot convert code to species name")

    elif choice == 9:
        print("Program Ended Successfully")

    else:
        print("Invalid Choice")