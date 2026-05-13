import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def remove_outlier(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR

    outliers = df[(df[col] < low) | (df[col] > high)][col].count()

    print("\nColumn:", col)
    print("Lowest allowed:", low)
    print("Highest allowed:", high)
    print("Outliers:", outliers)

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


df = pd.read_csv("student_dataset.csv")
print("Student Dataset Loaded Successfully")

num_cols = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']

choice = 0

while choice != 9:
    print("""
------------- MENU -------------
1. Display Dataset Information
2. Display Statistical Information
3. Find Missing Values
4. Detect and Remove Outliers
5. Convert Categorical to Numerical
6. Boxplot (gender vs raisedhands)
7. Boxplot (gender, nationality, discussion)
8. Scatterplot
9. Exit
""")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        df.info()
        print("Shape:", df.shape)
        print("Columns:", df.columns)
        print("Size:", df.size)
        print("Data Types:\n", df.dtypes)
        print("First 5 Rows:\n", df.head())
        print("Last 5 Rows:\n", df.tail())
        print("Random 5 Rows:\n", df.sample(5))

    elif choice == 2:
        print("Statistical Information:\n", df.describe())

    elif choice == 3:
        print("Missing Values:\n", df.isnull().sum())

    elif choice == 4:
        draw_boxplots(df, num_cols, "Before Removing Outliers")

        for col in num_cols:
            df = remove_outlier(df, col)

        draw_boxplots(df, num_cols, "After Removing Outliers")

    elif choice == 5:
        print("Before Encoding:", df['gender'].dtype)
        df['gender'] = df['gender'].astype('category').cat.codes
        print("After Encoding:", df['gender'].dtype)
        print("Encoded Values:", df['gender'].unique())

    elif choice == 6:
        sns.boxplot(data=df, x='gender', y='raisedhands', hue='gender')
        plt.title("Gender vs Raised Hands")
        plt.show()

    elif choice == 7:
        sns.boxplot(data=df, x='NationalITy', y='Discussion', hue='gender')
        plt.title("Gender, Nationality and Discussion")
        plt.xticks(rotation=90)
        plt.show()

    elif choice == 8:
        sns.scatterplot(data=df, x='raisedhands', y='VisITedResources', hue='gender')
        plt.title("Raised Hands vs Visited Resources")
        plt.show()

    elif choice == 9:
        print("Program Ended Successfully")

    else:
        print("Invalid Choice")