import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def remove_outlier(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR

    print(f"{col} outliers:",
          df[(df[col] < low) | (df[col] > high)][col].count())

    return df[(df[col] >= low) & (df[col] <= high)]


def boxplots(df, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(title)

    sns.boxplot(data=df, x='rm', ax=axes[0])
    sns.boxplot(data=df, x='lstat', ax=axes[1])

    axes[0].set_title('rm')
    axes[1].set_title('lstat')

    plt.tight_layout()
    plt.show()


df = pd.read_csv('BostonHousing.csv')

print("Columns:", df.columns.tolist())
print("First 5 rows:\n", df.head().T)
print("Null values:\n", df.isnull().sum())

boxplots(df, "Before Removing Outliers")

for col in ['lstat', 'rm']:
    df = remove_outlier(df, col)

boxplots(df, "After Removing Outliers")

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

print("Correlation with medv:\n",
      df.corr(numeric_only=True)['medv'].sort_values(ascending=False))

X = df.drop('medv', axis=1)
Y = df['medv']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

model = LinearRegression()
model.fit(X_train, Y_train)

print("Intercept:", model.intercept_)
print("Coefficients:\n", pd.Series(model.coef_, index=X_train.columns))

Y_pred = model.predict(X_test)

print("Actual values:\n", Y_test.values)
print("Predicted values:\n", Y_pred)

n = len(Y_test)

mae = np.sum(np.abs(Y_test.values - Y_pred)) / n
mse = np.sum((Y_test.values - Y_pred) ** 2) / n
rmse = mse ** 0.5

ss_res = np.sum((Y_test.values - Y_pred) ** 2)
ss_tot = np.sum((Y_test.values - Y_test.values.mean()) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

plt.scatter(Y_test, Y_pred)
plt.plot([min(Y_test), max(Y_test)],
         [min(Y_test), max(Y_test)], 'r')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()

residuals = Y_test.values - Y_pred

plt.scatter(Y_pred, residuals)
plt.axhline(y=0, color='r')

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
