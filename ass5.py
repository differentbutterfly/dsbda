import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def remove_outliers(df,var):
    q3 = df[var].quantile(0.75)
    q1 = df[var].quantile(0.25)
    iqr = q3 - q1

    high = q3 + 1.5*iqr
    low = q1 - 1.5*iqr

    df = df[(df[var] >= low) & (df[var] <= high)]
    return df


def draw(df,msg):
    fig,axes = plt.subplots(1,2,figsize=(10,7))
    fig.suptitle(msg)
    sns.boxplot(data = df,x='Age',ax=axes[0])
    sns.boxplot(data = df,x='EstimatedSalary',ax=axes[1])
    fig.tight_layout()
    plt.show()


df = pd.read_csv('Social_Network_Ads.csv')
print("Columns:", df.columns.tolist())
print("First 5 rows:\n", df.head().T)
print("Null values:\n", df.isnull().sum())


draw(df,'before removing outliers')

columns = ['Age','EstimatedSalary']
for col in columns:
    df = remove_outliers(df,col)

draw(df,'after removing outliers')

X=df[['Age','EstimatedSalary']]
Y=df['Purchased']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

print("predict: ",Y_pred[:10])
print("actual: ",Y_test[:10])

cm = confusion_matrix(Y_test,Y_pred)
print("confusion_matrix",cm)

sns.heatmap(cm,annot=True,cmap='Blues',
    xticklabels=['Not Purchased','Purchased'],yticklabels=['Not Purchased','Purchased'])
plt.title("confusion matrix")
plt.ylabel("actual")
plt.xlabel("predicted")
plt.show()


TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

accuracy = (TN+TP) /(TN+TP+FP+FN)
error = 1-accuracy
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1 = 2*(precision * recall) / (precision+recall)

print(f'TP         : {TP}')
print(f'TN         : {TN}')
print(f'FP         : {FP}')
print(f'FN         : {FN}')
print(f'Accuracy   : {accuracy:.2f}')
print(f'Error Rate : {error:.2f}')
print(f'Precision  : {precision:.2f}')
print(f'Recall     : {recall:.2f}')
print(f'F1 Score   : {f1:.2f}')
