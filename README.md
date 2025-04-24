# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Iris Dataset

2.Create a DataFrame from the Dataset

3.Add Target Labels to the DataFrame

4.Split Data into Features (X) and Target (y)

5.Split Data into Training and Testing Sets

6.Initialize the SGDClassifier Model

7.Train the Model on Training Data

8.Make Predictions on Test Data

9.Evaluate Accuracy of Predictions

10.Generate and Display Confusion Matrix

11.Generate and Display Classification Report

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Priyadharshini G
RegisterNumber:  212224230209
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
iris=load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target
print(df.head())
X=df.drop('target',axis=1)
y=df['target']
X_train,X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cn=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cn)
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)



```

## Output:

## df.head()
![image](https://github.com/user-attachments/assets/e66f3b8c-0abe-4bdd-8549-2d2f3b119d7d)
## Accuracy
![image](https://github.com/user-attachments/assets/31b38a8a-0fac-4039-bbdf-e510519924a0)
## Confusion matrix
![image](https://github.com/user-attachments/assets/064daf9c-1a9f-4f93-a1e2-07892d6e339e)
## Classification report
![image](https://github.com/user-attachments/assets/fbee3c30-051d-444d-99a9-dbb03f46fc24)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
