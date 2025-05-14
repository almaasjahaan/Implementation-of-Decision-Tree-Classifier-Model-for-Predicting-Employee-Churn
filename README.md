
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries and load the dataset.
2.Handle null values and encode categorical columns.
3.Split data into training and testing sets.
4.Train a DecisionTreeClassifier using entropy.
5.Predict and evaluate the model using accuracy and metrics.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: M ALMAAS JAHAAN
RegisterNumber:  212224230016
*/

import pandas as pd
df=pd.read_csv("/content/Employee.csv")
print("data.head():")
df.head()
print("data.info()")
df.info()
print("data.isnull().sum()")
df.isnull().sum()
print("data value counts")
df["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
df["salary"]=le.fit_transform(df["salary"])
df.head()
print("x.head():")
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
print("Data prediction")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt,filled=True,feature_names=x.columns,class_names=['salary' , 'left'])
plt.show()
```

## Output:
![decision tree classifier model](sam.png)

![image](https://github.com/user-attachments/assets/62bcbfc9-7dfc-408a-9ca1-e5b14c8f7ae5)
![image](https://github.com/user-attachments/assets/d59d22db-eced-45e8-bb19-6148b7df3c9a)
![image](https://github.com/user-attachments/assets/0fc8bb95-b9ec-412f-92fd-58495420b35b)
![image](https://github.com/user-attachments/assets/a6ad9b8b-5845-4057-a398-e4bc109c4ba0)
![image](https://github.com/user-attachments/assets/e0def7d8-1ba3-46d9-b495-e7f878e82e80)
![image](https://github.com/user-attachments/assets/d9a4e56c-6a4e-4c1d-9634-17eef7aab56e)
![image](https://github.com/user-attachments/assets/133dfc39-109f-40ec-9295-405f5adafcfb)
![image](https://github.com/user-attachments/assets/bcff86bd-e6af-4d55-b68b-6752f42df3cc)
![image](https://github.com/user-attachments/assets/37548fb5-eafd-4ce9-8a52-faf724e1f1c5)
![image](https://github.com/user-attachments/assets/0e7e8186-9aab-4e37-8443-48b680dccf4b)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
