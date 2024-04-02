# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KALPANA S
RegisterNumber: 212222040069 
*/
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
####DATASET
![image](https://github.com/Kalpanareshma/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122040453/af1469aa-2c9a-463b-b589-4b90cccfae02)
####data.info()
![image](https://github.com/Kalpanareshma/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122040453/45a77edc-0266-4ec0-868a-893ee0370bda)
####CHECKING IF NULL VALUES ARE PRESENT
![image](https://github.com/Kalpanareshma/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122040453/241e8b8d-7703-48d3-8ee3-63272c79f412)
####VALUE_COUNTS()
![image](https://github.com/Kalpanareshma/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122040453/0f7a04b1-72a2-48cb-842f-72bbc70603d6)
####DATASET AFTER ENCODING
![image](https://github.com/Kalpanareshma/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122040453/77b11385-396e-4a53-adcd-a9f01af0debf)
####X-VALUES
![image](https://github.com/Kalpanareshma/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122040453/8017c85b-691b-4c5d-a92e-b2c10c38f3f5)
####ACCURACY
![image](https://github.com/Kalpanareshma/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122040453/40625b9c-382b-4f0d-924e-a2e57af41e32)
####dt.predict()
![image](https://github.com/Kalpanareshma/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/122040453/b5d330bc-405d-4204-bed6-7ee777857dde)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
