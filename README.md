# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HariDharshini.S
RegisterNumber:  212221230033
*/
```
```
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### Original data(first five columns):
![output1](https://user-images.githubusercontent.com/94168395/202351081-a061638b-33e3-4e8a-a476-628db8795873.png)

### Data after dropping unwanted columns:
![output2](https://user-images.githubusercontent.com/94168395/202351147-17439149-7b3d-44fe-869e-a537cf2ae7b6.png)

### null values: 
![output3](https://user-images.githubusercontent.com/94168395/202351273-1349d8b0-df71-4bd0-90bd-25cfc138f8de.png)

### duplicated values:
![output4](https://user-images.githubusercontent.com/94168395/202351317-8f030551-84a9-4553-abaa-bbf0bd136209.jpg)

### Data after Encoding:
![output5](https://user-images.githubusercontent.com/94168395/202351369-985f9b55-114c-4cf4-aea6-49340882dd59.jpg)

### X-data:
![output6](https://user-images.githubusercontent.com/94168395/202351430-09bf7f81-ad19-46be-a86e-b8a374d44340.jpg)

### Y-data:
![output7](https://user-images.githubusercontent.com/94168395/202351512-0fec7fb6-fd87-4207-86e1-c0dafc37025f.jpg)

### preidcted values:
![output8](https://user-images.githubusercontent.com/94168395/202351576-106b4916-9c01-4751-850c-830b1f584c53.jpg)

### Accuracy score:
![output9](https://user-images.githubusercontent.com/94168395/202351638-b5e1d891-4a79-4a6e-ae80-ced96af74b35.jpg)

### Confusion matrix:
![output10](https://user-images.githubusercontent.com/94168395/202351684-61f984be-33aa-40ba-9d4c-f5cfd0babc98.jpg)

### Clasification:
![output11](https://user-images.githubusercontent.com/94168395/202351742-20f792d2-8111-4659-a6ed-ae4d42153537.jpg)

### predicting output from regression model:
![output12](https://user-images.githubusercontent.com/94168395/202351784-d4257516-f545-4988-8d9d-d903cfc29b14.jpg)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
