# import dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#Data Collection and Processing

# loading the csv data to pandas DataFrame
heart_data = pd.read_csv('heart.csv')

# print first 5 rows of the dataset
print("First 5 rows of data")
print(heart_data.head())

print("----------------------------------------------------------------")
print("Last 5 rows of data")
print(heart_data.tail())

# number of rows and columns in the dataset

print("Number of rows and columns")
print(heart_data.shape)


#Getting info about the data
print("-----------------------------------------------------------------")
print("Info about the data :")
print(heart_data.info())


# Checking for missing values
heart_data.isnull().sum()

# statistical measures about the data

print(heart_data.describe())

# check the distribution of target variable

print(heart_data['target'].value_counts())

# 1 -> defective heart
# 0 -> healthy heart

# Splitiing the features and target

X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']

print(X)
print(Y)

# Splitting the data into training data and test data

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size= 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

#Model Training 
#Logistic Regression 
model=LogisticRegression(max_iter=3000)

# 1- tarining the LogisticRegression model with training data
model.fit(X_train, Y_train)

# Accuracy score

#on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

#test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

#Building a Predictive system

input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

print(prediction)

if(prediction[0] == 0):
       print("Healthy heart")
else:
  print("Patient have Heart disease")



