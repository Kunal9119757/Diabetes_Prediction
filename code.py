import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes_unclean.csv') 

diabetes_dataset.head()

diabetes_dataset.tail()

# number of rows and Columns in this dataset
diabetes_dataset.shape

diabetes_dataset.info()

#give total sum of null values in each column
diabetes_dataset.isna().sum()

diabetes_dataset.isnull()

#DATA CLEANING


diabetes_dataset['AGE'] = diabetes_dataset['AGE'].interpolate(method='linear')
diabetes_dataset['Urea']=diabetes_dataset['Urea'].interpolate(method='linear')
diabetes_dataset['Cr'] = diabetes_dataset['Cr'].interpolate(method='linear')
diabetes_dataset['HbA1c'] = diabetes_dataset['HbA1c'].interpolate(method='linear')
diabetes_dataset['Chol'] = diabetes_dataset['Chol'].interpolate(method='linear')
diabetes_dataset['TG'] = diabetes_dataset['TG'].interpolate(method='linear')
diabetes_dataset['HDL'] = diabetes_dataset['HDL'].interpolate(method='linear')
diabetes_dataset['LDL'] = diabetes_dataset['LDL'].interpolate(method='linear')
diabetes_dataset['VLDL'] = diabetes_dataset['VLDL'].interpolate(method='linear')


diabetes_dataset.isna().sum()

diabetes_dataset.info()


# getting the statistical measures of the data
diabetes_dataset.describe()

# id column and id_pation do effect our dataset so drop them
diabetes_dataset = diabetes_dataset.drop(columns="ID")
diabetes_dataset = diabetes_dataset.drop(columns="No_Pation")

diabetes_dataset['Gender'] = diabetes_dataset['Gender'].str.replace('f','F')
diabetes_dataset['Gender'] =diabetes_dataset['Gender'].str.replace('m','M')

#get count of number of diabetic and non-diabetic 
diabetes_dataset['CLASS'].value_counts()

# important line
#mapping yes with 1 and no with 0
diabetes_dataset['CLASS'] = diabetes_dataset['CLASS'].map({'Y': 1, 'N': 0})

#mapping M with 1 and female with 0
diabetes_dataset['Gender'] = diabetes_dataset['Gender'].map({'M': 1, 'F': 0})

#get count of number of malignant and benign
diabetes_dataset['CLASS'].value_counts()

# 1 --> Diabetic

# 0 --> Non-Diabetic

diabetes_dataset.tail()

diabetes_dataset

#visualize co-relation using heatmap
plt.figure(figsize=(10,10))
sns.heatmap(diabetes_dataset.iloc[:,0:13].corr() , annot = True , fmt = ".0%")

diabetes_dataset.groupby('CLASS').mean()

# separating the data and labels
X = diabetes_dataset.drop(columns = 'CLASS', axis=1)
Y = diabetes_dataset['CLASS']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

#SVM
#Linear Kernel
classifier = svm.SVC(kernel='linear')
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

# accuracy score on the test data
print(classification_report(Y_test,classifier.predict(X_test)))
print('Accuracy : ',accuracy_score(Y_test,classifier.predict(X_test)))


# poly kernel
classifierpoly = svm.SVC(kernel='poly')
#training the support vector Machine Classifier
classifierpoly.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = classifierpoly.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifierpoly.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

# accuracy score on the test data
print(classification_report(Y_test,classifierpoly.predict(X_test)))
print('Accuracy : ',accuracy_score(Y_test,classifierpoly.predict(X_test)))


#rbf
classifierrbf = svm.SVC(kernel='rbf')
#training the support vector Machine Classifier
classifierrbf.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = classifierrbf.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)


# accuracy score on the test data
X_test_prediction = classifierrbf.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)


# accuracy score on the test data
print(classification_report(Y_test,classifierrbf.predict(X_test)))
print('Accuracy : ',accuracy_score(Y_test,classifierrbf.predict(X_test)))


# Logistic Regression
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(random_state=0)
log.fit(X_train,Y_train)

print('logistic regression accuracy:',log.score(X_train,Y_train))

# accuracy score on the test data
print(classification_report(Y_test,log.predict(X_test)))
print('Accuracy : ',accuracy_score(Y_test,log.predict(X_test)))

# decision tree

from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(random_state=0,criterion="entropy")
tree.fit(X_train,Y_train)

print('Decision tree accuracy:',tree.score(X_train,Y_train))

# accuracy score on the test data
print(classification_report(Y_test,tree.predict(X_test)))
print('Accuracy : ',accuracy_score(Y_test,tree.predict(X_test)))


# Random forest

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
forest.fit(X_train,Y_train)

print('Random forest accuracy:',forest.score(X_train,Y_train))

# accuracy score on the test data
print(classification_report(Y_test,forest.predict(X_test)))
print('Accuracy : ',accuracy_score(Y_test,forest.predict(X_test)))


# using random forest as my training algorithim

#female
# input_data = (0,50,4.7,46,4.9,4.2,0.9,2.4,1.4,0.5,24)
#male
input_data = (1,50,9.6,203,5.4,3.8,5.9,0.5,4.3,1.3,22)

#female
# input_data = (0,45,3.7,49,5.3,2.5,2.2,1,0.6,1,25)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# std_data =scaler.transform(input_data_reshaped)
# print(std_data)

prediction = forest.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

#saving trained model
import pickle
pickle.dump(forest ,open('model.pkl' , 'wb'))























