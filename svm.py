import numpy as np # it helps user to do oprations with array
import pandas as pd # it is used for data manupulation and analysis
# this library heps in ml projects -provide tools for ml and statistical modeling like clustering,classification
from sklearn.preprocessing import StandardScaler #standard scaler : it standardise the data value instandard value
from sklearn.model_selection import train_test_split #
from sklearn import svm #support vector machine : it is the supervised ml model 
from sklearn.metrics import accuracy_score # compute accuracy
import pickle #used to save and load model
import warnings
warnings.filterwarnings('ignore')



# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv') #pima india

# printing the first 5 rows of the dataset
diabetes_dataset.head()

# number of rows and Columns in this dataset
diabetes_dataset.shape

# getting the statistical measures of the data
diabetes_dataset.describe() #describing data



diabetes_dataset['Outcome'].value_counts()


diabetes_dataset.groupby('Outcome').mean() #calc mean value for 0 and 1

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1) #dropping outcome coloum sand storing it in x
Y = diabetes_dataset['Outcome'] #storing it in y

# print(X)

# print(Y)

"""Data Standardization"""

scaler = StandardScaler() #standard scaler : it standardise the data value instandard value

scaler.fit(X)

standardized_data = scaler.transform(X)

# print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

# print(X)
# print(Y)

"""Train Test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

# print(X.shape, X_train.shape, X_test.shape)

"""Training the Model"""

classifier = svm.SVC()

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

"""Model Evaluation

Accuracy Score
"""

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# print('Accuracy score of the test data : ', test_data_accuracy)

"""Making a Predictive System"""

input_data = (1,166,72,19,175,30,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
# print(std_data)

prediction = classifier.predict(std_data)
# print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  input_data=list(input_data)
  input_data[5]=21.7
  input_data=tuple(input_data)
  input_data_as_numpy_array = np.asarray(input_data)
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
  std_data = scaler.transform(input_data_reshaped)
  prediction = classifier.predict(std_data)
  if(prediction[0]==0):
    print('The person is diabetic')
    print('If the BMI was within the healthy range 18.5 to 24.9.The chances of Diabetes would be greatly reduced')
  else:
    print('The person is diabetic')

"""save the model to disk

"""

# filename = 'svm.sav'
# pickle.dump(classifier, open(filename, 'wb'))

# """load the modal"""

# loaded_model = pickle.load(open('svm.sav', 'rb'))
# input_data = (1,166,72,19,175,30,0.587,51)
# input_data_as_numpy_array = np.asarray(input_data)
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# std_data = scaler.transform(input_data_reshaped)

# result = loaded_model.predict(std_data)
# print(result)