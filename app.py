from unicodedata import numeric
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/portfolio-details')
def portfolio_details():
    return render_template('portfolio-details.html')

@app.route('/sample')
def sample():
    return render_template('sample.html')

@app.route('/sample', methods=['POST'])
def sample_output():
    if request.method == 'POST':

        import numpy as np 
        import pandas as pd 
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split #
        from sklearn import svm 
        from sklearn.metrics import accuracy_score
        import warnings
        warnings.filterwarnings('ignore')

        # sample = request.form['gender']
        # if sample == 0:
        #     Pregnancies = 0
        # else:
        Pregnancies=0
        if  request.form['Pregnancies'].isnumeric():
            Pregnancies=request.form['Pregnancies']

        # # if Pregnancies != numeric:
        # #     Pregnancies = 0
        glucose = request.form['glucose']
        BloodPressure = request.form['BloodPressure']
        
        SkinThickness=22
        if request.form['SkinThickness'].isnumeric():
            SkinThickness=request.form['SkinThickness']

        insulin = request.form['insulin']
        weight= int(request.form['weight'])
        height= int(request.form['height'])
        bmi=(weight*10000)/((height)**2)
        Pedegree = 0.5108
        if  request.form['pedegreeFunction'].isnumeric():
            Pedegree=request.form['pedegreeFunction']
        
        age = request.form['age']

        

        
            
        # input_1 = request.form['input_1']
        # input_2 = request.form['input_2']
        # input_3 = request.form['input_3']
        # input_4 = request.form['input_4']
        # input_5 = request.form['input_5']
        # input_6 = request.form['input_6']
        # input_7 = request.form['input_7']
        # input_8 = request.form['input_8']

        diabetes_dataset = pd.read_csv('diabetes.csv')
        diabetes_dataset.head()
        diabetes_dataset.shape
        diabetes_dataset.describe() 
        diabetes_dataset['Outcome'].value_counts()
        diabetes_dataset.groupby('Outcome').mean() 
        
        X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
        Y = diabetes_dataset['Outcome'] 

        scaler = StandardScaler()
        scaler.fit(X)
        standardized_data = scaler.transform(X)

        X = standardized_data
        Y = diabetes_dataset['Outcome']
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

        classifier = svm.SVC()
        classifier.fit(X_train, Y_train)

        X_train_prediction = classifier.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        X_test_prediction = classifier.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

        # input_data = (1,166,72,19,175,30,0.587,51)
        input_data = (Pregnancies,glucose,BloodPressure,SkinThickness,insulin,bmi,Pedegree,age)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        std_data = scaler.transform(input_data_reshaped)
        prediction = classifier.predict(std_data)
        if (prediction[0] == 0):
            out = "0"
            return render_template('output_negative.html', out=out)
        else:
            input_data=list(input_data)
            input_data[5]=21.7
            input_data=tuple(input_data)
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
            std_data = scaler.transform(input_data_reshaped)
            prediction = classifier.predict(std_data)
            if(prediction[0]==0):
                out = "1"
                return render_template('output_positive.html', out=out)
            else:
                out = "1"
                return render_template('output_positive.html', out=out)

        # return render_template('output.html', out=out)

@app.route('/form')
def form():
    return render_template('inner-page.html')

if __name__ == "__main__":
    app.run(debug=False)
