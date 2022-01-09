import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def index():
    dataset = pd.read_csv('Purpose-of-Visit-by-Port-and-Country-JTB-Open-data-request.csv')
    dataset.drop('Number of Visitors', inplace=True, axis=1)

    dataset = dataset.dropna()

    le = LabelEncoder()
    le.fit(dataset.loc[:, ('Port of Entry')])
    dataset.loc[:, ('Port of Entry')] = le.transform(dataset.loc[:, ('Port of Entry')])
    portEntry = dataset['Port of Entry'].unique()
    portEntryName = le.inverse_transform(portEntry)
    zipbObj = zip(portEntryName, portEntry)
    portEntryDict = dict(zipbObj)

    le.fit(dataset.loc[:, ('Type of Visit')])
    dataset.loc[:, ('Type of Visit')] = le.transform(dataset.loc[:, ('Type of Visit')])
    visitType = dataset['Type of Visit'].unique()
    visitTypeName = le.inverse_transform(visitType)
    zipbObj = zip(visitTypeName, visitType)
    visitTypeDict = dict(zipbObj)

    le.fit(dataset.loc[:, ('Origin')])
    dataset.loc[:, ('Origin')] = le.transform(dataset.loc[:, ('Origin')])
    origin = dataset['Origin'].unique()
    originName = le.inverse_transform(origin)
    zipbObj = zip(originName, origin)
    originDict = dict(zipbObj)

    le.fit(dataset.loc[:, ('Month')])
    dataset.loc[:, ('Month')] = le.transform(dataset.loc[:, ('Month')])
    month = dataset['Month'].unique()
    monthName = le.inverse_transform(month)
    zipbObj = zip(monthName, month)
    monthDict = dict(zipbObj)    
    
    return render_template('index.html', dict1 = portEntryDict, dict2 = visitTypeDict, dict3 = originDict, dict4 = monthDict)

@app.route("/predictedVisitor", methods=['POST'])
def predict():
    from joblib import load
    with open('model.pkl', 'rb') as file:
        model = load(file)

    portEntry = request.form['portEntry']
    visitType = request.form['visitType']
    origin = request.form['origin']
    month = request.form['month']
    
    input = [[portEntry, visitType, origin, month]]
    predict = model.predict(input)
    return render_template('results.html', predictData = predict)

if __name__ == "__main__":
    app.run()
