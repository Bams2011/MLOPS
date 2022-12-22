# import libraries
import joblib
from functions import data_extract, data_prep,model_train,model_eval

# Import Data and perform Data quality/Data statistical analysis  
url = "data\\"
file = "creditcard.csv"
data, fraud, valid = data_extract(url,file)

# Data preparation
xTrain, xTest, yTrain, yTest = data_prep(data)

# Model training
rfc = model_train(xTrain,yTrain)

# Model saving
print('Saving model ...')
joblib.dump(rfc, 'trained_model\\rfc.pkl', compress=9)


# Model prediction
yPred = rfc.predict(xTest)

# Model Evaluation & Validation
model_eval(yPred,yTest)




