import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def data_extract(url,file):
    print('Data extraction ...')
    #Data extraction from csv file
    data = pd.read_csv(url+file)
    #Imbalance the data
    fraud = data[data['Class'] == 1] 
    valid = data[data['Class'] == 0] 
    outlierFraction = len(fraud)/float(len(valid)) 
    print(outlierFraction)
    print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))) 
    print('Valid Transactions: {}'.format(len(data[data['Class'] == 0]))) 
    #plotting the correlation matrix
    corrmat = data.corr() 
    fig = plt.figure(figsize = (12, 9)) 
    sns.heatmap(corrmat, vmax = .8, square = True)
    plt.savefig('plots\\corr_matrix.png')
    return data, fraud, valid

from sklearn.model_selection import train_test_split
def data_prep(data):
    print('Data preparation ...')
    #separating the X and the Y values
    X = data.drop(['Class'], axis = 1) 
    Y = data["Class"] 
    print(X.shape) 
    print(Y.shape)

    # getting just the values for the sake of processing  
    # (its a numpy array with no columns) 
    xData = X.values 
    yData = Y.values 

    #split the data into training and testing sets 
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 42)

    return xTrain, xTest, yTrain, yTest

from sklearn.ensemble import RandomForestClassifier 
def model_train(xTrain,yTrain):
    print('Model Training ...')
    #random forest model creation 
    rfc = RandomForestClassifier(n_jobs=-1) 
    rfc.fit(xTrain, yTrain) 
    
    return rfc

from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef 
from sklearn.metrics import confusion_matrix 
def model_eval(yPred,yTest):
#building all kinds of evaluating parameters    
    print("The model used is Random Forest classifier") 
    
    acc = accuracy_score(yTest, yPred) 
    print("The accuracy is {}".format(acc)) 
    prec = precision_score(yTest, yPred) 
    print("The precision is {}".format(prec)) 
    rec = recall_score(yTest, yPred) 
    print("The recall is {}".format(rec)) 
    f1 = f1_score(yTest, yPred) 
    print("The F1-Score is {}".format(f1)) 
    MCC = matthews_corrcoef(yTest, yPred) 
    print("The Matthews correlation coefficient is{}".format(MCC)) 

    #visulalizing the confusion matrix
    LABELS = ['Normal', 'Fraud'] 
    conf_matrix = confusion_matrix(yTest, yPred) 
    plt.figure(figsize =(12, 12)) 
    sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d"); 
    plt.title("Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.savefig('plots\\conf_matrix.png')
