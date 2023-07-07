import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score,roc_auc_score
from sklearn.model_selection import  train_test_split

import argparse
# for this argparse we hav to create an object separatly as test.py


def get_data():
    URL="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


    #reading the data as df
    try:
        df=pd.read_csv(URL,sep=";")
        return df
    except Exception as e:
        raise e
#its for regrression    
def evaluate(y_true,y_pred):
    '''mae=mean_absolute_error(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    r2=r2_score(y_true,y_pred)
    return mae,mse,rmse,r2'''

    accuracy=accuracy_score(y_true,y_pred)
    return accuracy

    
def main(n_estimators,max_depth):
    df=get_data()
    #print(df)
    train,test=train_test_split(df)
    
    #its random split
    #train test split with raw data
    x_train=train.drop(['quality'],axis=1)
    x_test=test.drop(['quality'],axis=1)
    #here quality is a target column,depended

    y_train=train[['quality']]
    y_test=test[['quality']]
    
    #model training
    '''lr=ElasticNet()
    lr.fit(x_train,y_train)
    pred=lr.predict(x_test)'''
    
    rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    rf.fit(x_train, y_train)
    pred=rf.predict(x_test)
    
    
    #evaluate the model
    #mae,mse,rmse,r2=evaluate(y_test,pred)
    accuracy=evaluate(y_test,pred)
    #print(f"mean absolute error {mae}, mean squared error {mse}, root mean squared error {rmse}, r2_score {r2}")
    print(f"accuracy: {accuracy}")
    
    


if __name__=='__main__':
    
    args=argparse.ArgumentParser()
    # reference from sklearn.ensemble.RandomForestClassifier check it at web
    args.add_argument("--n_estimators", "-n", default="50", type=int)
    args.add_argument("--max_depth", "-m", default=5, type=int)
    parse_args=args.parse_args()
    #we can access from cmd


    try:
        main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)
    except Exception as e:
        raise e            

