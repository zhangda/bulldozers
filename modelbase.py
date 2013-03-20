from collections import defaultdict
import numpy as np
import pandas as pd
import util,random, types
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

def get_date_dataframe(date_column):
    return pd.DataFrame({
        "SaleYear": [d.year for d in date_column],
        "SaleMonth": [d.month for d in date_column],
        "SaleDay": [d.day for d in date_column]
        }, index=date_column.index)

def clean_df():
    train, test = util.get_train_test_df()
    columns = set(train.columns)
    columns.remove("SalesID")
    columns.remove("SalePrice")
    columns.remove("saledate")
    train_fea = get_date_dataframe(train["saledate"])
    test_fea = get_date_dataframe(test["saledate"])
    for col in columns:
        if train[col].dtype == np.dtype('object'):
            s = np.unique(train[col].fillna(-1).values)
            mapping = pd.Series([x[0] for x in enumerate(s)], index = s)
            train_fea = train_fea.join(train[col].map(mapping).fillna(-1))
            test_fea = test_fea.join(test[col].map(mapping).fillna(-1))
        else:
            train_fea = train_fea.join(train[col].fillna(0))
            test_fea = test_fea.join(test[col].fillna(0))
    train_fea = train_fea.join(train['SalePrice']) 
    return train_fea, test_fea

def split_train(train_fea, n):
    years = set(train_fea['SaleYear'].values)
    train, test = None, None 
    for year in years:
        tmp = train_fea[train_fea['SaleYear']==year]
        size = tmp.shape[0]
        sample_s = size/n
        rows = random.sample(tmp.index, sample_s)
        test_part = tmp.ix[rows]
        train_part = tmp.drop(rows)
        if type(train) == types.NoneType:
            train,test = train_part, test_part
        else:
            train = train.append(train_part)
            test = test.append(test_part)   
    return train, test

def train(algos, train_fea, test_fea, nloop, nfold):
    validation = None
    prediction = None
    for i in range(0, nloop):
        train, test = split_train(train_fea, nfold)
        train = train.set_index([range(0,train.shape[0])])
        test = test.set_index([range(0,test.shape[0])])
        train_label, test_label = train['SalePrice'], test['SalePrice']
        test_label_table = pd.DataFrame(test_label)
        del train['SalePrice']
        del test['SalePrice']
        for algo in algos:
            algo.fit(train, train_label)
            v = algo.predict(test)
            p = algo.predict(test_fea)
            if type(validation) == types.NoneType:
                validation = pd.DataFrame({str(i): v})
            else:
                validation = validation.join(pd.DataFrame({str(i): v}))
            if type(prediction) == types.NoneType:
                prediction = pd.DataFrame({str(i): p})
            else:
                prediction = prediction.join(pd.DataFrame({str(i): p}))
        validation = validation.join(test_label_table)
    return validation, prediction



    
    return prediction
       
    #rf = RandomForestRegressor(n_estimators=10, n_jobs=1, compute_importances = True)
    #et = ExtraTreesRegressor(n_estimators=100, n_jobs=1, compute_importances = True)
    #gb = GradientBoostingRegressor(n_estimators=100)
    
#util.write_submission("result.csv", predictions)
