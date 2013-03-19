from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import util
import random

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
    values = train_fea['SaleYear'].values
    years = set(values)
    mini =  values.min()
    train = 0 
    test = 0
    for year in years:
        tmp = train_fea[train_fea['SaleYear']==year]
        size = tmp.shape[0]
        sample_s = size/n
        rows = random.sample(tmp.index, sample_s)
        test_part = tmp.ix[rows]
        train_part = tmp.drop(rows)
        if mini == year:
            train = train_part
            test = test_part
            print train.shape
            print test.shape
        else:
            train = train + train_part
            test = test + test_part                        
            print train.shape
            print test.shape

    return train, test

#rf = RandomForestRegressor(n_estimators=50, n_jobs=1, compute_importances = True)
#rf.fit(train_fea, train["SalePrice"])
#predictions = rf.predict(test_fea)

#util.write_submission("result.csv", predictions)
