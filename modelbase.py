from collections import defaultdict
import numpy as np
import pandas as pd
import util,random, types
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

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
        print 'loop %d' % (i)
        train, test = split_train(train_fea, nfold)
        train = train.set_index([range(0,train.shape[0])])
        test = test.set_index([range(0,test.shape[0])])
        train_label, test_label = train['SalePrice'], test['SalePrice']
        test_label_table = pd.DataFrame(test_label)
        del train['SalePrice']
        del test['SalePrice']
        tmp_v = None
        tmp_p = None
        for j, algo in enumerate(algos):
            print 'algo %d' % (j)
            algo.fit(train, train_label)
            v = algo.predict(test)
            p = algo.predict(test_fea)
            if type(tmp_v) == types.NoneType:
                tmp_v, tmp_p = pd.DataFrame({j: v}), pd.DataFrame({j: p}, index=None)
            else:
                tmp_v = tmp_v.join(pd.DataFrame({j: v}, index=None))
                tmp_p = tmp_p.join(pd.DataFrame({j: p}, index=None))
        tmp_v = tmp_v.join(test_label_table)
        if type(validation) == types.NoneType:
            validation, prediction = tmp_v, tmp_p
        else:
            validation = validation.append(tmp_v)
            prediction = prediction + tmp_p
    return validation, prediction/nloop

def predict(validation, prediction):
    lr = LinearRegression()
    v_label = validation['SalePrice']
    del validation['SalePrice']
    lr.fit(validation, v_label)
    p = lr.predict(prediction)
    util.write_submission("result.csv", p)
       
   
def main():
    train_fea, test_fea = clean_df()
    rf1 = RandomForestRegressor(n_estimators=50, n_jobs=-1)
    rf2 = RandomForestRegressor(n_estimators=50, n_jobs=-1)
    et1 = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
    et2 = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
    #gb1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=10)
    #gb2 = GradientBoostingRegressor(n_estimators=100,learning_rate=0.05, max_depth=10)
    v, p = train([rf1, rf2, et1, et2], train_fea, test_fea, 2, 5)
    predict(v, p)


if __name__ == "__main__":
    main()

