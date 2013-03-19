from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import util, random, types, sys

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

def get_pairs(years, step):
    result = []
    for i in range(0, len(years)-step):
        result.append((years[i],years[i+step])) 
    return result

def train_year(train_fea, trees):
    values = train_fea['SaleYear'].values
    years = sorted(list(set(values)))
    rfs =[]
    for i in range(0, len(years)):
        print 'train model %d' % (years[i])
        rf = RandomForestRegressor(n_estimators=trees, n_jobs=1, compute_importances = True)
        y = train_fea[train_fea['SaleYear']==years[i]]
        y_fea = y.copy()
        del y_fea['SalePrice']
        rf.fit(y_fea, y["SalePrice"])
        rfs.append(rf)
    errors = None
    for i in range(1, len(years)):
        pairs = get_pairs(years, i)
        for p in pairs:
            print 'compare %d, %d' % (p[0], p[1])
            y1 = train_fea[train_fea['SaleYear']==p[0]]
            y2 = train_fea[train_fea['SaleYear']==p[1]]
            y1_fea, y2_fea = y1.copy(), y2.copy()
            del y1_fea['SalePrice']
            del y2_fea['SalePrice']
            rf = rfs[years.index(p[0])]
            y2_p = rf.predict(y2_fea)
            y2_r = np.array([v for v in y2['SalePrice']])
            error_rates = np.array(map(lambda x,y: (x-y)/y, y2_p, y2_r))
            if type(errors)==types.NoneType:
                errors = pd.DataFrame({'dist':i, 'mean':error_rates.mean(), 'var':error_rates.var(), 'std':error_rates.std()}, index=[i])
            else:
                errors = errors.append(pd.DataFrame({'dist':i, 'mean':error_rates.mean(), 'var':error_rates.var(), 'std':error_rates.std()}, index=[i]))
    errors_list = []
    for i in range(1, len(years)):
        errors_list.append(errors.ix[i]['mean'].mean())
    return rfs, errors_list

def predict(test_fea, rfs, errors):
    error_sum = np.array(errors).sum()
    p = None
    for i in range(0,len(errors)):
        if i==0:
            p = rfs[i+1].predict(test_fea)*(errors[i]/error_sum)
        else:
            p = p + rfs[i+1].predict(test_fea)*(errors[i]/error_sum)
    
    util.write_submission("result.csv", np.array(p))

def main():
    trees = int(sys.argv[1])
    train_fea, test_fea = clean_df()
    rfs, errors = train_year(train_fea, trees)
    predict(test_fea, rfs, errors)

if __name__ == "__main__":
    main()

