import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def split_data(filename):
    df = pd.read_csv(filename)
    df = df.drop('Unnamed: 0', axis=1)
    county_info = df.copy()
    y = df.pop('bars')
    X = df.drop(['geo_id', 'county_name'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)
    # print(type(X_train.iloc[250]))
    return(X_train, X_test, y_train, y_test, county_info)


def lin_reg(X_train, X_test, y_train, y_test):
    lr = LinearRegression(fit_intercept=True)
    global model
    model = lr.fit(X_train,y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return(model, train_score, test_score)

def predict_bars(county, state):
    number = 2000
    X = np.asarray(county_info.iloc[number][['area_sqmi', 'pop_est_2015', 'hotels']]).reshape(1,-1)
    # print(type(X))
    y = county_info.iloc[number]['bars']
    y_pred = model.predict(X)[0]
    # y=2
    # y_pred=1.5
    return(y_pred, y, y_pred-y)


if __name__ == '__main__':
    filename = '../data/2015_toy_sd_1_5_nan_to_min.csv'
    X_train, X_test, y_train, y_test, county_info = split_data(filename)
    lr_model, lr_train_score, lr_test_score = lin_reg(X_train, X_test, y_train, y_test)
    print('Linear Regression Score\nTrain: {0}\nTest: {1}\n'. format(lr_train_score, lr_test_score))
    print('Number of bars the model says it suports: {0}\nReal number of bars: {1}\nDifference: {2}\n'.format(predict_bars('Boulder', 'CO')[0], int(predict_bars('Boulder', 'CO')[1]), predict_bars('Boulder', 'CO')[2]))
