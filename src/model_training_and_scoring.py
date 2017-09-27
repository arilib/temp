import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

def split_data(filename):
    df = pd.read_csv(filename)
    df = df.drop('Unnamed: 0', axis=1)
    county_info = df.copy()
    y = df.pop('bars')
    X = df.drop(['geo_id', 'county_name'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)
    cols = X_train.columns
    return(X_train, X_test, y_train, y_test, county_info, cols)


def lin_reg(X_train, X_test, y_train, y_test):
    lr = LinearRegression(fit_intercept=True)
    model = lr.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)
    return(model, train_score, test_score)


def predict_bars(county, state, model, cols):
    number = 250
    X = np.asarray(county_info.iloc[number][cols]).reshape(1,-1)
    y = county_info.iloc[number]['bars']
    y_pred = model.predict(X)[0]
    return(y_pred, int(y))


def geo_id_finder(county, state, df=county_info):
    if len(state) == 2:
        state_df = df[df['state_code'] == county.lower().title()]
    else:
        state_df = df[df['state_name'] == county.lower().title()]
    df[df['county_name'] == county.lower().title()]

    # To check if it helps:
    # df2.iloc[np.where(df2["ALAND_SQMI"] == df2['ALAND_SQMI'].max())]


    # state_df.index[state_df['county_name'] == 'Boulder County, CO'].tolist()[0] gives you the index 250
    # state_df.get_value(250, 'geo_id')    Out[158]: 8013 gives you the geo_id

    #this other needs tunning
    # select_index = list(np.where(df["county_name"] == 'Boulder County, CO)
    #  ...: [0])
    #  ...: df.iloc[select_index]


if __name__ == '__main__':
    filename = '../data/2015_toy_sd_1_5_nan_to_min.csv'
    X_train, X_test, y_train, y_test, county_info, cols = split_data(filename)
    lr_model, lr_train_score, lr_test_score = lin_reg(X_train, X_test, y_train, y_test)
    print('Linear Regression Score\nTrain: {0}\nTest: {1}\n'. format(lr_train_score, lr_test_score))
    pred_y, actual_y = predict_bars('Boulder', 'CO', lr_model, cols)
    print('Number of bars the model says it suports: {0}\nReal number of bars: {1}\nDifference: {2}\n'.format(pred_y, actual_y, pred_y - actual_y))
