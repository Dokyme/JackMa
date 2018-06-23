import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.stattools as st
import statsmodels.tsa.x13 as x13
import numpy as np
from util import *
from sklearn import metrics

pmax = 5
qmax = 5
diffmax = 3


def train_test(field):
    alldata = get_labeled_value(tp="all", field=field)
    train = get_labeled_value(field=field)
    test = get_labeled_value(tp="test", field=field)
    order = paramaters_auto_optimization(train)
    model, results = fit_arima(train, order[0], 1, order[1])
    fitted = restore_diff(train, results.fittedvalues)
    plt.figure(figsize=(16, 7))
    plt.plot(train, label="labeled")
    plt.plot(fitted, label="fitted")
    plt.legend()
    plt.show()
    print(metrics.mean_squared_error(fitted, train))
    print(np.sqrt(metrics.mean_squared_error(fitted, train)))
    predicts, stderr, conf = results.forecast(31)
    predicts = pd.Series(predicts, index=test.index)
    plt.figure(figsize=(16, 7))
    plt.plot(test, label="labeled")
    plt.plot(predicts, label="predict")
    plt.legend()
    plt.show()
    print(metrics.mean_squared_error(predicts, test))
    print(np.sqrt(metrics.mean_squared_error(predicts, test)))


def train_predict(field, p, d, q):
    train = get_labeled_value(tp="all", field=field)
    model, result = fit_arima(train, p, d, q)
    predicts, stderr, conf = result.forecast(30)
    drange = pd.date_range("2014-09-01", periods=30)
    return restore_scale(pd.Series(predicts, index=drange))


def paramaters_auto_optimization(series):
    diff_series = series.diff(1).dropna()
    order = st.arma_order_select_ic(
        diff_series, pmax, qmax, ic=["bic", "aic", "hqic"])
    print(order)
    return order.aic_min_order


def fit_arima(series, p, d, q):
    model = ARIMA(series, order=(p, d, q), freq="D")
    results = model.fit(disp=-1)
    return model, results


def restore_diff(series, diff_series):
    t = pd.Series([series[0]],index=[series.index[0]]).append(diff_series).cumsum()
    return t


train_test("Interest_1_W")

# tpa all 4,3
# tra all 2,3
# paramaters_auto_optimization(get_labeled_value(tp="all",field="total_redeem_amt"))


# tpa = train_predict("total_purchase_amt", 4, 1, 3)
# tra = train_predict("total_redeem_amt", 2, 1, 3)
# tpatra = pd.DataFrame(data={"date": pd.date_range(
#     '2014-09-01', '2014-09-30', freq='D'), "tpa": tpa, "tra": tra})
# tpatra = tpatra.set_index("date")
# tpatra[["tpa", "tra"]] = tpatra[["tpa", "tra"]].astype(int)
# tpatra.to_csv("data/tc_comp_predict_table.csv", date_format="%Y%m%d")

# tpa_train = get_labeled_value()
# tpa_test = get_labeled_value("test")
# # paramaters_auto_optimization(tpa_train)
# model, results = fit_arima(tpa_train, 3, 1, 3)
# predicts, stderr, conf = results.forecast(31)
# predicts = pd.DataFrame({"utpa": predicts}, index=tpa_test.index)
# predicts.to_csv("data/predict_utpa.csv")
# plt.figure(figsize=(16, 7))
# plt.plot(tpa_test, label="labeled")
# plt.plot(predicts, label="predict")
# plt.plot(tpa_train.diff().dropna(), label="labeled_diff",color="yellow")
# plt.plot(restore_diff(tpa_train, results.fittedvalues),
#          color='red', label="fitted")
# plt.plot(results.fittedvalues, color='green', label="fitted")
# plt.plot(tpa_train, color='green', label="fitted")
# # plt.plot(restore_diff(tpa_train,tpa_train.diff().dropna()), color='blue', label="fitted")
# plt.legend()
# plt.show()
# print(metrics.mean_squared_error(predicts, tpa_test))
# print(np.sqrt(metrics.mean_squared_error(predicts, tpa_test)))
# plt.figure(figsize=(16, 7))
# plt.plot(tpa_train, label="labeled")
# # plt.plot(tpa_train.diff().dropna(), label="labeled_diff",color="yellow")
# plt.plot(restore_diff(tpa_train, results.fittedvalues),
#          color='red', label="fitted")
# # plt.plot(results.fittedvalues, color='green', label="fitted")
# # plt.plot(tpa_train, color='green', label="fitted")
# # plt.plot(restore_diff(tpa_train,tpa_train.diff().dropna()), color='blue', label="fitted")
# plt.legend()
# plt.show()

# series = get_labeled_value()

# model = ARIMA(data["total_purchase_amt"]['2014-05':'2014-07'], order=(1, 0, 1))
# results_ARIMA = model.fit(disp=-1)


# paramaters_auto_optimization(series)
