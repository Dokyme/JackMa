from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import traceback

import requests
import json


def get_holiday_info(date_str):
    resp = requests.get("http://tool.bitefu.net/jiari/?d=" + date_str)
    return int(resp.text)


data = pd.read_csv("data/data.csv", parse_dates=["report_date"],
                   date_parser=lambda date: pd.datetime.strptime(date, "%Y-%m-%d"))
raw = data.set_index("report_date")


# 构造时间特征
def get_time_feature():
    # try:
    #     df2 = pd.read_csv("time_features.csv",
    #                       parse_dates=["11"],
    #                       date_parser=lambda date: pd.datetime.strptime(date, "%Y-%m-%d"))
    #     return df2, df2["11"]
    # except Exception:
    #     traceback.print_exc()
    #     pass
    res = []
    indexes = []
    start_time = datetime.datetime.strptime("20130701", "%Y%m%d")
    end_time = datetime.datetime.strptime("20140930", "%Y%m%d")
    delta = datetime.timedelta(days=1)
    ldh = None
    while start_time <= end_time:
        h = get_holiday_info(start_time.strftime("%Y%m%d"))
        print(start_time.strftime("%Y%m%d")+"\t"+str(h))
        if start_time.day == 1:
            first_of_month = 1
        else:
            first_of_month = 0
        if start_time.weekday() == 5 or start_time.weekday() == 6:
            weekend = 1
        else:
            weekend = 0
        if h == 2:
            holiday = 1
        else:
            holiday = 0
        if start_time.day < 5:
            start_of_month = 1
        else:
            start_of_month = 0
        if start_time.day > 25:
            end_of_month = 1
        else:
            end_of_month = 0
        if start_time.day > 10 and start_time.day < 20:
            mid_of_month = 1
        else:
            mid_of_month = 0
        if ldh == 2:
            last_day_holiday = 1
        else:
            last_day_holiday = 0
        if ldh == 1:
            last_day_weekend = 1
        else:
            last_day_weekend = 0
        week_of_year = int(start_time.strftime("%W"))
        first_week_of_year = int(datetime.datetime(
            start_time.year, start_time.month, 1).strftime("%W"))
        week_of_month = week_of_year - first_week_of_year + 1
        res.append([start_time.strftime("%Y-%m-%d"),
                    weekend,
                    holiday,
                    start_of_month,
                    mid_of_month,
                    end_of_month,
                    week_of_month,
                    0,
                    0,
                    0,
                    0,
                    first_of_month])
        indexes.append(pd.Timestamp(start_time.strftime("%Y-%m-%d")))
        ldh = h
        start_time += delta
    for i in range(len(res)):
        if res[i][1] == 1 and res[i-1][1] != 1:
            res[i - 1][9] = 1
        if res[i][2] == 1 and res[i-1][2] != 1:
            res[i - 1][10] = 1
        if res[i][1] == 1 and res[i+1][1] != 1:
            res[i][7] = 1
        if res[i][2] == 1 and res[i+1][2] != 1:
            res[i][8] = 1
    return res, indexes, ["report_date", "isweekend", "isholiday", "isstartofmonth", "ismidofmonth", "isendofmonth", "weekofmonth", "islastdayofweekend", "islastdayofholiday", "yesterdayisweekday", "yesterdayisholiday", "firstdayofmonth"]


time_fdata, indexes, cols = get_time_feature()
time_features = pd.DataFrame(data=time_fdata, columns=cols)
time_features = time_features.set_index("report_date")
time_features.to_csv("all_time_features.csv")
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# print(get_holiday_info("20131001"))
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# data = raw.sort_index()
# target = data["total_purchase_amt"]
# data.drop("total_purchase_amt", axis=1, inplace=True)

# time_features.to_csv("time_features.csv")
# data = time_features
# train_x = data[-106:-31]
# test_x = data[-31:]
# target = (target - target.min()) / (target.max() - target.min())
# train_y = target[-106:-31]
# test_y = target[-31:]

model = LinearRegression()
print("Start to fit")
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
pred_y = pd.Series(pred_y, index=pd.date_range('7/18/2014', periods=31, freq='B'))
plt.plot(test_y, label="labeled")
plt.plot(pred_y, label="predicted")
# plt.plot(test_y, label="labeled")
# plt.plot(pred_y, label="predicted")
plt.legend()
plt.show()
print("MSE:%f" % (metrics.mean_squared_error(test_y, pred_y)))
print("RMSE:%f" % np.sqrt(metrics.mean_squared_error(test_y, pred_y)))
