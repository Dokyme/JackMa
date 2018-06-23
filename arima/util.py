import pandas as pd

LOOK_BACK = 1
BATCH_START = 0
TIME_STEPS = 7
BATCH_SIZE = 1
INPUT_SIZE = 11
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.0005
ITR_STEP = 1000

PURCHASE_SCALE = 0.1
PURCHASE_BIAS = 0.1


def get_time_features(tp="train"):
    time_features = pd.read_csv("data/all_time_features.csv",
                                parse_dates=["report_date"],
                                date_parser=lambda date: pd.datetime.strptime(date, "%Y-%m-%d"))
    time_features = time_features.set_index("report_date")
    if tp is "train":
        t = time_features["2014-04":"2014-07"].values
    elif tp is "test":
        t = time_features["2014-08":"2014-08"].values
    elif tp is "predict":
        t = time_features["2014-09":].values
    rows, cols = t.shape
    t = t.reshape(rows, cols, 1)
    return t


def get_labeled_value(tp="train", field="total_purchase_amt"):
    global PURCHASE_SCALE, PURCHASE_BIAS

    data = pd.read_csv("data/stable.csv",
                       parse_dates=["report_date"],
                       date_parser=lambda date: pd.datetime.strptime(date, "%Y-%m-%d"))
    data = data.set_index("report_date")
    tpa = data[field]
    PURCHASE_SCALE = tpa.max()-tpa.min()
    PURCHASE_BIAS = tpa.min()
    # 归一化
    utpa = (tpa-PURCHASE_BIAS)/PURCHASE_SCALE
    if tp is "train":
        t = utpa["2014-04":"2014-07"]
    elif tp is "test":
        t = utpa["2014-08":"2014-08"]
    elif tp is "all":
        t = utpa["2014-04":]
    return t


def restore_scale(series):
    return series*PURCHASE_SCALE+PURCHASE_BIAS
