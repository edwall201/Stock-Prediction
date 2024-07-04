import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def OutputCSV(SAMPLE_List):
    name=["Exact prices","Predict prices"]
    df_SAMPLE = pd.DataFrame(columns=name,data=SAMPLE_List)
    df_SAMPLE.to_csv('1310.csv', index=False,encoding="utf-8")

stocktxtpath = '1310.txt'
ds = pd.read_csv(stocktxtpath)
predictdays=60
if (int(len(ds) * 0.20) < (predictdays + 1)):
    addnotuse = open("nouse.csv", "a")
    writer = csv.writer(addnotuse)
    writer.writerow(row1)
    addnotuse.close()
else:
    train_ds, test_ds = ds[0:int(len(ds) * 0.80)], ds[int(len(ds) * 0.80):len(ds)]

    # defining training set taking column 'open'
    train_set = train_ds.loc[:, ["開盤價"]].values


    # applying feature scaling,Normalization is preferred in RNN
    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler(feature_range=(0, 1))
    scaled_train_set = sc.fit_transform(train_set)

    # creating a data structure with 60 timestamps and 1 output
    x_train = []
    y_train = []

    # values can be changed but i got best results with 60 timestamps
    for i in range(predictdays, len(train_set)):
        x_train.append(scaled_train_set[i - predictdays:i, 0])
        y_train.append(scaled_train_set[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)