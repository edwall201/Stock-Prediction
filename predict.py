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
    train_set = train_ds.loc[:, ["Opening prices"]].values


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

    # here reshape(batch_size,timesteps,input_dim)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # building the RNN
    rnn = keras.models.Sequential()

    # adding first layer of LSTM
    rnn.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    rnn.add(keras.layers.Dropout(0.5))

    # adding second layer of LSTM
    rnn.add(keras.layers.LSTM(units=50, return_sequences=True))
    rnn.add(keras.layers.Dropout(0.5))

    # adding third layer of LSTM
    rnn.add(keras.layers.LSTM(units=50, return_sequences=True))
    rnn.add(keras.layers.Dropout(0.5))

    # adding fourth layer of LSTM
    rnn.add(keras.layers.LSTM(units=50))
    rnn.add(keras.layers.Dropout(0.5))

    # adding output layer of LSTM
    rnn.add(keras.layers.Dense(units=1))

    # compiling the model
    rnn.compile(optimizer='adam', loss='mean_squared_error')

    # rnn.summary()

    # training the model
    rnn.fit(x_train, y_train, epochs=30, batch_size=60)

    rnn.save("1310.h5")

    # defining y_test(actual values)
    y_test = test_ds.iloc[:, 1:2].values

    # inputs contain 60 previous values of the first element of test_ds
    # bcause for prediction of first values of test_ds we need 60 prior values
    ds_total = pd.concat((train_ds["Opening prices"], test_ds["Opening prices"]), axis=0)

    inputs = ds_total[(len(ds_total) - len(test_ds) - predictdays):].values

    inputs = inputs.reshape(-1, 1)

    inputs = sc.transform(inputs)

    # creating a x_test
    x_test = []

    for i in range(predictdays, predictdays + int(len(ds) * 0.2 + 1)):
        x_test.append(inputs[i - predictdays:i, 0])
    x_test = np.array(x_test)
    # reshaping x_test for prediction
    # here reshape(batch_size,timesteps,input_dim)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # print(x_test.shape)
    y_pred = rnn.predict(x_test)
    y_pred = sc.inverse_transform(y_pred)

    y_predList=y_pred.tolist()
    y_testList=y_test.tolist()
    PredictTestList=[]

    for i in range(0, len(y_testList)):
        PredictTestList.append([y_testList[i][0], y_predList[i][0]])
    OutputCSV(PredictTestList)

    plt.figure(figsize=(12.8, 9.6))
    plt.plot(y_test, color='red', label='Real Stock Price')
    plt.plot(y_pred, color='blue', label='Predicted Stock Price')
    plt.title('1310 Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig("1310.png")
    plt.show()
    plt.cla()