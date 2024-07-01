import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def OutputCSV(SAMPLE_List):
    name=["Exact prices","Predict prices"]
    df_SAMPLE = pd.DataFrame(columns=name,data=SAMPLE_List)
    df_SAMPLE.to_csv('1310.csv', index=False,encoding="utf-8")