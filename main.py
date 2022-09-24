from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np


app = FastAPI()

@app.get("/suggest")
async def root_handler():

    i_data = pd.read_csv("data.csv",sep=',')
    data = i_data[["Activity Type","Time Spent (hours)","Location Category","Location Name"]]
    predict = i_data[["Recommend"]]
    
    pickle_in = open("model", "rb")
    linear = pickle.load(pickle_in)
    predictions = linear.predict(pd.get_dummies(data))
    for x in range(len(predictions)):
        print(predict[x])
    
    return {
        "reply":"none"
    }




