import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import random
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

df = pd.read_csv("Seattle_Police_Department_911_Incident_Response.csv")

df["Event Clearance Date"] = pd.to_datetime(df["Event Clearance Date"], format="%m/%d/%Y %I:%M:%S %p")

def get_hour(ts):
    return ts.hour

# encode class values as integers
def one_hot_encoding(df):
    encoder = LabelEncoder()
    encoder.fit(df)
    encoded_df = encoder.transform(df)
    # convert integers to dummy variables (i.e. one hot encoded)
    return np_utils.to_categorical(encoded_df)


df = df[pd.notnull(df["Event Clearance Group"])]
df = df[pd.notnull(df["Zone/Beat"])]
df["hour"] = df["Event Clearance Date"].apply(get_hour)

zone_beat = one_hot_encoding(df["Zone/Beat"])
event_clearance_group = one_hot_encoding(df["Event Clearance Group"])

# create model
basic_model = Sequential()
basic_model.add(Dense(100, input_shape=(90,), activation='relu', name="dense_1"))
basic_model.add(Dense(44, input_shape=(100,), activation="sigmoid", name="dense_4"))
basic_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
basic_model.fit(zone_beat, event_clearance_group, epochs=1, verbose=1)
    
