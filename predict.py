import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder as lb
from sklearn.preprocessing import MinMaxScaler


# Load the pickled model
model = pickle.load(open("car_price_prediction_model1.pkl", 'rb'))


def main():
    st.header('Daddy Car predictor Services')
    st.title("Reliable model for your car prediction services")
    
    name_of_cars = st.text_input('Name of Car')
    year = st.number_input("Year",min_value=2000, max_value=2023,value=2000)
    miles = st.number_input("Miles",min_value=100, max_value=5000000,value=100)
    data = pd.DataFrame({"Name_of_Cars":[name_of_cars], "Year":[year], "Miles":[miles]})
    data2 = pd.read_csv("new_cars.csv")
    main_data = pd.concat([data, data2])
    le = lb()
    main_data["Name_of_Cars"] = le.fit_transform(main_data["Name_of_Cars"])
    scaler = MinMaxScaler()
    scaler.fit(main_data[["Year"]])
    main_data["Year"] = scaler.transform(main_data[["Year"]])
    scaler.fit(main_data[["Miles"]])
    main_data["Miles"] = scaler.fit_transform(main_data[["Miles"]])
    test_data = main_data.iloc[0, 0:3].values
    test_data = np.expand_dims(test_data, axis=0)

    if st.button('Predict'):
        makeprediction = model.predict(test_data)
        print(type(makeprediction))
        output = round(makeprediction[0], 2)
        st.success("Predicted Price of Car is: {}$dollars".format(output))

print(__name__)
if __name__ == '__main__':
    main()
