import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image

st.title("Data Science App")

image_path = Image.open("wine.jpeg")
st.image(image_path)

df = pd.read_csv("wine_quality_red.csv")

st.dataframe(df.head(5))

st.subheader("01 Description of the dataset")
st.dataframe(df.describe())

st.subheader("02 Missing Values")
dfnull = df.isnull()/len(df)*100
total_missing = dfnull.sum().round(2)
st.write(total_missing)
st.write(dfnull)
if total_missing[0] == 0:
    st.success("Congrats you have no missing values")

st.subheader("03 Data Visualization")
list_columns = df.columns

values = st.multiselect("Select two variables:", list_columns,['quality','citric acid'])
st.line_chart(df, x = values[0], y =values[1])

st.bar_chart(df, x = values[0], y =values[1])

values_pairplot = st.multiselect("Select 4 variables", list_columns,['quality','citric acid', 'alcohol', 'chlorides'])

df2 = df[[values_pairplot[0], values_pairplot[1],values_pairplot[2],values_pairplot[3]]]

st.pyplot(sns.pairplot(df2, diag_kind = 'kde'))