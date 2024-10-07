import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import codecs
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

st.title("Data Science App")

image_path = Image.open("wine.jpeg")
st.image(image_path)

app_page = st.sidebar.selectbox("Select Page", ['Data Exploration', 'Visualization', 'Prediction'])
df = pd.read_csv("wine_quality_red.csv")

if app_page == 'Data Exploration':
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
    
    if st.button("Generate Report") :

    #function to load HTML file
        def read_html_report(file_path): 
            with codecs.open(file_path, "r", encoding = "utf-8") as f:
                return f.read()
            
        html_report = read_html_report("report.html")

        st.title("Streamlit Quality Report")
        st.components.v1.html(html_report, height = 1000, scrolling = True)
        

if app_page == 'Visualization':
    st.subheader("03 Data Visualization")
    list_columns = df.columns

    values = st.multiselect("Select two variables:", list_columns,['quality','citric acid'])
    st.line_chart(df, x = values[0], y =values[1])

    st.bar_chart(df, x = values[0], y =values[1])

    values_pairplot = st.multiselect("Select 4 variables", list_columns,['quality','citric acid', 'alcohol', 'chlorides'])

    df2 = df[[values_pairplot[0], values_pairplot[1],values_pairplot[2],values_pairplot[3]]]

    st.pyplot(sns.pairplot(df2, diag_kind = 'kde'))

if app_page == 'Prediction':

        st.title("03 Prediction")
        list_columns = df.columns

        input_lr = st.multiselect("Select variables:", list_columns,["quality", "citric acid"])

        df2 = df[input_lr]
        X = df2
        #target variable

        y = df["alcohol"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        lr = LinearRegression()

        lr.fit(X_train, y_train)

        pred = lr.predict(X_test)

        mae = metrics.mean_absolute_error(pred, y_test)

        r2 = metrics.r2_score(pred,y_test)

        st.write("Mean Absolute Error:", mae)
        st.write("R2:", r2)

