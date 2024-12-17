import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, classification_report
import streamlit as st

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Step 1: Exploring skewness and outliers
def explore_data(df):
    skewness = df.skew()
    outlier_info = {}
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
        outlier_info[column] = outliers.shape[0]
    return skewness, outlier_info

# Step 2: Data Cleaning and Preprocessing
def clean_and_preprocess(df):
    # Remove rows with STATUS values other than 'WON' or 'LOST'
    df = df[df['STATUS'].isin(['WON', 'LOST'])]
    # Encode STATUS as binary
    df['STATUS'] = df['STATUS'].map({'WON': 1, 'LOST': 0})
    
    # Handle skewness
    transformer = PowerTransformer()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = transformer.fit_transform(df[numerical_columns])

    # Standardize data
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df

# Step 3: Regression Model
def train_regression_model(df):
    X = df.drop(['Selling_Price', 'STATUS'], axis=1)
    y = df['Selling_Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

# Step 4: Classification Model
def train_classification_model(df):
    X = df.drop(['Selling_Price', 'STATUS'], axis=1)
    y = df['STATUS']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return model, report

# Step 5: Streamlit App
def streamlit_app():
    st.title("Copper Industry Modeling")
    
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        skewness, outliers = explore_data(df)
        st.write("Skewness:")
        st.write(skewness)
        st.write("Outliers:")
        st.write(outliers)
        
        df = clean_and_preprocess(df)
        st.write("Preprocessed Data Preview:")
        st.dataframe(df.head())

        reg_model, reg_mse = train_regression_model(df)
        st.write("Regression Model Mean Squared Error:", reg_mse)

        clf_model, clf_report = train_classification_model(df)
        st.write("Classification Model Report:")
        st.text(clf_report)

        st.sidebar.title("Prediction Inputs")
        user_input = {}
        for col in df.drop(['Selling_Price', 'STATUS'], axis=1).columns:
            user_input[col] = st.sidebar.number_input(f"{col}")
        
        user_input_df = pd.DataFrame([user_input])
        
        if st.sidebar.button("Predict Selling Price"):
            predicted_price = reg_model.predict(user_input_df)[0]
            st.write("Predicted Selling Price:", predicted_price)
        
        if st.sidebar.button("Predict Status"):
            predicted_status = clf_model.predict(user_input_df)[0]
            st.write("Predicted Status:", "WON" if predicted_status == 1 else "LOST")

if __name__ == "__main__":
    streamlit_app()
