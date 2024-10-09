import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv("bengaluru_house_prices.csv")
    return df

# Data Preprocessing
def preprocess_data(df):
    # Drop unnecessary columns
    df2 = df.drop(['area_type','society','balcony','availability'],axis='columns')
    df3 = df2.dropna()

    # Create new 'bhk' column
    df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

    # Convert sqft column to numeric
    def convert_sqft_to_num(x):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return None

    df3['total_sqft'] = df3['total_sqft'].apply(convert_sqft_to_num)
    df4 = df3[df3.total_sqft.notnull()]

    # Add price per sqft column
    df4['price_per_sqft'] = df4['price']*100000/df4['total_sqft']

    # Simplify locations
    df4['location'] = df4['location'].apply(lambda x: x.strip())
    location_stats = df4['location'].value_counts(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df4['location'] = df4['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

    # Remove outliers
    df5 = df4[~(df4.total_sqft/df4.bhk < 300)]
    
    # Remove location-specific price per sqft outliers
    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    df6 = remove_pps_outliers(df5)

    # Remove BHK outliers
    def remove_bhk_outliers(df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk-1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < stats['mean']].index.values)
        return df.drop(exclude_indices, axis='index')

    df7 = remove_bhk_outliers(df6)
    
    df8 = df7[df7.bath < df7.bhk + 2]
    df9 = df8.drop(['size', 'price_per_sqft'], axis='columns')
    
    # One-hot encoding for location
    dummies = pd.get_dummies(df9.location)
    df10 = pd.concat([df9, dummies.drop('other', axis='columns')], axis='columns')
    df11 = df10.drop('location', axis='columns')

    return df11

# Train model
def train_model(df11):
    X = df11.drop(['price'], axis='columns')
    y = df11['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    lr_clf = LinearRegression()
    lr_clf.fit(X_train, y_train)

    return lr_clf, X

# Predict house price
def predict_price(model, X, location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]

# Streamlit UI
st.title("Bengaluru House Price Prediction")

# Load data and preprocess
df = load_data()
df_preprocessed = preprocess_data(df)

# Train model
model, X = train_model(df_preprocessed)

# UI Inputs
st.header("Enter the details below to predict house price")
location = st.selectbox("Select Location", options=X.columns[3:])
sqft = st.number_input("Enter Total Square Feet", min_value=500)
bath = st.number_input("Enter Number of Bathrooms", min_value=1)
bhk = st.number_input("Enter Number of Bedrooms (BHK)", min_value=1)

# Prediction Button
if st.button("Predict Price"):
    price = predict_price(model, X, location, sqft, bath, bhk)
    st.write(f"The predicted price for the house is: ₹{price:,.2f} Lakhs")

# Footer
st.markdown("---")
st.write("This application predicts house prices in Bengaluru based on parameters such as location, area, and the number of bathrooms and bedrooms.")
