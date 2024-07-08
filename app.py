import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import json
import boto3
from botocore.exceptions import NoCredentialsError

# Define the function to extract data from MySQL
def extract_data_mysql():
    url = "http://18.188.180.174:3000/data"
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data)

# Define the function to extract data from MongoDB
def extract_data_mongo():
    url = "http://18.188.180.174:3001/data"
    response = requests.get(url)
    other_data = response.json()
    return pd.DataFrame(other_data)

# Define the function to extract data from S3
def extract_data_from_s3():
    bucket = 'jack-bucket-aws'
    s3_file = 'datos_10_tiendas.json'
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=bucket, Key=s3_file)
        data = response['Body'].read().decode('utf-8')
        data = json.loads(data)
        return pd.DataFrame(data)
    except NoCredentialsError:
        st.error("Credentials not available")
        return None

# Define the function to clean data
def clean_data(df):
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df = df.dropna()
    return df

# Define the function to combine data
def combine_data(mysql_df, mongo_df, s3_df):
    combined_df = mysql_df.merge(mongo_df, on='product_id').merge(s3_df, on='store_id')
    return combined_df

# Define the function to preprocess data
def preprocess_data(df):
    df['transaction_date'] = df['transaction_date'].map(pd.Timestamp.toordinal)
    X = df[['store_id', 'product_id', 'transaction_date']]
    y = df['sales_amount']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32)

# Define the function to train the model
def train_model(X_train, y_train, X_test, y_test):
    class SalesPredictor(nn.Module):
        def __init__(self):
            super(SalesPredictor, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SalesPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            st.write(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze()
        test_loss = criterion(predictions, y_test)
        st.write(f'Test Loss: {test_loss.item():.4f}')
    
    return model

# Streamlit App
st.title('Data Pipeline and Sales Prediction App')

# Extract Data
st.header('Extract Data')
mysql_data = extract_data_mysql()
mongo_data = extract_data_mongo()
s3_data = extract_data_from_s3()

st.write('MySQL Data:', mysql_data)
st.write('MongoDB Data:', mongo_data)
st.write('S3 Data:', s3_data)

# Clean Data
st.header('Clean Data')
clean_mysql_data = clean_data(mysql_data)
#clean_mongo_data = clean_data(mongo_data)
#clean_s3_data = clean_data(s3_data)

st.write('Clean MySQL Data:', clean_mysql_data)
#st.write('Clean MongoDB Data:', clean_mongo_data)
#st.write('Clean S3 Data:', clean_s3_data)

# Combine Data
st.header('Combine Data')
combined_data = combine_data(clean_mysql_data, mongo_data, s3_data)
st.write('Combined Data:', combined_data)

# Preprocess Data
st.header('Preprocess Data')
X_train, X_test, y_train, y_test = preprocess_data(combined_data)

# Train Model
st.header('Train Model')
model = train_model(X_train, y_train, X_test, y_test)

# Save DataFrames to CSV
st.header('Save DataFrames to CSV')
mysql_data.to_csv("mysql_data.csv", index=False)
mongo_data.to_csv("mongo_data.csv", index=False)
s3_data.to_csv("s3_data.csv", index=False)
combined_data.to_csv("combined_data.csv", index=False)

st.write("DataFrames saved to CSV files.")
