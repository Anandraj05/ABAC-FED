import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import logging

def validate_data(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    non_numeric_cols = data.columns.difference(numeric_cols)

    if data[numeric_cols].isnull().any().any():
        numeric_imputer = SimpleImputer(strategy='mean')
        data[numeric_cols] = pd.DataFrame(numeric_imputer.fit_transform(data[numeric_cols]), columns=numeric_cols)
        logging.warning(f"Missing values in numeric columns imputed with mean.")

    if data[non_numeric_cols].isnull().any().any():
        non_numeric_imputer = SimpleImputer(strategy='most_frequent')  # Using mode (most frequent)
        data[non_numeric_cols] = pd.DataFrame(non_numeric_imputer.fit_transform(data[non_numeric_cols]), columns=non_numeric_cols)
        logging.warning(f"Missing values in non-numeric columns imputed with most frequent value.")

    for column in data.columns:
        if data[column].dtype == object:
            try:
                data[column] = pd.to_datetime(data[column], errors='coerce')
            except ValueError:
                try:
                    data[column] = pd.to_numeric(data[column], errors='coerce')
                except ValueError:
                    pass

    return data

def preprocess_data(data):

    if "User ID" in data.columns:
        data["User ID"] = data["User ID"].astype(str)
    if "Resource ID" in data.columns:
        data["Resource ID"] = data["Resource ID"].astype(str)
    if "Behavioral Score" in data.columns:
        data["Behavioral Score"] = data["Behavioral Score"].astype(int)
    if "Access Granted" in data.columns:
        data["Access Granted"] = data["Access Granted"].map({"TRUE": True, "FALSE": False})


    if data.empty:
        logging.warning("Empty data detected. Please check the input data.")
        return None

    expected_dtypes = ['object', 'int64', 'bool', 'datetime64[ns]', 'float64']
    if not all(dtype.name in expected_dtypes for dtype in data.dtypes):
        unexpected_dtypes = set(data.dtypes) - set(expected_dtypes)
        logging.warning(f"Unexpected data types found in the dataset: {unexpected_dtypes}. Please check the input data.")

    timestamp_cols = ['Timestamp', 'Access Time', 'Request Time']  # Specify the names of timestamp columns
    for col in timestamp_cols:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')

    return data