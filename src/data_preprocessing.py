import os
import logging
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Tuple, List

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Create output directory if it doesn't exist   
os.makedirs(os.path.dirname(config['output_paths']['preprocessed_data_file']), exist_ok=True)

# Configure the logger
logger = logging.getLogger(__name__)

def load_data(transactions_file: str, labels_file: str)->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load transaction and label data from CSV files.
    
    Args:
        transactions_file (str): Path to the transaction data CSV file.
        labels_file (str): Path to the label data CSV file.
        
    Returns:
        tuple: A tuple containing the loaded transaction and label DataFrames.
    """
    df_transactions = pd.read_csv(transactions_file)
    df_labels = pd.read_csv(labels_file)
    
    return df_transactions, df_labels


def preprocess_datetime(df_transactions:pd.DataFrame, df_labels:pd.DataFrame)->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess datetime columns in transaction and label DataFrames.
    
    Args:
        df_transactions (DataFrame): Transaction data DataFrame.
        df_labels (DataFrame): Label data DataFrame.
        
    Returns:
        tuple: A tuple containing the preprocessed transaction and label DataFrames.
    """
    df_transactions["transactionTime"] = pd.to_datetime(df_transactions["transactionTime"]).dt.tz_localize(None)
    df_labels["reportedTime"] = pd.to_datetime(df_labels["reportedTime"]).dt.tz_localize(None)
    
    return df_transactions, df_labels


def create_frequency_features(df_transactions:pd.DataFrame)->pd.DataFrame:
    """
    Create frequency-based features from the transaction DataFrame.
    
    Args:
        df_transactions (DataFrame): Transaction data DataFrame.
        
    Returns:
        DataFrame: Transaction DataFrame with frequency-based features added.
    """
    frequency_columns = ['accountNumber', 'merchantId', 'mcc']
    for column in frequency_columns:
        frequency_series = df_transactions[column].value_counts()
        df_transactions[f'{column}_freq'] = df_transactions[column].map(frequency_series)
    
    return df_transactions


def extract_datetime_features(df_transactions:pd.DataFrame)->pd.DataFrame:
    """
    Extract datetime-based features from the transaction DataFrame.
    
    Args:
        df_transactions (DataFrame): Transaction data DataFrame.
        
    Returns:
        DataFrame: Transaction DataFrame with datetime-based features added.
    """
    df_transactions['hour'] = df_transactions['transactionTime'].dt.hour
    df_transactions['day'] = df_transactions['transactionTime'].dt.day
    df_transactions['month'] = df_transactions['transactionTime'].dt.month
    df_transactions['weekday'] = df_transactions['transactionTime'].dt.weekday
    df_transactions['is_weekend'] = df_transactions['transactionTime'].dt.weekday.isin([5, 6]).astype(int)
    
    return df_transactions


def categorize_hours(df_transactions:pd.DataFrame)->pd.DataFrame:
    """
    Categorize hours into morning, afternoon, evening, and night.
    
    Args:
        df_transactions (DataFrame): Transaction data DataFrame.
        
    Returns:
        DataFrame: Transaction DataFrame with hour categorization features added.
    """
    df_transactions['hour_morning'] = np.where((df_transactions['hour'] >= 6) & (df_transactions['hour'] < 12), 1, 0)
    df_transactions['hour_afternoon'] = np.where((df_transactions['hour'] >= 12) & (df_transactions['hour'] < 18), 1, 0)
    df_transactions['hour_evening'] = np.where((df_transactions['hour'] >= 18) & (df_transactions['hour'] <= 23), 1, 0)
    df_transactions['hour_night'] = np.where((df_transactions['hour'] >= 0) & (df_transactions['hour'] < 6), 1, 0)
    
    return df_transactions


def one_hot_encode_features(df_transactions:pd.DataFrame)->pd.DataFrame:
    """
    One-hot encode weekday and posEntryMode features.
    
    Args:
        df_transactions (DataFrame): Transaction data DataFrame.
        
    Returns:
        DataFrame: Transaction DataFrame with one-hot encoded features.
    """
    df_transactions = pd.concat([df_transactions, pd.get_dummies(df_transactions['weekday'], prefix='weekday', prefix_sep='_')], axis=1)
    df_transactions = pd.concat([df_transactions, pd.get_dummies(df_transactions['posEntryMode'], prefix='pos', prefix_sep='_')], axis=1)
    
    return df_transactions


def compute_cash_pct_spent(df_transactions:pd.DataFrame)->pd.DataFrame:
    """
    Compute the percentage of available cash spent in each transaction.
    
    Args:
        df_transactions (DataFrame): Transaction data DataFrame.
        
    Returns:
        DataFrame: Transaction DataFrame with cash percentage spent feature added.
    """
    df_transactions['cash_pct_spent'] = df_transactions['transactionAmount'] / df_transactions['availableCash']
    
    return df_transactions


def merge_transactions_labels(df_transactions:pd.DataFrame, df_labels:pd.DataFrame)->pd.DataFrame:
    """
    Merge transaction and label DataFrames.
    
    Args:
        df_transactions (DataFrame): Transaction data DataFrame.
        df_labels (DataFrame): Label data DataFrame.
        
    Returns:
        DataFrame: Merged DataFrame with fraud labels.
    """
    df_merged = pd.merge(left=df_transactions, right=df_labels, how="left", on="eventId")
    df_merged['reported_delay'] = (df_merged['reportedTime'] - df_merged['transactionTime']).dt.total_seconds() / 3600  # Convert to hours
    df_merged["Fraud"] = df_merged["reportedTime"].notnull().astype(int)

    return df_merged


def compute_transaction_velocity(df:pd.DataFrame, time_window:str ='24H')->pd.DataFrame:
    """
    Compute transaction velocity for each account within a specified time window.
    
    Args:
        df (DataFrame): Transaction data DataFrame.
        time_window (str): Time window for calculating transaction velocity (default: '24H').
        
    Returns:
        DataFrame: Transaction DataFrame with transaction velocity feature added.
    """
    transaction_velocity = df.groupby(['accountNumber', pd.Grouper(key='transactionTime', freq=time_window)]).size().reset_index()
    transaction_velocity.columns = ['accountNumber', 'transactionTime', 'transaction_velocity']
    df = pd.merge(df, transaction_velocity, on=['accountNumber', 'transactionTime'], how='left')
    
    return df


def compute_merchant_category_frequency(df:pd.DataFrame)->pd.DataFrame:
    """
    Compute merchant category frequency for each account.
    
    Args:
        df (DataFrame): Transaction data DataFrame.
        
    Returns:
        DataFrame: Transaction DataFrame with merchant category frequency feature added.
    """
    merchant_category_frequency = df.groupby(['accountNumber', 'mcc']).size().reset_index()
    merchant_category_frequency.columns = ['accountNumber', 'mcc', 'merchant_category_frequency']
    df = pd.merge(df, merchant_category_frequency, on=['accountNumber', 'mcc'], how='left')
    
    return df


def compute_time_since_last_transaction(df:pd.DataFrame)->pd.DataFrame:
    """
    Compute the time elapsed since the last transaction for each account.
    
    Args:
        df (DataFrame): Transaction data DataFrame.
        
    Returns:
        DataFrame: Transaction DataFrame with time since last transaction feature added.
    """
    df = df.sort_values(['accountNumber', 'transactionTime'])
    df['time_since_last_transaction'] = df.groupby('accountNumber')['transactionTime'].diff().dt.total_seconds()
   
    return df


def compute_average_transaction_amount_by_country_and_hour(df:pd.DataFrame)->pd.DataFrame:
    """
    Compute average transaction by country and hour.
    
    Args:
        df (DataFrame): Transaction data DataFrame.
        
    Returns:
        DataFrame: Transaction DataFrame with average transaction by country and hour added.
    """
    avg_amount_by_country = df.groupby('merchantCountry')['transactionAmount'].mean()
    df['avg_transaction_amount_by_country'] = df['merchantCountry'].map(avg_amount_by_country)
    
    avg_amount_by_hour = df.groupby('hour')['transactionAmount'].mean()
    df['avg_transaction_amount_by_hour'] = df['hour'].map(avg_amount_by_hour)
    
    return df


def scale_numeric_features(df:pd.DataFrame, numeric_columns:List[str])->pd.DataFrame:
    """
    Scale numeric features using RobustScaler.
    
    Args:
        df (DataFrame): Transaction data DataFrame.
        numeric_columns (list): List of numeric columns to scale.
        
    Returns:
        DataFrame: Transaction DataFrame with scaled numeric features.
    """
    scaler = RobustScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
  
    return df


def drop_unnecessary_columns(df_preprocessed:pd.DataFrame, columns_to_drop:List[str])->pd.DataFrame:
    """
    Drop unnecessary columns for fraud detection from the preprocessed DataFrame.
     
    Args:
        df_preprocessed (DataFrame): Preprocessed transaction data DataFrame.
        columns_to_drop (list): List of columns to drop in the preprocessed DataFrame.
        
    Returns:
        DataFrame: Preprocessed DataFrame with unnecessary columns dropped.
    """               
    
    df_preprocessed = df_preprocessed.drop(columns_to_drop, axis=1)
    
    return df_preprocessed


def check_missing_values(df_preprocessed):
    """
    Check for missing values in the preprocessed DataFrame.
    
    Args:
        df_preprocessed (DataFrame): Preprocessed transaction data DataFrame.
        
    Returns:
        DataFrame: DataFrame containing the count and percentage of missing values for each column.
    """
    total = df_preprocessed.isnull().sum().sort_values(ascending=False)
    percent = (df_preprocessed.isnull().sum() / df_preprocessed.isnull().count() * 100).sort_values(ascending=False)
    missing_values_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    return missing_values_df


def fill_missing_values(df_preprocessed:pd.DataFrame)->pd.DataFrame:
    """
    Fill missing values in the preprocessed DataFrame.
    
    Args:
        df_preprocessed (DataFrame): Preprocessed transaction data DataFrame.
        
    Returns:
        DataFrame: Preprocessed DataFrame with missing values filled.
    """
    # Fill missing values with appropriate strategies based on the feature
    df_preprocessed['transaction_velocity'].fillna(0, inplace=True)
    df_preprocessed['merchant_category_frequency'].fillna(0, inplace=True)
    df_preprocessed['time_since_last_transaction'].fillna(0, inplace=True)
    df_preprocessed['reported_delay'].fillna(0, inplace=True)  # Fill NaN values with 0 for non-fraudulent transactions
    
    return df_preprocessed


def preprocess_data(df_transactions:pd.DataFrame, df_labels:pd.DataFrame, numeric_columns_to_scale:List[str])->pd.DataFrame:
    """
    Preprocess and engineer features from transaction and label data.
    
    Args:
        df_transactions (DataFrame): Transaction data DataFrame.
        df_labels (DataFrame): Label data DataFrame.
        numeric_columns_to_scale (list): List of numeric columns to scale.
        
    Returns:
        DataFrame: Preprocessed DataFrame with engineered features.
    """
    
    # Preprocess datetime columns
    df_transactions, df_labels = preprocess_datetime(df_transactions, df_labels)

    logger.info('Engineering features...')
    # Create frequency-based features
    df_transactions = create_frequency_features(df_transactions)

    # Extract datetime-based features
    df_transactions = extract_datetime_features(df_transactions)

    # Categorize hours
    df_transactions = categorize_hours(df_transactions)

    # One-hot encode features
    df_transactions = one_hot_encode_features(df_transactions)

    # Compute cash percentage spent
    df_transactions = compute_cash_pct_spent(df_transactions)

    # Compute transaction velocity
    df_transactions = compute_transaction_velocity(df_transactions, time_window='24H')

    # Compute merchant category frequency
    df_transactions = compute_merchant_category_frequency(df_transactions)

    # Compute time since last transaction
    df_transactions = compute_time_since_last_transaction(df_transactions)

    logger.info('Merging transaction and label data...')
    # Merge transaction and label data
    df_merged = merge_transactions_labels(df_transactions, df_labels)

    # Keep track of original transaction amount for the 
    df_merged['transactionAmount_original'] = df_merged['transactionAmount']
    
    logger.info('Scaling numeric features...')
    # Scale numeric features
    df_preprocessed = scale_numeric_features(df_merged, numeric_columns_to_scale)

    logger.info('Computing additional features...')
    # Compute average transaction amount by country and hour
    df_preprocessed = compute_average_transaction_amount_by_country_and_hour(df_preprocessed)
    
    return df_preprocessed


def main():
    """
    Main function to execute the preprocessing and feature engineering steps.

    Data Preprocessing and Feature Engineering:

    This script performs data preprocessing and feature engineering on the raw transaction dataset. 
    The engineered features aim to capture patterns and characteristics that may be indicative of fraudulent transactions.

    Assumption for Online Learning:

    While this script applies the preprocessing and feature engineering steps to all available data, in a real-world production environment, 
    this procedure would likely be performed on a daily basis as new transaction data becomes available. 

    The preprocessed dataset is saved as 'df_preprocessed.csv' for further use in fraud detection modeling.
    """
    
    # Load data
    transactions_file = config['data_paths']['transactions_file']
    labels_file = config['data_paths']['labels_file']
    df_transactions, df_labels = load_data(transactions_file, labels_file)
   
    # Preprocess data
    numerics_colums_to_scale = config['model_settings']['numeric_columns_to_scale']
    df_preprocessed = preprocess_data(df_transactions, df_labels,numerics_colums_to_scale)

    # Check missing values
    df_missing_values = check_missing_values(df_transactions)
    logger.info("Missing Values:\n%s", df_missing_values)

    # Fill missing values
    logger.info("Fill Missing Values...")
    df_preprocessed = fill_missing_values(df_preprocessed)

    # Drop unnecessary columns used in by the classifier
    columns_to_drop = config['model_settings']['columns_to_drop']
    df_preprocessed = drop_unnecessary_columns(df_preprocessed,columns_to_drop)
    
    # Calculate fraud percentage
    fraud_percentage = df_preprocessed["Fraud"].sum() / df_preprocessed.shape[0] * 100
    logger.info(f"Fraud Percentage: {fraud_percentage:.2f}%")
    
    logger.info("Data Types:\n%s", df_preprocessed.dtypes)
    logger.info("Total Missing Values: %d", df_preprocessed.isnull().sum().sum())
    logger.info("Total number of columns: %d", len(df_preprocessed.columns))

    # Save resultss
    output_file = config['output_paths']['preprocessed_data_file']
    df_preprocessed.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()