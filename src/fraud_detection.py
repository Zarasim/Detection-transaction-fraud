import os
import logging
import json
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline


# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Create output directory if it doesn't exist   
os.makedirs(os.path.dirname(config['output_paths']['fraud_detection_results_file']), exist_ok=True)

# Configure the logger
logger = logging.getLogger(__name__)


def load_preprocessed_data_file(preprocessed_data_file: str) -> pd.DataFrame:
    """
    Load preprocessed fraud dataframe from CSV file.
    
    Args:
        preprocessed_data_file (str): Path to the preprocessed data CSV file.
        
    Returns:
        DataFrame: preprocessed fraud dataframe.
    """
    try:
        df_preprocessed = pd.read_csv(preprocessed_data_file)
    except FileNotFoundError as e:
        raise Exception('Cannot read preprocessed data file') from e
    
    return df_preprocessed


def anomaly_detection_isolation_forest(data: pd.DataFrame, contamination: float = 0.1, random_state: int = 42) -> pd.Series:
    """
    Perform anomaly detection using Isolation Forest.
    
    Args:
        data (DataFrame): Input data for anomaly detection.
        contamination (float): Proportion of outliers in the data (default: 0.1).
        random_state (int): Random seed for reproducibility (default: 42).
        
    Returns:
        Series: Anomaly scores for each data point.
    """

    iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
    iso_forest.fit(data)
    anomaly_scores = iso_forest.decision_function(data)
    
    return anomaly_scores


def select_top_outliers(data: pd.DataFrame, anomaly_scores: pd.Series, n: int = 400) -> pd.DataFrame:
    """
    Select the top n outliers based on anomaly scores.
    
    Args:
        data (DataFrame): Input data.
        anomaly_scores (Series): Anomaly scores for each data point.
        n (int): Number of top outliers to select (default: 400).
        
    Returns:
        DataFrame: Top n outliers.
    """

    outliers = pd.DataFrame(data={'anomaly_score': anomaly_scores}, index=data.index)
    outliers = outliers.sort_values('anomaly_score', ascending=True)
    
    return outliers.head(n)


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, prev_model: XGBClassifier = None, random_state: int = 42) -> XGBClassifier:
    """
    Train an XGBoost model using stratified k-fold cross-validation and transfer learning.
    
    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        prev_model (XGBClassifier): Previous XGBoost model for transfer learning (default: None).
        random_state (int): Random seed for reproducibility (default: 42).
        
    Returns:
        Best XGBoost model
    """
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    best_precision = 0
    best_model = None

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # create pipeline with over and under sampling
        over_sampler = ADASYN(random_state=random_state)
        under_sampler = NearMiss(version=1)
        steps = [('over', over_sampler), ('under', under_sampler), ('model', XGBClassifier(random_state=random_state))]
        pipeline = Pipeline(steps=steps)

        if prev_model is not None:
            pipeline.fit(X_train_fold, y_train_fold,model__xgb_model=prev_model)
        else:
            pipeline.fit(X_train_fold, y_train_fold)

        y_pred_fold = pipeline.predict(X_val_fold)
        precision = precision_score(y_val_fold, y_pred_fold)

        if precision > best_precision:
            best_precision = precision
            best_model = pipeline.named_steps['model']

    return best_model


def main():
    """
    Main function to execute fraud detection on a monthly basis.

    The goal of this fraud detection system is to identify and flag the 400 most likely fraudulent transactions each month for review by the fraud analyst team. 

    The approach combines anomaly detection using Isolation Forest and supervised learning using XGBoost, with a focus on adapting to evolving fraud patterns over time.

    More information on README.md
    """

    # Load preprocessed data
    preprocessed_data_file = config['output_paths']['preprocessed_data_file']
    data = load_preprocessed_data_file(preprocessed_data_file)

    # Load contamination and random state
    contamination = config['model_settings']['contamination']
    random_state = config['model_settings']['random_state']

    results = []
    total_transaction_amount_saved = 0
    total_potential_amount_saved = 0
    prev_model = None


    for month in range(1, 13):
        logger.info(f'Processing month {month}...')
        current_month_data = data[data['month'] == month]
        X = current_month_data.drop(['Fraud','transactionAmount_original'], axis=1)
        y = current_month_data['Fraud']

        if month == 1:
            # Use Isolation Forest for anomaly detection in January
            logger.info('Using Isolation Forest for anomaly detection')
            anomaly_scores = anomaly_detection_isolation_forest(X, contamination)
            top_outliers = select_top_outliers(X, anomaly_scores, n=400)
        else:
            # Use the supervised model trained on the previous month's data with transfer learning
            logger.info('Using supervised model with transfer learning')
            prev_month_data = data[data['month'] == month - 1]

            # Drop features that cannot be used for making prediction in the current month
            # Read Limitations section in the README for more information about reported_delay
            X_train = prev_month_data.drop(['Fraud','transactionAmount_original'], axis=1)
            y_train = prev_month_data['Fraud']
            supervised_model = train_xgboost_model(X_train, y_train, prev_model)
            prev_model = supervised_model

            # Combine the predictions from the supervised model and Isolation Forest
            supervised_predictions = supervised_model.predict_proba(X)[:, 1]
            isolation_forest_scores = anomaly_detection_isolation_forest(X, contamination,random_state=random_state)
            combined_scores = supervised_predictions * isolation_forest_scores

            top_outliers = select_top_outliers(X, combined_scores, n=400)

        y_true = y.loc[top_outliers.index]
        y_pred = pd.Series(1, index=top_outliers.index)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        num_true_fraud = y_true.sum()
        num_true_fraud_current_month = y.sum()

        transaction_amount_saved = current_month_data.loc[y_true[y_true == 1].index, 'transactionAmount_original'].sum()
        total_transaction_amount_saved += transaction_amount_saved
        potential_amount_saved = current_month_data.loc[y[y == 1].index, 'transactionAmount_original'].sum()
        total_potential_amount_saved += potential_amount_saved
        
        results.append({
            'Month': month,
            'Precision': precision,
            'F1 Score': f1,
            'Number of True Frauds in the 400 samples': num_true_fraud,
            'Number of True Frauds in the current month': num_true_fraud_current_month,
            'Transaction Amount Saved': transaction_amount_saved,
            "Potential Transaction Amount Saved": potential_amount_saved
        })

    results_df = pd.DataFrame(results)
    logger.info('Results:\n%s', results_df)
    logger.info(f"Total Transaction Amount Saved in the Year (£): {total_transaction_amount_saved:.2f}")
    logger.info(f"Potential Transaction Amount Saved in the Year (£): {total_potential_amount_saved:.2f}")
    
    # Save results
    output_file = config['output_paths']['fraud_detection_results_file']
    results_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()