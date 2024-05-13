# Fraud Detection System

This repository contains the code and documentation for a fraud detection system designed to identify and flag the 400 most likely fraudulent transactions each month for review by a bank's fraud analyst team.

## Table of Contents
- [Running the System](#running-the-system)
- [Data Preprocessing](#data-preprocessing)
  - [Engineered Features](#engineered-features)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Fraud Detection Approach](#fraud-detection-approach)
  - [Summary of the Approach](#summary-of-the-approach)
  - [Limitations](#limitations)
  - [Dealing with the reportedTime Feature](#dealing-with-the-reportedtime-feature)
- [Model Performance Evaluation](#model-performance-evaluation)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)

## Running the System

To run the fraud detection system, follow these steps:

1. Install the required dependencies listed in the `requirements.txt` file via pip within a virtual environment:
  ```
   pip install -r requirements.txt
  ```

2. Update the `config.json` file with the appropriate paths and settings for your project.

3. Run the `src/main.py` script to execute the entire fraud detection pipeline. You can provide the `--run_eda` flag to optionally run the exploratory data analysis before fraud detection:

   ```
   python src/main.py --run_eda
   ```

   If you want to run the system without performing EDA before fraud detection, simply omit the flag:

   ```
   python src/main.py
   ```

4. The script will preprocess the data, run fraud detection, perform exploratory data analysis (if specified), and plot the statistics.

5. The log files will be saved in the `logs` directory, and the output files will be generated in the `output` and `plots` directories.

## Data Preprocessing

The `src/data_preprocessing.py` script performs data preprocessing and feature engineering on the raw transaction dataset. The engineered features aim to capture patterns and characteristics that may be indicative of fraudulent transactions.

### Engineered Features

1. `accountNumber_freq`, `merchantId_freq`, and `mcc_freq`: These features represent the frequency of each account number, merchant ID, and merchant category code (mcc) in the dataset.

2. `hour`, `day`, `month`, `weekday`, and `is_weekend`: These temporal features capture the time-related patterns of transactions. Fraudulent activities may exhibit distinct temporal patterns compared to genuine transactions.

3. `hour_morning`, `hour_afternoon`, `hour_evening`, and `hour_night`: These binary features categorize transactions based on the time of day. Fraudulent transactions may be more prevalent during certain periods of the day.

4. One-hot encoded `weekday` and `posEntryMode`: These features provide a more granular representation of the day of the week and the point-of-sale entry mode. Different types of fraudulent transactions may be associated with specific days or entry modes.

5. `cash_pct_spent`: This feature calculates the percentage of available cash spent in each transaction. High percentages may indicate unusual spending behavior or potential fraud.

6. `transaction_velocity`: This feature represents the number of transactions made by each account within a specific time window (e.g., 24 hours). High transaction velocities may be indicative of fraudulent activity.

7. `merchant_category_frequency`: This feature captures the frequency of transactions for each account and merchant category combination. Unusual combinations or high frequencies may suggest potential fraud.

8. `time_since_last_transaction`: This feature calculates the time elapsed since the last transaction for each account. Fraudulent transactions may occur in rapid succession or after long periods of inactivity.

9. `avg_transaction_amount_by_country` and `avg_transaction_amount_by_hour`: These features calculate the average transaction amount for each merchant country and hour of the day. Deviations from the average amounts may indicate potential fraud.

These engineered features serve as proxies for fraudulent transactions by capturing patterns and anomalies in account behavior, transaction timings, merchant categories, and spending habits. By incorporating these features, the fraud detection model can learn to identify suspicious transactions more effectively.

The script also performs outlier handling using RobustScaler to mitigate the impact of extreme values on the model's performance.

The preprocessed dataset is saved as `df_preprocessed.csv` for further use in fraud detection modeling and exploratory data analysis.

## Exploratory Data Analysis

The `src/data_preprocessing_EDA.py` script performs exploratory data analysis (EDA) on the preprocessed transaction dataset. It generates various plots and visualizations to gain insights into the data and identify patterns or anomalies that may be relevant for fraud detection.

The script creates the following plots:

1. Histogram of the `reported_delay` feature, which represents the time difference between the reported time of a fraud and the time the fraudulent transaction occurred.

2. Correlation matrix plot to visualize the correlations between different features in the dataset.

3. Count plot of the `Fraud` feature to compare the distribution of fraudulent and non-fraudulent transactions.

4. Box plots of quantitative features (`transactionAmount`, `availableCash`, `reported_delay`, `time_since_last_transaction`) to compare their distributions for fraudulent and non-fraudulent transactions.

These plots provide valuable insights into the relationships between features, the distribution of fraudulent transactions, and the potential factors contributing to fraudulent behavior.

The generated plots are saved in the `plots` directory for further reference.

## Fraud Detection Approach

The `src/fraud_detection.py` script implements the fraud detection system using a combination of anomaly detection and supervised learning techniques.

### Summary of the Approach

The focus of this approach is to provide a practical and efficient solution for fraud detection while allowing room for future enhancements and improvements.

For the first month (January), due to the lack of labeled data, Isolation Forest is used to identify the 400 most anomalous transactions.

From the second month onwards, a supervised learning model, XGBoost, is trained on the labeled data from the previous month. The model is updated each month using transfer learning, allowing it to learn from the newly labeled data while retaining knowledge from previous months. This enables the model to adapt to changing fraud patterns over time.

To select the 400 most likely fraudulent transactions each month, the anomaly scores from Isolation Forest and the predicted probabilities from the XGBoost model are combined by multiplying them together. The transactions with the highest combined scores are flagged for review.

This combination allows for prioritizing transactions that are identified as anomalous by both the unsupervised and supervised models, improving the chances of detecting true fraudulent activities.

The performance of the fraud detection system is evaluated using metrics such as precision and F1 score. Recall is not computed as all proposed transactions are considered as fraud in this setup.

### Limitations

While the current performance of the model may be poor, it can be greatly improved by implementing an online monitoring system. The current code simplifies the process by only reviewing transactions at the end of each month. However, based on the distribution of the time difference between reported fraud and transaction time.

To strike a balance between performance and efficiency, the script does not utilize GridSearchCV or other computationally intensive hyperparameter optimization algorithms. In a real-case scenario, such techniques may require significant computational resources and time, making them impractical for frequent model updates.

### Dealing with the reportedTime Feature

The `reported_delay` feature represents the time difference between the reported time of a fraud and the time the fraudulent transaction occurred.

Typically, in the financial industry, the reported time of a fraudulent transaction is only available after the transaction has been investigated and confirmed as fraud by the financial institution or the customer.

Therefore, to ensure the model's predictions are based on reliable information, we exclude the `reported_delay` feature from the prediction process.

The model should rely on the other engineered features and patterns learned from the previous months' data to make predictions for the current month.

However, if there is specific information about the availability of the `reported_delay` feature before fraud confirmation, it should be taken into consideration in the training procedure. This would greatly improve the performance of the training procedure (Check in the code by NOT dropping this feature for training and prediction).

## Model Performance Evaluation

The `src/plot_statistics.py` script evaluates the performance of the fraud detection system and generates visualizations to analyze the model's effectiveness over time.

The script creates the following plots:

1. Cumulative Amount Saved Over Time: This plot shows the cumulative amount of money saved by detecting and preventing fraudulent transactions each month. It helps assess the financial impact of the fraud detection system.

2. Precision and F1 Score Over Time: This plot displays the precision and F1 score of the model for each month. It allows monitoring the model's performance and identifying any trends or fluctuations.

3. Percentage of Correctly Detected Fraud per Month (Donut Plot): This plot visualizes the percentage of correctly detected fraudulent transactions out of the total number of fraudulent transactions in each month. It provides insights into the model's effectiveness in identifying true fraudulent activities.

These plots assist in evaluating the model's performance, making informed decisions about model improvements, and communicating the system's effectiveness to stakeholders.

The generated plots are saved in the `plots` directory for further analysis and reporting.

## Repository Structure

The repository is structured as follows:

- `src/`: Directory containing the source code files.
  - `data_preprocessing.py`: Script for data preprocessing and feature engineering.
  - `data_preprocessing_EDA.py`: Script for exploratory data analysis and generating insightful plots.
  - `fraud_detection.py`: Script implementing the fraud detection system using anomaly detection and supervised learning techniques.
  - `plot_statistics.py`: Script for evaluating the model's performance and generating visualizations.
  - `main.py`: Main script to orchestrate the execution of the fraud detection pipeline.
- `data/`: Directory containing the data files.
  - `raw/`: Contains the raw, unprocessed data files (`transactions_obf.csv` and `labels_obf.csv`).
  - `processed/`: Contains the preprocessed data file (`df_preprocessed.csv`).
- `logs/`: Directory containing the log files for script execution.
- `output/`: Directory containing the output files generated by the scripts, such as the model performance results (`df_fraud_detection_results.csv`).
- `plots/`: Directory containing the generated plots and visualizations.
- `config.json`: Configuration file specifying the file paths, model settings, and logging configuration.
- `README.md`: Documentation providing an overview of the fraud detection system and its components.
- `requirements.txt`: File listing the dependencies required to run the scripts.

## Configuration

The `config.json` file contains the configuration settings for the fraud detection system. It includes the following sections:

- `data_paths`: Specifies the paths to the input data files (`transactions_file` and `labels_file`).
- `output_paths`: Defines the output paths for the preprocessed data file (`preprocessed_data_file`), fraud detection results file (`fraud_detection_results_file`), and plots directory (`plots_dir`).
- `model_settings`: Contains the model-specific settings, such as the numeric columns to be scaled (`numeric_columns_to_scale`), columns to drop (`columns_to_drop`), the contamination parameter for the Isolation Forest algorithm, and the random state for reproducibility.
- `logging`: Specifies the log file path and log level.


