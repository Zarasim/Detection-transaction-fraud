import os
import json
import logging
import argparse
from data_preprocessing import main as preprocess_data
from fraud_detection import main as run_fraud_detection
from data_preprocessing_EDA import main as run_eda_script
from plot_statistics import main as plot_statistics

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

log_file = config['logging']['log_file']
log_level = config['logging']['log_level']

# Create log directory if it doesn't exist
os.makedirs(os.path.dirname(log_file),exist_ok=True)

# Configure logging
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()])

# Configure the logger
logger = logging.getLogger(__name__)

def main(run_eda=False):
    # Preprocess data
    logger.info('Starting data preprocessing...')
    preprocess_data()
    logger.info('Data preprocessing completed.')

    if run_eda:
        # Run EDA
        logger.info('Starting exploratory data analysis...')
        run_eda_script()
        logger.info('Exploratory data analysis completed.')

    # Run fraud detection
    logger.info('Starting fraud detection...')
    run_fraud_detection()
    logger.info('Fraud detection completed.')

    # Plot statistics
    logger.info('Plotting statistics...')
    plot_statistics()
    logger.info('Plotting statistics completed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument('--run_eda', action='store_true', default=False, help='Run EDA before fraud detection')
    args = parser.parse_args()

    main(run_eda=args.run_eda)
