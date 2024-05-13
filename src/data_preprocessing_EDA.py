import os
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Create output directory if it doesn't exist   
plots_dir = config['output_paths']['plots_dir']
os.makedirs(plots_dir, exist_ok=True)

# Set up logger
logger = logging.getLogger(__name__)

def load_data_file(preprocessed_data_file: str) -> pd.DataFrame:
    """
    Load preprocessed fraud dataframe from CSV file.
    
    Args:
        preprocessed_data_file (str): Path to the preprocessed data CSV file.
        
    Returns:
        DataFrame: preprocessed fraud dataframe.
    """
           
    try:
        df = pd.read_csv(preprocessed_data_file) 
    except FileNotFoundError as e:
        raise Exception('Cannot read preprocessed data') from e
    
    return df


def save_histplot(data: pd.DataFrame, column: str, plot_name: str) -> None:
    """
    Create a histogram plot of the specified column.
    
    Args:
        data (DataFrame): Input DataFrame.
        column (str): Column name to plot the histogram.
        plot_name (str): Name of the plot image file.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data[data[column] > 0][column], fill=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{plot_name}")
    plt.close()
    logger.info(f'Histogram of {column} saved as {plot_name}')


def save_corrplot(data: pd.DataFrame, plot_name: str) -> None:
    """
    Create a correlation plot of the input DataFrame.
    
    Args:
        data (DataFrame): Input DataFrame.
        plot_name (str): Name of the plot image file.
    """
    plt.figure(figsize=(10, 10))
    corr = data.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False, square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{plot_name}")
    plt.close()
    logger.info(f'Correlation plot saved as {plot_name}')


def save_freqplot(data: pd.DataFrame, column: str, plot_name: str) -> None:
    """
    Create a frequency plot of the specified column.
    
    Args:
        data (DataFrame): Input DataFrame.
        column (str): Column name to plot the frequency plot.
        plot_name (str): Name of the plot image file.
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate the frequency of each category
    freq = data[column].value_counts(normalize=True)
    
    # Create a bar plot of frequencies
    sns.barplot(x=freq.index, y=freq.values)
    
    plt.title(f'Frequency Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(range(len(freq.index)), freq.index.astype(str))
    plt.tight_layout()
    plt.savefig(f"{config['output_paths']['plots_dir']}/{plot_name}")
    plt.close()
    logger.info(f'Frequency plot of {column} saved as {plot_name}')


def save_boxplots(data: pd.DataFrame, quantitative_features: List[str], plot_name: str) -> None:
    """
    Create boxplots for the specified quantitative features.
    
    Args:
        data (DataFrame): Input DataFrame.
        quantitative_features (list): List of quantitative feature names.
        plot_name (str): Name of the plot image file.
    """
    num_features = len(quantitative_features)
    num_rows = (num_features + 1) // 2
    num_cols = 2
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 4))
    fig.suptitle('Boxplots of Quantitative Features')
    
    for i, feature in enumerate(quantitative_features):
        row = i // num_cols
        col = i % num_cols
        
        if num_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        
        sns.boxplot(x='Fraud', y=feature, data=data, ax=ax)
        ax.set_title(feature)
        ax.set_xlabel('Fraud')
        ax.set_ylabel(feature)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{plots_dir}/{plot_name}")
    plt.close()
    logger.info(f'Boxplots of quantitative features saved as {plot_name}')


def main():
    """
    Main function to generate EDA plots.
    """

    # Load preprocessed data
    preprocessed_data_file = config['output_paths']['preprocessed_data_file']
    logger.info(f'Loading preprocessed data from {preprocessed_data_file}')
    df = load_data_file(preprocessed_data_file)

    # Create plots
    save_histplot(df, 'reported_delay', 'histogram_reported_delay.png')
    save_corrplot(df, 'correlation_matrix.png')
    save_freqplot(df, 'Fraud', 'freqplot_fraud.png')
    
    # Specify the quantitative features
    quantitative_features = ['transactionAmount', 'availableCash', 'reported_delay', 'time_since_last_transaction']

    # Create boxplots
    save_boxplots(df, quantitative_features, 'boxplots_quantitative_features.png')

if __name__ == '__main__':
    main()