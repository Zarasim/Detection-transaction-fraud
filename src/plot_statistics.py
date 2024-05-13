import os
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Create output directory if it doesn't exist   
plots_dir = config['output_paths']['plots_dir']
os.makedirs(plots_dir, exist_ok=True)

# Set up logger
logger = logging.getLogger(__name__)

def load_data_file(fraud_detection_results: str) -> pd.DataFrame:
    """
    Load fraud detection results from CSV file.
    
    Args:
        fraud_detection_results (str): Path to the fraud detection results CSV file.
        
    Returns:
        DataFrame: fraud detection results.
    """
        
    try:
        df_results = pd.read_csv(fraud_detection_results)
    except FileNotFoundError as e:
        raise Exception('Cannot read fraud detection results') from e
    
    return df_results


def plot_cumulative_amount_saved(df_results: pd.DataFrame, plot_name: str) -> None:
    """
    Create a line plot of the cumulative transaction amount saved over time.
    
    Args:
        df_results (pd.DataFrame): DataFrame containing fraud detection results.
        plot_name (str): Name of the plot image file.
    """
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_results, x='Month', y='Cumulative Amount Saved', marker='o', linewidth=2, markersize=8)
    plt.title('Cumulative Transaction Amount Saved Over Time', fontsize=18)
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Transaction Amount (£)', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Format y-axis labels to display values in thousands (e.g., 10k)
    ylabels = [f"{int(y / 1000)}k" for y in plt.gca().get_yticks()]

    # Set your y ticks
    plt.gca().set_yticks(plt.gca().get_yticks())
    plt.gca().set_yticklabels(ylabels)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{plot_name}", dpi=300)
    plt.close()
    logger.info(f'Cumulative Amount Saved plot saved as {plot_name}')


def plot_precision_f1_score(df_results: pd.DataFrame, plot_name: str) -> None:
    """
    Create a line plot of the precision and F1 score over time.
    
    Args:
        df_results (pd.DataFrame): DataFrame containing fraud detection results.
        plot_name (str): Name of the plot image file.
    """
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_results, x='Month', y='Precision', marker='o', linewidth=2, markersize=8, label='Precision')
    sns.lineplot(data=df_results, x='Month', y='F1 Score', marker='o', linewidth=2, markersize=8, label='F1 Score')
    plt.title('Precision and F1 Score Over Time', fontsize=20)
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{plot_name}", dpi=300)
    plt.close()
    logger.info(f'Precision and F1 Score plot saved as {plot_name}')


def plot_percentage_correctly_detected_fraud(df_results: pd.DataFrame, plot_name: str) -> None:
    """
    Create a donut plot of the percentage of correctly detected fraud per month.
    
    Args:
        df_results (pd.DataFrame): DataFrame containing fraud detection results.
        plot_name (str): Name of the plot image file.
    """
    plt.figure(figsize=(12, 12))
    
    # Calculate the percentage of correctly detected fraud per month
    df_results['Percentage Correctly Detected Fraud'] = df_results['Number of True Frauds in the 400 samples'] / df_results['Number of True Frauds in the current month'] * 100
    
    # Create the donut plot
    sizes = df_results['Percentage Correctly Detected Fraud'].tolist()
    labels = df_results['Month'].astype(str).tolist()
    colors = ['#FF9B54', '#FFD56F', '#EFE7BC', '#A2D5AB', '#39ADA7', '#557B83', '#82C09A', '#A7D49B', '#C5E1A5', '#E6EE9C', '#FFF59D', '#FFE082']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85, textprops={'fontsize': 20})
    plt.title('Percentage of Correctly Detected Fraud per Month', fontsize=30)
    
    # Create a white circle in the center to make it a donut
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(center_circle)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{plot_name}", dpi=300)
    plt.close()
    logger.info(f'Percentage Correctly Detected Fraud plot saved as {plot_name}')


def main():
    """
    Main function to load fraud detection results and calculate additional statistics, generate plots.
    """

    # Load the results DataFrame from fraud_detection.py script
    fraud_detection_results_file = config['output_paths']['fraud_detection_results_file']
    df_results = load_data_file(fraud_detection_results_file)

    # Calculate additional statistics
    df_results['Cumulative Amount Saved'] = df_results['Transaction Amount Saved'].cumsum()
    
    # Generate plots
    plot_cumulative_amount_saved(df_results, "cumulative_amount_saved.png")
    plot_precision_f1_score(df_results, "precision_f1_score.png")
    plot_percentage_correctly_detected_fraud(df_results, "percentage_correctly_detected_fraud_donut.png")

if __name__ == '__main__':
    main()