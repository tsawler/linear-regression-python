#!/usr/bin/env python3
"""
Housing Price Linear Regression Analysis

This script performs linear regression analysis on housing data,
predicting prices based on square footage. It includes data validation,
preprocessing, model training/testing, and visualization.

What is Linear Regression?
-------------------------
Linear regression is a statistical method that attempts to find a linear relationship 
between input variables (features) and an output variable (target). 

In this case, we're trying to find the relationship between:
- Input/Feature: Square footage of a house
- Output/Target: Price of the house (in thousands of dollars)

The goal is to find the line that best fits the data points, represented by the equation:
    y = mx + b
which in this case is:
    Price = m × Square Footage + b
where:
- m is the slope (how much price increases when square footage increases by 1)
- b is the intercept (the theoretical price of a house with 0 square footage)

This "line of best fit" allows us to make predictions for new houses based on their square footage.
"""

import sys
import argparse
import os
import logging
import numpy as np  # For numerical operations and array handling
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For creating visualizations
from sklearn.linear_model import LinearRegression  # The actual model we'll use
from sklearn.metrics import r2_score, mean_squared_error  # For evaluating model performance
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets
from sklearn.preprocessing import StandardScaler  # For normalizing our data

# Configuration constants
# These values control various aspects of the analysis and visualization
CONFIG = {
    'test_size': 0.2,       # 20% of data used for testing, 80% for training
    'random_state': 42,     # Seed for random operations, ensures reproducibility
    'figure_size': (10, 6), # Size of the visualization plot (width, height)
    'point_color': 'blue',  # Color for data points
    'line_color': 'red',    # Color for regression line
    'line_width': 2,        # Thickness of regression line
    'grid_alpha': 0.3,      # Transparency of grid lines (0=invisible, 1=solid)
    'default_csv': 'house_data.csv',  # Default data file if none specified
    'output_image': 'housing_regression.png'  # Where to save the visualization
}

# Set up logging
# Logging helps us track what the program is doing and any issues that arise
logging.basicConfig(
    level=logging.INFO,  # Show all messages at INFO level or higher (INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format: timestamp - level - message
)
logger = logging.getLogger(__name__)  # Create a logger for this script

def parse_arguments():
    """
    Parse command line arguments.
    
    This function defines what command-line options the script accepts:
    - A file path for the input CSV data
    - An option to hide the plot (but still save it to a file)
    
    Command-line arguments let users customize the script's behavior without changing the code.
    """
    parser = argparse.ArgumentParser(
        description='Linear regression analysis on housing data.'
    )
    parser.add_argument(
        '-f', '--file', 
        type=str, 
        default=CONFIG['default_csv'],
        help=f'Path to CSV file (default: {CONFIG["default_csv"]})'
    )
    parser.add_argument(
        '--no-plot', 
        action='store_true',  # This is a flag (doesn't take a value, just present or not)
        help='Do not display the plot (still saves to file)'
    )
    return parser.parse_args()

def load_data(file_path):
    """
    Load and validate housing data from CSV file.
    
    This function:
    1. Checks if the specified file exists
    2. Attempts to load it as a CSV
    3. Verifies that it contains the required columns
    
    The CSV file should have at least these two columns:
    - 'square_footage': The size of the house (our predictor/feature)
    - 'price_thousands': The price in thousands of dollars (our target)
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Validated DataFrame with at least the required columns
        
    Exits:
        - If file doesn't exist
        - If required columns are missing
        - If there's an error reading the file
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        # Log the error and exit if file doesn't exist
        logger.error(f"File does not exist: {file_path}")
        sys.exit(1)  # Exit with error code 1
    
    try:
        # Load data
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)  # Read the CSV file into a pandas DataFrame
        
        # Validate required columns
        # For linear regression, we need both our feature (square_footage) and target (price)
        required_columns = ['square_footage', 'price_thousands']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in the CSV file")
                sys.exit(1)  # Exit with error code 1
        
        return df
    
    except Exception as e:
        # Catch any other errors that might occur
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)  # Exit with error code 1

def preprocess_data(df):
    """
    Preprocess housing data for regression analysis.
    
    Data preprocessing is crucial because real-world data is often messy.
    This function:
    1. Removes missing values (NaN)
    2. Removes outliers (data points that are too extreme)
    3. Ensures all data is numeric
    
    Why remove outliers?
    -------------------
    Outliers are data points that are very different from most of the data.
    For example, a mansion might be 10x larger than a typical house.
    Including outliers can skew our regression line, making it less accurate for typical cases.
    
    Args:
        df (pd.DataFrame): Raw DataFrame with house data
        
    Returns:
        pd.DataFrame: Cleaned DataFrame ready for analysis
    """
    logger.info("Preprocessing data")
    
    # Make a copy to avoid modifying the original
    # This is a good practice in data analysis - preserve your original data
    processed_df = df.copy()
    
    # Handle missing values (NaN)
    # Missing values can't be used in calculations, so we need to remove them
    if processed_df[['square_footage', 'price_thousands']].isna().any().any():
        logger.warning("Missing values found, dropping rows with missing values")
        processed_df = processed_df.dropna(subset=['square_footage', 'price_thousands'])
    
    # Filter out potential outliers (very simple method - could be improved)
    # This removes data points that are more than 3 standard deviations from the mean
    # In a normal distribution, 99.7% of data falls within 3 standard deviations
    for col in ['square_footage', 'price_thousands']:
        mean = processed_df[col].mean()  # Average value
        std = processed_df[col].std()    # Standard deviation (measure of spread)
        lower_bound = mean - 3 * std     # Lower threshold
        upper_bound = mean + 3 * std     # Upper threshold
        
        # Identify outliers
        outliers = (processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)
        if outliers.any():
            logger.warning(f"Removing {outliers.sum()} outliers from {col}")
            processed_df = processed_df[~outliers]  # Keep only non-outliers
    
    # Ensure numeric types
    # Sometimes CSV data might be loaded as strings - convert to numbers
    processed_df['square_footage'] = pd.to_numeric(processed_df['square_footage'], errors='coerce')
    processed_df['price_thousands'] = pd.to_numeric(processed_df['price_thousands'], errors='coerce')
    
    # Drop any rows that couldn't be converted to numeric
    # 'coerce' above converts invalid values to NaN, which we now remove
    processed_df = processed_df.dropna(subset=['square_footage', 'price_thousands'])
    
    return processed_df

def train_model(X, y):
    """
    Train a linear regression model.
    
    What is training?
    ---------------
    "Training" means finding the best parameters (slope and intercept) for our linear equation
    to make our predictions as close as possible to the actual prices.
    
    What is scaling?
    --------------
    Scaling means transforming our data so values are in a similar range.
    This is often helpful for machine learning algorithms.
    For linear regression on one feature, it's not strictly necessary,
    but we do it as a best practice.
    
    Args:
        X (np.ndarray): Features array (square footage values)
        y (np.ndarray): Target array (house prices)
        
    Returns:
        tuple: (model, scaler) - Trained model and feature scaler
    """
    # Scale features
    # StandardScaler transforms data to have mean=0 and standard deviation=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Learn scaling parameters and apply them
    
    # Train the model
    # LinearRegression finds the best line that fits our data points
    model = LinearRegression()
    model.fit(X_scaled, y)  # Fit = find the best parameters
    
    return model, scaler

def evaluate_model(model, X, y, scaler):
    """
    Evaluate the model performance.
    
    What are these metrics?
    ---------------------
    R-squared (r2_score): Measures how well the model explains the variation in house prices
      - Range: 0 to 1 (higher is better)
      - Example: R² = 0.75 means the model explains 75% of price variation
    
    RMSE (Root Mean Squared Error): Average prediction error in the same units as the target
      - Lower is better
      - Example: RMSE = 15 means predictions are off by ~15 thousand dollars on average
    
    Args:
        model: Trained LinearRegression model
        X (np.ndarray): Features array (square footage values)
        y (np.ndarray): Target array (actual house prices)
        scaler: Feature scaler to transform the data
        
    Returns:
        tuple: (predictions, r_squared, rmse) - Model predictions and quality metrics
    """
    X_scaled = scaler.transform(X)  # Apply the same scaling we used during training
    predictions = model.predict(X_scaled)  # Generate price predictions
    
    # Calculate quality metrics
    r_squared = r2_score(y, predictions)  # How well the model explains price variation
    rmse = np.sqrt(mean_squared_error(y, predictions))  # Average prediction error
    
    return predictions, r_squared, rmse

def create_visualization(X_train, y_train, X_test, y_test, 
                        train_predictions, test_predictions, 
                        model, scaler, output_file, show_plot=True):
    """
    Create and save visualization of the regression results.
    
    This function creates a plot showing:
    1. Training data points (blue)
    2. Test data points (green)
    3. The regression line (red)
    4. Model performance metrics and equation
    
    Args:
        X_train, y_train: Training data (square footage and prices)
        X_test, y_test: Test data (square footage and prices)
        train_predictions, test_predictions: Model predictions for both sets
        model: Trained model
        scaler: Feature scaler
        output_file (str): Path to save the output image
        show_plot (bool): Whether to display the plot
    """
    plt.figure(figsize=CONFIG['figure_size'])  # Create figure with specified size
    
    # Plot training data
    plt.scatter(X_train, y_train, color=CONFIG['point_color'], 
                alpha=0.7, label='Training data')  # Alpha controls transparency
    
    # Plot test data
    plt.scatter(X_test, y_test, color='green', 
                alpha=0.7, label='Test data')
    
    # Create range for regression line
    # We need to create a range of x-values to draw our line
    x_range = np.linspace(
        min(X_train.min(), X_test.min()),  # Minimum square footage
        max(X_train.max(), X_test.max()),  # Maximum square footage
        100  # 100 points for a smooth line
    ).reshape(-1, 1)  # Reshape to 2D array as required by sklearn
    
    # Scale the range and predict the corresponding y-values
    x_range_scaled = scaler.transform(x_range)
    y_range_pred = model.predict(x_range_scaled)
    
    # Plot regression line
    plt.plot(x_range, y_range_pred, color=CONFIG['line_color'], 
            linewidth=CONFIG['line_width'], label='Regression line')
    
    # Add labels and title
    plt.xlabel('Square Footage')
    plt.ylabel('Price (thousands $)')
    plt.title('Linear Regression: Housing Price vs Square Footage')
    plt.legend()  # Add legend based on labels
    plt.grid(True, alpha=CONFIG['grid_alpha'])  # Add grid lines
    
    # Calculate and display model parameters
    # We need to convert the scaled model parameters back to the original scale
    slope = model.coef_[0] / scaler.scale_[0]  # Adjust slope for scaling
    intercept = model.intercept_ - (model.coef_[0] * scaler.mean_[0] / scaler.scale_[0])  # Adjust intercept
    
    # Add formula and metrics
    r_squared_train = r2_score(y_train, train_predictions)
    r_squared_test = r2_score(y_test, test_predictions)
    
    # Format text to display on the plot
    formula_text = f"Price = {slope:.4f} × Square Footage + {intercept:.4f}"
    r2_train_text = f"R² (train): {r_squared_train:.4f}"
    r2_test_text = f"R² (test): {r_squared_test:.4f}"
    
    # Add text to the plot
    plt.figtext(0.15, 0.85, formula_text, fontsize=12)
    plt.figtext(0.15, 0.82, r2_train_text, fontsize=12)
    plt.figtext(0.15, 0.79, r2_test_text, fontsize=12)
    
    # Save the figure
    plt.savefig(output_file)
    logger.info(f"Plot saved as '{output_file}'")
    
    # Display if requested
    if show_plot:
        plt.show()
    
    plt.close()  # Close the figure to free memory

def print_results(X_train, y_train, X_test, y_test, 
                 train_predictions, test_predictions, model, scaler):
    """
    Print regression results and metrics.
    
    This function:
    1. Calculates and prints the model formula (slope and intercept)
    2. Prints quality metrics (R-squared, RMSE)
    3. Shows sample predictions vs actual values
    
    What does this tell us?
    ---------------------
    - The formula shows exactly how price relates to square footage
    - The R-squared tells us how much of price variation is explained by square footage
    - The RMSE gives us the average prediction error
    - The sample results let us see individual predictions
    
    Args:
        X_train, y_train: Training data (square footage and prices)
        X_test, y_test: Test data (square footage and prices)
        train_predictions, test_predictions: Model predictions
        model: Trained model
        scaler: Feature scaler
    """
    # Calculate adjusted coefficients (to account for scaling)
    # We convert the scaled model parameters back to the original scale
    slope = model.coef_[0] / scaler.scale_[0]  # Adjust for feature scaling
    intercept = model.intercept_ - (model.coef_[0] * scaler.mean_[0] / scaler.scale_[0])  # Adjust intercept
    
    # Calculate metrics
    # R-squared: Proportion of variance explained (higher is better)
    # RMSE: Average prediction error in same units as price (lower is better)
    r_squared_train = r2_score(y_train, train_predictions)
    r_squared_test = r2_score(y_test, test_predictions)
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    # Print formula and metrics
    print(f"\nLinear Regression Formula: Price = {slope:.4f} × Square Footage + {intercept:.4f}")
    print(f"R-squared (training): {r_squared_train:.4f}")
    print(f"R-squared (test): {r_squared_test:.4f}")
    print(f"RMSE (training): {rmse_train:.4f}")
    print(f"RMSE (test): {rmse_test:.4f}")
    
    # Create DataFrames for training and test results
    # This gives us a nice way to display the actual vs predicted values
    train_df = pd.DataFrame({
        'Square Footage': X_train.flatten(),  # Convert 2D array to 1D
        'Actual Price ($K)': y_train,
        'Predicted Price ($K)': np.round(train_predictions, 2)
    })
    
    test_df = pd.DataFrame({
        'Square Footage': X_test.flatten(),
        'Actual Price ($K)': y_test,
        'Predicted Price ($K)': np.round(test_predictions, 2)
    })
    
    # Print sample of results
    print("\nTraining Data Sample (first 5 rows):")
    print(train_df.head().to_string(index=False))
    
    print("\nTest Data Sample (first 5 rows):")
    print(test_df.head().to_string(index=False))

def main():
    """
    Main function to execute the analysis pipeline.
    
    This function coordinates the entire analysis process:
    1. Parse command-line arguments
    2. Load and preprocess the data
    3. Split data into training and test sets
    4. Train the model
    5. Evaluate the model
    6. Print results and visualize
    
    What is train/test splitting?
    ---------------------------
    We divide our data into two parts:
    - Training data: Used to find the best model parameters
    - Test data: Used to evaluate how well the model performs on new data
    
    This helps ensure our model will work well on houses it hasn't seen before.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Load and preprocess data
    df = load_data(args.file)  # Load raw data from CSV
    processed_df = preprocess_data(df)  # Clean and prepare the data
    
    # Prepare data for modeling
    # X: feature (square footage), y: target (price)
    X = processed_df['square_footage'].values.reshape(-1, 1)  # 2D array required by sklearn
    y = processed_df['price_thousands'].values
    
    # Split data into training and test sets
    # This is a crucial step - we use separate data for training and evaluation
    # to check if our model generalizes well to new data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'],  # Proportion of data used for testing
        random_state=CONFIG['random_state']  # For reproducible results
    )
    logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Train model
    # This learns the relationship between square footage and price
    model, scaler = train_model(X_train, y_train)
    
    # Evaluate model on both training and test data
    # This tells us how well our model performs
    train_predictions, train_r2, train_rmse = evaluate_model(model, X_train, y_train, scaler)
    test_predictions, test_r2, test_rmse = evaluate_model(model, X_test, y_test, scaler)
    
    logger.info(f"Model trained. R² (train): {train_r2:.4f}, R² (test): {test_r2:.4f}")
    
    # Print results
    print_results(X_train, y_train, X_test, y_test, 
                 train_predictions, test_predictions, model, scaler)
    
    # Create visualization
    create_visualization(
        X_train, y_train, X_test, y_test,
        train_predictions, test_predictions,
        model, scaler, 
        CONFIG['output_image'],
        not args.no_plot  # Show plot unless --no-plot flag was used
    )

# Python's way of identifying if this script is being run directly
# (versus being imported by another script)
if __name__ == "__main__":
    main()