# Housing Price Linear Regression Analysis

## Overview

This program analyzes housing data to predict house prices based on their square footage using linear regression. It takes a CSV file containing housing data, processes it, builds a predictive model, and visualizes the relationship between square footage and price. It can also predict the price of a new house based on its square footage.

## What is Linear Regression?

Linear regression is a simple but powerful statistical method that tries to find a linear relationship between input variables (features) and an output variable (target). In this case:

- **Input/Feature**: Square footage of a house
- **Output/Target**: Price of the house (in thousands of dollars)

The goal is to find the line that best fits the data points, represented by the equation:
```
Price = m × Square Footage + b
```
where:
- `m` is the slope (how much price increases when square footage increases by 1)
- `b` is the intercept (the theoretical price of a house with 0 square footage)

This "line of best fit" allows us to make predictions for new houses based on their square footage.

## Requirements

- Python 3.6+
- Required packages (install with `pip install package_name`):
  - numpy
  - pandas
  - matplotlib
  - scikit-learn

## Installation

1. Clone or download this repository
2. Install required packages:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

### Basic Usage

Run the script with a CSV file containing housing data:

```bash
python app.py -f your_housing_data.csv
```

### Command Line Options

- `-f, --file`: Path to CSV file (default: house_data.csv)
- `--no-plot`: Do not display the plot (still saves to file)
- `-predict, --predict-sqft`: Predict the price for a house with the given square footage

Examples:
```bash
# Run with custom data file
python app.py -f custom_data.csv

# Run without displaying the plot
python app.py --no-plot

# Predict price for a house with 2000 square feet
python app.py --predict-sqft 2000

# Combine options
python app.py -f custom_data.csv --predict-sqft 1750 --no-plot
```

### Input Data Format

The CSV file should have at least these two columns:
- `square_footage`: The size of the house in square feet
- `price_thousands`: The price in thousands of dollars

Example CSV:
```
square_footage,price_thousands
1500,235
1800,285
2200,340
...
```

## How the Program Works

The program follows these steps:

1. **Loading Data**: Reads the CSV file and verifies it has the required columns.
2. **Preprocessing**: Cleans the data by removing missing values and outliers.
3. **Train/Test Split**: Divides data into training (80%) and test (20%) sets.
4. **Training**: Finds the best line that fits the training data.
5. **Evaluation**: Tests how well the model performs on unseen data.
6. **Visualization**: Creates a plot showing the data points and regression line.
7. **Prediction (optional)**: If requested via command line, predicts the price for a house with the specified square footage.

## Understanding the Output

### Terminal Output

The program will print:
- The regression formula (Price = m × Square Footage + b)
- R-squared values for training and test data (measure of model fit)
- RMSE (Root Mean Squared Error) for training and test data (average prediction error)
- Sample results showing actual vs. predicted prices
- If requested, the predicted price for a house with the specified square footage

Example output:
```
Linear Regression Formula: Price = 0.1542 × Square Footage + 35.4212
R-squared (training): 0.8734
R-squared (test): 0.8521
RMSE (training): 15.4327
RMSE (test): 16.2145

Training Data Sample (first 5 rows):
Square Footage  Actual Price ($K)  Predicted Price ($K)
          1500            235.00               266.52
          1800            285.00               312.78
          2200            340.00               374.45
          1650            255.00               289.65
          1950            310.00               336.40

Test Data Sample (first 5 rows):
Square Footage  Actual Price ($K)  Predicted Price ($K)
          1750            270.00               305.57
          2100            325.00               359.04
          1600            245.00               282.09
          2000            315.00               343.62
          1850            295.00               320.40

Predicted price for a house with 2000 square footage: $343.62 thousand
This is equivalent to approximately $343,620.00
```

### Metrics Explained

1. **R-squared (R²)**: 
   - Range: 0 to 1 (higher is better)
   - Interpretation: Percentage of price variation explained by square footage
   - Example: R² = 0.85 means square footage explains 85% of price variation

2. **RMSE (Root Mean Squared Error)**:
   - Lower is better
   - Interpretation: Average prediction error in the same units (thousands of dollars)
   - Example: RMSE = 15 means predictions are off by ~$15,000 on average

### Visualization

The program generates a plot showing:
- Training data points (blue)
- Test data points (green)
- The regression line (red)
- The model formula and R² values

This plot is saved as `housing_regression.png` in the same directory.

## Making Predictions

The new prediction feature allows you to estimate the price of a house based on its square footage:

1. After training the model on your data, the program can predict the price of a house with any square footage.
2. Use the `-predict` or `--predict-sqft` flag followed by the square footage value.
3. The output will show both the price in thousands of dollars and the full dollar amount.

Example:
```bash
python app.py --predict-sqft 2500
```

This might output:
```
Predicted price for a house with 2500 square footage: $420.72 thousand
This is equivalent to approximately $420,720.00
```

## Tips for Better Results

1. **Data Quality**: The more houses in your dataset, the better the model will likely be.
2. **Additional Features**: This model only uses square footage. Real housing prices depend on many factors like location, bedrooms, etc.
3. **Outliers**: The program automatically removes outliers, but you might want to review your data first.
4. **Prediction Range**: For best results, predict within the range of your training data. Extrapolating far beyond your data range may lead to less reliable predictions.

## How Machine Learning Works Here

This program uses machine learning but keeps it simple:

1. **Learning Phase**: The algorithm finds the best parameters (slope and intercept) by minimizing the error between predicted and actual prices in the training data.
2. **Testing Phase**: We check if these parameters work well on new data the model hasn't seen before.
3. **Scaling**: The program scales the data to have similar ranges, which is a common practice in machine learning.
4. **Prediction**: Once trained, the model can predict prices for new square footage values using the learned relationship.

## Limitations

1. **Linear Assumption**: The model assumes a straight-line relationship between square footage and price, which may not always be true.
2. **Single Feature**: Real estate prices depend on many factors beyond just square footage.
3. **No Geographic Context**: Housing markets vary dramatically by location.

## Future Improvements

Some ways this model could be enhanced:
1. Add more features (bedrooms, bathrooms, location, etc.)
2. Try non-linear models for potentially better fit
3. Add geographic data to account for location differences
4. Implement batch prediction from a file of house square footages