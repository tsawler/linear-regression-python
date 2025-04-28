import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the CSV data
df = ""

if len(sys.argv) > 1:
    df = sys.argv[1]
else:
    df = 'house_data.csv'


df = pd.read_csv(df)

# Prepare data for linear regression
X = df['square_footage'].values.reshape(-1, 1)  # Independent variable
y = df['price_thousands'].values  # Dependent variable

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get model parameters
slope = model.coef_[0]
intercept = model.intercept_

# Generate predictions for the data points
predictions = model.predict(X)

# Calculate R-squared to measure goodness of fit
r_squared = r2_score(y, predictions)

# Create a DataFrame to display the original data and predictions
results_df = pd.DataFrame({
    'Square Footage': df['square_footage'],
    'Actual Price ($K)': df['price_thousands'],
    'Predicted Price ($K)': np.round(predictions, 2)
})

# Print the regression formula
print(f"Linear Regression Formula: Price = {slope:.4f} × Square Footage + {intercept:.4f}")
print(f"R-squared: {r_squared:.4f}")

# Display the data table
print("\nData Table:")
print(results_df.to_string(index=False))

# Create the regression plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual data points')
plt.plot(X, predictions, color='red', linewidth=2, label='Regression line')
plt.xlabel('Square Footage')
plt.ylabel('Price (thousands $)')
plt.title('Linear Regression: Housing Price vs Square Footage')
plt.legend()
plt.grid(True, alpha=0.3)

# Add the formula and R² as text on the plot
formula_text = f"Price = {slope:.4f} × Square Footage + {intercept:.4f}"
r2_text = f"R² = {r_squared:.4f}"
plt.figtext(0.15, 0.85, formula_text, fontsize=12)
plt.figtext(0.15, 0.82, r2_text, fontsize=12)

# Save the figure
plt.savefig('housing_regression.png')

# Display the plot
plt.show()

print("\nPlot has been saved as 'housing_regression.png'")
