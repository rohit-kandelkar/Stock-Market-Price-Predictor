import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg

# Function to load the dataset based on user's choice
def load_stock_data(stock_choice):
    stock_files = {
        'BHEL': 'BHEL.csv',
        'OIL': 'OIL.csv',
        'TataMotors': 'TataMotors.csv',
        'TCS': 'TCS.csv',
        'Zomato': 'Zomato.csv'
    }
    
    df = pd.read_csv(stock_files[stock_choice])
    print("Columns in the dataset:", df.columns)  # To verify column names
    df.columns = df.columns.str.strip()  # Remove any leading or trailing spaces from column names
    return df

# Function to clean and convert relevant columns to float
def clean_numeric_columns(df):
    # Convert all relevant columns to string
    for col in ['OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', 'ltp', 'close', 'vwap', '52W H', '52W L', 'VOLUME', 'VALUE', 'No of trades']:
        df[col] = df[col].astype(str)  # Convert to string
        df[col] = df[col].str.replace(',', '')  # Remove commas
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float, force errors to NaN

    # Debugging: Check for any non-numeric values after cleaning
    print("Data after cleaning:")
    print(df[['VOLUME', 'VALUE', 'No of trades']].head())
    
    return df


# Function to preprocess the data
def preprocess_data(df):
    # Parse the 'Date' column with the correct format
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    
    # Set the 'Date' column as the index for easy plotting later
    df.set_index('Date', inplace=True)
    
    # Clean numeric columns
    df = clean_numeric_columns(df)
    df = df.dropna()
    # Selecting features that influence close price
    X = df[['OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', 'vwap', '52W H', '52W L', 'VOLUME', 'VALUE', 'No of trades']]
    y = df['close']
    
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test

# Function to train and evaluate the AdaBoost Regressor model
def train_model(X_train, y_train, X_test):
    model = AdaBoostRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    return model, predictions

# Function to plot predicted vs actual prices and forecast
def plot_predictions(y_test, predictions, forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
    plt.plot(y_test.index, predictions, label='Predicted Price', color='red')
    
    # Prepare the forecast index for 60 days
    forecast_index = pd.date_range(start=y_test.index[-1] + pd.Timedelta(days=1), periods=60, freq='B')
    
    plt.plot(forecast_index, forecast, label='Forecast Price', color='green', linestyle = 'dotted')
    
    plt.title('Actual, Predicted, Forecasted using AdaBoost Regression')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Function to forecast using AutoReg
def forecast_with_autoreg(df):
    model = AutoReg(df['close'], lags=5)  # Using 5 lags, you can adjust this
    model_fit = model.fit()
    
    # Forecasting for the next 60 periods
    forecast = model_fit.predict(start=len(df), end=len(df) + 59)
    return forecast

# Main function
def main():
    # Stock options
    stock_options = ['BHEL', 'OIL', 'TataMotors', 'TCS', 'Zomato']
    print("Available stock options:", stock_options)
    
    # User selects a stock
    stock_choice = input("Choose a stock from the options above: ")
    while stock_choice not in stock_options:
        print("Invalid choice. Please choose a valid stock.")
        stock_choice = input("Choose a stock from the options above: ")
    
    # Load the selected stock data
    df = load_stock_data(stock_choice)
    
    # Preprocess the selected stock data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Train the model
    model, predictions = train_model(X_train, y_train, X_test)
    
    # Forecast using AutoReg
    forecast = forecast_with_autoreg(df)
    
    # Plot the results
    plot_predictions(y_test, predictions, forecast)
    
    # Calculate and display model performance
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.2f}")
    
    # Predict today's close price based on user input for today's open price
    today_open = float(input(f"Enter today's OPEN price for {stock_choice}: "))
    today_features = np.array([[today_open, df['HIGH'].iloc[-1], df['LOW'].iloc[-1], 
                                df['PREV. CLOSE'].iloc[-1], df['vwap'].iloc[-1], 
                                df['52W H'].iloc[-1], df['52W L'].iloc[-1], 
                                df['VOLUME'].iloc[-1], df['VALUE'].iloc[-1], 
                                df['No of trades'].iloc[-1]]])
    today_close_pred = model.predict(today_features)
    print(f"Predicted close price for today: {today_close_pred[0]:.2f}")

# Run the main function
if __name__ == "__main__":
    main()
