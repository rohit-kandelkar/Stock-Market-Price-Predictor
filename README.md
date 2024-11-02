# Stock-Market-Price-Predictor
## Description

This project predicts stock prices for Tata Motors, BHEL, Zomato, TCS, and OIL using Linear Regression, SVM, Random Forest, and XGBoost. We trained models on historical data, evaluated using MSE and R-squared. Future enhancements include deep learning techniques like LSTM for better predictions.

## Introduction

This repository contains the research paper, code, and results for the project "Stock Market Prediction Using Machine Learning: A Comparative Study of Supervised Algorithms." We focus on predicting stock prices for companies such as Tata Motors, BHEL, Zomato, TCS, and OIL using various machine learning models.

## Dataset

Source: Historical stock market data sourced from National Stock Exchange (NSE) Website.

Description: Includes features such as Date, OPEN, HIGH, LOW, PREV. CLOSE, ltp (Last Traded Price), close, vwap (Volume-Weighted Average Price), 52W H (52-Week High), 52W L (52-Week Low), VOLUME, VALUE, and No of trades.

## Models Implemented

(1) Linear Regression<br> 
(2) Support Vector Machines (SVM)<br>
(3) Random Forest<br>
(4) XGBoost<br>
(5) Adaptive Boosting (extra)<br> 

## Results

 (i) Performance Metrics: Evaluated using Mean Squared Error (MSE) and R-squared values.

 (ii) Comparison: Summary of model performance:

      (a) Random Forest: Struggles to capture upward trends.<br>
      (b) XGBoost: Slight improvement, but needs better tuning.<br>
      (c) SVM: Flat predictions, indicating poor adaptability.<br>
      (d) Linear Regression: Closely follows actual trends but requires more features for accuracy.<br>

 (iii) Visualizations: Actual vs. predicted price plots for each model.

## Future Work

Future enhancements include incorporating deep learning techniques like Long Short-Term Memory (LSTM) networks for more robust predictions. Regular updates and additional features such as macroeconomic indicators can improve model accuracy.

## Acknowledgements

We extend our gratitude to the National Stock Exchange (NSE) repositories for providing historical stock data. Special thanks to our academic advisor, Dr. Sujata Kolhe ma’am for their valuable insights into the stock market domain.

## Contributors

Yash Jahagirdar - Project Lead (https://github.com/Yash-Jahagirdar)
Teammate 1 - Rohit Kandelkar (https://github.com/rohit-kandelkar)
Teammate 2 - Rajeshwari Golande
