# 📈 Stock Market Price Predictor

## 📝 Description

This project predicts stock prices for **Tata Motors**, **BHEL**, **Zomato**, **TCS**, and **OIL** using **Linear Regression**, **SVM**, **Random Forest**, and **XGBoost**. We trained models on historical data, evaluated using 📉 **Mean Squared Error (MSE)** and 📊 **R-squared**. Future enhancements include deep learning techniques like **LSTM** for better predictions.

---

## 📘 Introduction

This repository contains the **research paper**, **code**, and **results** for the project **"Stock Market Prediction Using Machine Learning: A Comparative Study of Supervised Algorithms."**  
We focus on predicting stock prices for companies such as Tata Motors, BHEL, Zomato, TCS, and OIL using various machine learning models.

---

## 📂 Dataset

- **📌 Source**: Historical stock market data sourced from [National Stock Exchange (NSE)](https://www.nseindia.com).
- **📋 Description**:  
  Features include:
  - Date
  - OPEN
  - HIGH
  - LOW
  - PREV. CLOSE
  - ltp (Last Traded Price)
  - close
  - vwap (Volume-Weighted Average Price)
  - 52W H (52-Week High)
  - 52W L (52-Week Low)
  - VOLUME
  - VALUE
  - No. of trades

---

## 🤖 Models Implemented

1. 📈 Linear Regression  
2. 🔧 Support Vector Machines (SVM)  
3. 🌲 Random Forest  
4. 🚀 XGBoost  
5. ⚡ Adaptive Boosting *(extra)*  

---

## 📊 Results

- **📐 Performance Metrics**: Evaluated using **Mean Squared Error (MSE)** and **R-squared values**.

- **📉 Comparison**:  
  - 🟠 **Random Forest**: Struggles to capture upward trends.  
  - 🔵 **XGBoost**: Slight improvement, but needs better tuning.  
  - 🟢 **SVM**: Flat predictions, indicating poor adaptability.  
  - 🔴 **Linear Regression**: Closely follows actual trends but requires more features for accuracy.

- **📈 Visualizations**:  
  Actual vs. predicted price plots for each model are available in the notebook.

---

## 🔮 Future Work

📌 Future enhancements may include:
- 🧠 Deep learning models like **LSTM (Long Short-Term Memory)**.
- 📈 Addition of **macroeconomic indicators** for improved accuracy.
- 🔄 **Real-time data updates** for continuous learning and prediction.

---

## 🙏 Acknowledgements

We extend our gratitude to the **National Stock Exchange (NSE)** for historical stock data.  
Special thanks to our academic advisor, **Dr. Sujata Kolhe ma’am**, for valuable insights into the stock market domain.

---

## 👥 Contributors

- 👨‍💼 **Yash Jahagirdar** [🔗 GitHub](https://github.com/Yash-Jahagirdar)  
- 👨‍💻 **Rohit Kandelkar** [🔗 GitHub](https://github.com/rohit-kandelkar)  
- 👩‍💼 **Rajeshwari Golande**

---

### 📌 Note:

- 🗂️ Update the CSV files of each stock daily for accurate results.  
- 💻 This project can be run on **Visual Studio Code** or any Python-supported IDE.
