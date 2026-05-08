# 🌍 Air Quality Prediction using Machine Learning

## 📌 Project Overview

This project predicts **Air Quality Levels** using Machine Learning algorithms such as:

- 🌲 Decision Tree Classifier
- 🌳 Random Forest Classifier

The system analyzes environmental and pollution-related parameters like temperature, humidity, PM2.5, PM10, NO₂, CO, and AQI to classify air quality into categories such as:

- Good
- Moderate
- Unhealthy for Sensitive Groups
- Unhealthy
- Hazardous

---

# 🎯 Objectives

- Load and analyze air quality datasets
- Perform data preprocessing
- Train Machine Learning models
- Evaluate model performance
- Visualize important insights
- Predict air quality for new samples

---

# 🧠 Machine Learning Models Used

## 1️⃣ Decision Tree Classifier
A supervised learning algorithm that creates decision rules based on dataset features.

## 2️⃣ Random Forest Classifier
An ensemble learning algorithm that combines multiple decision trees for improved accuracy and reduced overfitting.

---

# 📂 Dataset Features

| Feature Name | Description |
|---|---|
| temperature_c | Temperature in Celsius |
| humidity_pct | Relative Humidity (%) |
| wind_speed_kmh | Wind Speed (km/h) |
| pm25 | Fine Particulate Matter |
| pm10 | Coarse Particulate Matter |
| no2 | Nitrogen Dioxide |
| co | Carbon Monoxide |
| aqi | Air Quality Index |
| city | City Name |

### 🎯 Target Column
`air_quality_label`

Possible labels:
- Good
- Moderate
- Unhealthy for Sensitive
- Unhealthy
- Hazardous

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 📦 Required Libraries

Install all dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
