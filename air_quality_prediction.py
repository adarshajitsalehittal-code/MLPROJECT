"""
╔══════════════════════════════════════════════════════════════╗
║         AIR QUALITY PREDICTION - Beginner ML Project        ║
║         Using Random Forest & Decision Tree Classifiers     ║
╚══════════════════════════════════════════════════════════════╝

WHAT YOU WILL LEARN:
  ✅ How to load and explore a dataset
  ✅ How to preprocess data (handle missing values, encode labels)
  ✅ How to train ML models (Decision Tree + Random Forest)
  ✅ How to evaluate model performance
  ✅ How to visualize results
  ✅ How to make predictions on new data

DATASET FEATURES:
  - temperature_c     : Temperature in Celsius
  - humidity_pct      : Relative humidity (%)
  - wind_speed_kmh    : Wind speed (km/h)
  - pm25              : Fine particulate matter (µg/m³)
  - pm10              : Coarse particulate matter (µg/m³)
  - no2               : Nitrogen dioxide (µg/m³)
  - co                : Carbon monoxide (mg/m³)
  - aqi               : Air Quality Index (numeric)

TARGET (what we predict):
  - air_quality_label : Good / Moderate / Unhealthy for Sensitive /
                        Unhealthy / Hazardous

INSTALL REQUIREMENTS (run once):
  pip install pandas numpy scikit-learn matplotlib seaborn
"""

# ─────────────────────────────────────────────
# STEP 1: IMPORT LIBRARIES
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  AIR QUALITY PREDICTION — Beginner ML Project")
print("=" * 60)


# ─────────────────────────────────────────────
# STEP 2: LOAD THE DATASET
# ─────────────────────────────────────────────
print("\n📂 STEP 2: Loading Dataset...")

# Change this path if needed
df = pd.read_csv("air_quality_dataset.csv")

print(f"   ✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
print("\n📋 First 5 rows:")
print(df.head())


# ─────────────────────────────────────────────
# STEP 3: EXPLORE THE DATA
# ─────────────────────────────────────────────
print("\n🔍 STEP 3: Exploring the Data...")

print("\n📊 Dataset Info:")
print(df.dtypes)

print("\n📈 Basic Statistics:")
print(df.describe().round(2))

print("\n🏷️  Air Quality Label Distribution:")
label_counts = df['air_quality_label'].value_counts()
print(label_counts)

print("\n🌆 City Distribution:")
print(df['city'].value_counts())

# Check for missing values
print("\n❓ Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✅ No missing values found!")
else:
    print(missing[missing > 0])


# ─────────────────────────────────────────────
# STEP 4: DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n🔧 STEP 4: Preprocessing Data...")

# Select features (inputs) for prediction
FEATURE_COLUMNS = [
    'temperature_c',
    'humidity_pct',
    'wind_speed_kmh',
    'pm25',
    'pm10',
    'no2',
    'co',
    'aqi'
]

TARGET_COLUMN = 'air_quality_label'

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

print(f"   Features: {FEATURE_COLUMNS}")
print(f"   Target  : {TARGET_COLUMN}")
print(f"   X shape : {X.shape}")
print(f"   y shape : {y.shape}")

# Encode labels to numbers (ML models need numbers, not text)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n   Label Encoding:")
for i, label in enumerate(label_encoder.classes_):
    print(f"     {i} → {label}")


# ─────────────────────────────────────────────
# STEP 5: SPLIT DATA INTO TRAIN & TEST SETS
# ─────────────────────────────────────────────
print("\n✂️  STEP 5: Splitting into Train & Test Sets...")

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded  # keeps class balance
)

print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")


# ─────────────────────────────────────────────
# STEP 6: TRAIN MODEL 1 — DECISION TREE
# ─────────────────────────────────────────────
print("\n🌲 STEP 6: Training Decision Tree Classifier...")

dt_model = DecisionTreeClassifier(
    max_depth=5,        # limit tree depth to prevent overfitting
    random_state=42
)
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print(f"   ✅ Decision Tree Accuracy: {dt_accuracy:.2%}")


# ─────────────────────────────────────────────
# STEP 7: TRAIN MODEL 2 — RANDOM FOREST
# ─────────────────────────────────────────────
print("\n🌳 STEP 7: Training Random Forest Classifier...")

rf_model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    max_depth=10,
    random_state=42,
    n_jobs=-1           # use all CPU cores
)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"   ✅ Random Forest Accuracy: {rf_accuracy:.2%}")


# ─────────────────────────────────────────────
# STEP 8: EVALUATE MODELS
# ─────────────────────────────────────────────
print("\n📊 STEP 8: Evaluating Models...")

print("\n" + "─" * 50)
print("DECISION TREE — Classification Report")
print("─" * 50)
print(classification_report(
    y_test, dt_predictions,
    target_names=label_encoder.classes_
))

print("─" * 50)
print("RANDOM FOREST — Classification Report")
print("─" * 50)
print(classification_report(
    y_test, rf_predictions,
    target_names=label_encoder.classes_
))

print(f"\n{'─'*50}")
print(f"  MODEL COMPARISON SUMMARY")
print(f"{'─'*50}")
print(f"  Decision Tree  Accuracy: {dt_accuracy:.2%}")
print(f"  Random Forest  Accuracy: {rf_accuracy:.2%}")
winner = "Random Forest" if rf_accuracy >= dt_accuracy else "Decision Tree"
print(f"  🏆 Winner: {winner}")
print(f"{'─'*50}")


# ─────────────────────────────────────────────
# STEP 9: FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n🔑 STEP 9: Feature Importance (Random Forest)...")

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': FEATURE_COLUMNS,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n   Feature Importance Ranking:")
for _, row in feature_importance_df.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"   {row['Feature']:18s} {bar} {row['Importance']:.4f}")


# ─────────────────────────────────────────────
# STEP 10: VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n📊 STEP 10: Generating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Air Quality Prediction — ML Analysis", fontsize=16, fontweight='bold')

# Plot 1: Label distribution
ax1 = axes[0, 0]
colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']
label_counts.plot(kind='bar', ax=ax1, color=colors[:len(label_counts)], edgecolor='white')
ax1.set_title("Air Quality Label Distribution", fontweight='bold')
ax1.set_xlabel("Label")
ax1.set_ylabel("Count")
ax1.tick_params(axis='x', rotation=30)

# Plot 2: Feature importance
ax2 = axes[0, 1]
feature_importance_df.plot(
    kind='barh', x='Feature', y='Importance',
    ax=ax2, color='#3498db', edgecolor='white', legend=False
)
ax2.set_title("Feature Importance (Random Forest)", fontweight='bold')
ax2.set_xlabel("Importance Score")

# Plot 3: Confusion Matrix - Random Forest
ax3 = axes[0, 2]
cm = confusion_matrix(y_test, rf_predictions)
sns.heatmap(
    cm, annot=True, fmt='d', ax=ax3,
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
ax3.set_title("Confusion Matrix — Random Forest", fontweight='bold')
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
ax3.tick_params(axis='x', rotation=30)
ax3.tick_params(axis='y', rotation=0)

# Plot 4: PM2.5 vs AQI scatter
ax4 = axes[1, 0]
colors_map = {
    'Good': '#2ecc71',
    'Moderate': '#f39c12',
    'Unhealthy for Sensitive': '#e67e22',
    'Unhealthy': '#e74c3c',
    'Hazardous': '#8e44ad'
}
for label, color in colors_map.items():
    mask = df['air_quality_label'] == label
    ax4.scatter(
        df.loc[mask, 'pm25'],
        df.loc[mask, 'aqi'],
        label=label, color=color, alpha=0.7, s=60
    )
ax4.set_title("PM2.5 vs AQI by Label", fontweight='bold')
ax4.set_xlabel("PM2.5 (µg/m³)")
ax4.set_ylabel("AQI")
ax4.legend(fontsize=7)

# Plot 5: Temperature vs AQI by city
ax5 = axes[1, 1]
cities = df['city'].unique()
city_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
for i, city in enumerate(cities):
    mask = df['city'] == city
    ax5.scatter(
        df.loc[mask, 'temperature_c'],
        df.loc[mask, 'aqi'],
        label=city, color=city_colors[i], alpha=0.7, s=60
    )
ax5.set_title("Temperature vs AQI by City", fontweight='bold')
ax5.set_xlabel("Temperature (°C)")
ax5.set_ylabel("AQI")
ax5.legend(fontsize=8)

# Plot 6: Model accuracy comparison
ax6 = axes[1, 2]
models = ['Decision Tree', 'Random Forest']
accuracies = [dt_accuracy * 100, rf_accuracy * 100]
bars = ax6.bar(models, accuracies, color=['#e74c3c', '#3498db'], width=0.5, edgecolor='white')
ax6.set_title("Model Accuracy Comparison", fontweight='bold')
ax6.set_ylabel("Accuracy (%)")
ax6.set_ylim(0, 110)
for bar, acc in zip(bars, accuracies):
    ax6.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{acc:.1f}%", ha='center', va='bottom', fontweight='bold'
    )

plt.tight_layout()
plt.savefig("air_quality_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Saved visualization to air_quality_results.png")


# ─────────────────────────────────────────────
# STEP 11: MAKE PREDICTIONS ON NEW DATA
# ─────────────────────────────────────────────
print("\n🔮 STEP 11: Making Predictions on New Samples...")

# New unseen samples
new_samples = pd.DataFrame({
    'temperature_c':  [12.0, 32.0, 25.0],
    'humidity_pct':   [80.0, 45.0, 60.0],
    'wind_speed_kmh': [5.0,  22.0, 15.0],
    'pm25':           [200.0, 15.0, 55.0],
    'pm10':           [245.0, 32.0, 90.0],
    'no2':            [95.0,  8.0,  30.0],
    'co':             [2.4,   0.2,  0.9],
    'aqi':            [255.0, 38.0, 100.0]
})

sample_labels = ["Winter Delhi (cold, stagnant)", "Summer Bangalore (hot, windy)", "Autumn Mumbai"]

rf_preds = rf_model.predict(new_samples)
rf_proba = rf_model.predict_proba(new_samples)

print(f"\n{'─'*60}")
print(f"  {'Sample':<30} {'Predicted Label'}")
print(f"{'─'*60}")
for i, (idx, sample) in enumerate(new_samples.iterrows()):
    pred_label = label_encoder.inverse_transform([rf_preds[i]])[0]
    confidence = rf_proba[i].max() * 100
    print(f"  {sample_labels[i]:<30} {pred_label}  ({confidence:.0f}% confident)")
print(f"{'─'*60}")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print(f"""
{'='*60}
  🎉 PROJECT COMPLETE! 

  What you built:
  ✅ Loaded & explored a real-world style dataset
  ✅ Preprocessed features and encoded labels
  ✅ Trained a Decision Tree Classifier
  ✅ Trained a Random Forest Classifier
  ✅ Evaluated both with accuracy + classification report
  ✅ Visualized feature importance & confusion matrix
  ✅ Made predictions on brand new data

  Best Model   : {winner}
  Best Accuracy: {max(dt_accuracy, rf_accuracy):.2%}

  Next Steps to Explore:
  → Try other models: XGBoost, SVM, KNN
  → Add more features: population density, traffic
  → Try regression to predict exact AQI value
  → Build a simple web app with Flask or Streamlit
{'='*60}
""")
