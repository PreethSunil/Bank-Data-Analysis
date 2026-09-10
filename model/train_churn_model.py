import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os

GOLD = '#f59e0b'
BLUE = '#3b82f6'
GREEN = '#10b981'
RED = '#ef4444'
BG = '#0f172a'

plt.rcParams.update({
    'axes.facecolor': BG,
    'figure.facecolor': BG,
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': (1,1,1,0.2)
})

os.makedirs('model', exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('dataset/BankChurn.csv')

# Preprocessing
X = df.drop(['CustomerID', 'Exited'], axis=1)
y = df['Exited']

X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Training model...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'f1': float(f1_score(y_test, y_pred))
}

print("Saving model and metrics...")
joblib.dump(rf, 'model/churn_model.pkl')
with open('model/churn_metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Generating charts...")
# Chart 1: Churn Distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = y.value_counts()
ax.pie(counts, labels=['Retained (0)', 'Churned (1)'], autopct='%1.1f%%', colors=[BLUE, RED],
       wedgeprops={'edgecolor': 'white'})
ax.set_title("Customer Churn Distribution", color='white')
fig.savefig('static/charts/churn_distribution.png', bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=True)
plt.close(fig)

# Chart 2: Feature Importance
fig, ax = plt.subplots(figsize=(8, 6))
importances = rf.feature_importances_
indices = np.argsort(importances)
features = X.columns

ax.barh(range(len(indices)), importances[indices], color=GOLD)
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([features[i] for i in indices])
ax.set_xlabel("Importance")
ax.set_title("Feature Importance in Churn Prediction")
fig.savefig('static/charts/churn_feature_importance.png', bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=True)
plt.close(fig)

print("Churn model training complete.")
