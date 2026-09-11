# Bank Customer Data Analysis & Personal Loan ML Web Application

A professional, full-stack Machine Learning web application designed for analyzing banking customer data, performing exploratory data analysis (EDA), training and comparing multiple ML classification algorithms, and predicting customer personal loan acceptance in real-time.

Built specifically as an academic final-year / B.Tech Machine Learning project demonstration and viva presentation using the **Universal Bank Personal Loan Modelling** dataset from Kaggle.

---

## 🌟 Key Features

### 1. Executive Banking Dashboard
- **7 Live KPI Cards**: Total Customers, Average Age, Average Income, Average Mortgage, Loan Acceptance Rate, Online Banking %, Credit Card %.
- **Interactive Visualizations**: Personal loan acceptance doughnut chart, income distribution histogram.
- **Quick Evaluator Widget**: Instant live loan prediction teaser powered by Random Forest.
- **Top Model Snapshot**: Highlights top accuracy, precision, recall, and F1 metrics.

### 2. Exploratory Data Analysis (EDA)
- **14 Interactive Chart.js Visualizations**:
  - Age Distribution
  - Income Distribution
  - Work Experience Distribution
  - Education Breakdown (Undergrad, Graduate, Professional)
  - Family Size Distribution
  - Mortgage Status ($0 vs Active Mortgage)
  - Loan Accepted vs Rejected Ratio
  - Online Banking Users
  - Securities Account Holders
  - Credit Card Users
  - CD Account Holders
  - Education Level vs Loan Acceptance Rate (%)
  - Income vs Credit Card Spending (CCAvg) Scatter Plot
  - Full Feature Correlation Matrix Heatmap
- **Global Dynamic Filters**: Filter visualizations by Age range, Education level, and Personal Loan status in real-time.

### 3. Machine Learning Performance Hub
- **4 Algorithms Trained & Compared**:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest (Top Performing Model)
  4. K-Nearest Neighbors (KNN)
- **Evaluation Metrics Displayed**: Accuracy, Precision, Recall, F1 Score.
- **Interactive Confusion Matrix**: Switch between algorithms to view True Negatives, False Positives, False Negatives, and True Positives.
- **Feature Importance Ranking**: Ranks key predictors influencing customer loan acceptance.

### 4. Live Personal Loan Predictor
- Form input for 11 financial and demographic customer variables.
- Real-time Flask ML inference backend using pre-trained `best_model.pkl` and `scaler.pkl`.
- Probability gauge, confidence rating %, customer persona risk classification, and tailored banking strategy recommendation.

### 5. Data Records Grid
- Searchable by Customer ID, Age, and Income.
- Column sorting & pagination controls.
- Export capabilities to **CSV** and **PDF/Print**.

### 6. Additional UX Features
- Dark & Light mode toggle with persistent local storage settings.
- Glassmorphism dark banking UI with blue and gold accents.
- Toast notifications and loading overlays.
- Fully responsive across Desktop, Tablet, and Mobile screens.

---

## 📁 Folder Structure

```
Bank-Customer-Data-Analysis/
├── app.py                      # Main Flask Web Application & REST APIs
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
├── dataset/
│   ├── generate_dataset.py     # UniversalBank.csv Dataset Generator
│   └── UniversalBank.csv       # 5,000-row Universal Bank Dataset
├── model/
│   ├── train_model.py          # Data cleaning, 80/20 train split & ML training
│   ├── scaler.pkl              # Feature StandardScaler
│   ├── random_forest.pkl       # Trained Random Forest Model
│   ├── best_model.pkl          # Saved Best Performing ML Classifier
│   └── model_metrics.json      # Pre-calculated Model Evaluation Metrics
├── templates/
│   ├── layout.html             # Base Layout (Header, Sidebar, Toast, Clock)
│   ├── dashboard.html          # Executive Dashboard View
│   ├── analysis.html           # EDA Page (14 Interactive Charts & Filters)
│   ├── customer_table.html     # Searchable & Paginated Data Table
│   ├── models.html             # ML Model Comparison & Confusion Matrix
│   ├── prediction.html         # Live Prediction Form & Confidence Gauge
│   ├── insights.html           # Automated Business Intelligence Insights
│   └── about.html              # Academic Presentation & Tech Overview
└── static/
    ├── css/
    │   └── style.css           # Glassmorphism Dark Theme Styling
    └── js/
        └── script.js          # ES6 Dynamic Rendering & API Handler
```

---

## 🚀 How to Run the Application

### 1. Requirements
Ensure Python 3.8+ is installed on your machine.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask Web Application
```bash
python app.py
```

### 4. Access in Web Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔬 Machine Learning Pipeline Details

1. **Dataset**: 5,000 observations, 14 initial features.
2. **Preprocessing & Cleaning**:
   - Removed duplicate records.
   - Dropped non-predictive columns (`ID` and `ZIP Code`).
3. **Train/Test Split**: 80% training set (4,000 samples) and 20% test set (1,000 samples) with stratified target splitting.
4. **Feature Scaling**: Applied `StandardScaler` to normalize feature distributions for distance-based and parametric algorithms.
5. **Model Evaluation & Selection**: Models evaluated using Accuracy, Precision, Recall, and F1-Score; Random Forest achieved top performance and is serialized with `Joblib`.

---

## 🎓 Academic Viva & Presentation Highlights

- **Problem Statement**: Universal Bank seeks to convert credit card customers into personal loan customers while minimizing campaign acquisition cost.
- **Key Insight**: Annual Income (> $100k), Monthly Credit Card Spending (CCAvg), and CD Account ownership are the top 3 features driving loan acceptance.
- **Model Choice**: Random Forest was chosen for deployment due to high resistance to overfitting and robust handling of non-linear financial interactions.
