import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'UniversalBank.csv')
    model_dir   = os.path.join(base_dir, 'model')
    charts_dir  = os.path.join(base_dir, 'static', 'charts')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # ── 1. Load dataset with Pandas ───────────────────────────────────────────
    print("1. Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"   Raw shape: {df.shape}")

    # ── 2. Data Cleaning ──────────────────────────────────────────────────────
    print("2. Cleaning data...")
    df = df.drop_duplicates()
    df = df.dropna()
    drop_cols = [c for c in ['ID', 'ZIP Code'] if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"   Clean shape: {df.shape} | Features: {list(df.columns)}")

    target = 'Personal Loan'
    X = df.drop(columns=[target])
    y = df[target]
    feature_names = list(X.columns)

    # ── 3. Train / Test Split (80 / 20) with NumPy random state ──────────────
    print("3. Train/Test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # Standard scaling using Pandas/NumPy values
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train.values)
    X_test_arr  = scaler.transform(X_test.values)

    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"   Saved scaler -> {scaler_path}")

    # -- 4. Train Random Forest ------------------------------------------------
    print("4. Training Random Forest Classifier (n_estimators=100)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train.values, y_train.values)          # tree models use raw values

    y_pred = rf.predict(X_test.values)
    y_prob = rf.predict_proba(X_test.values)[:, 1]

    acc   = float(accuracy_score(y_test, y_pred))
    prec  = float(precision_score(y_test, y_pred, zero_division=0))
    rec   = float(recall_score(y_test, y_pred, zero_division=0))
    f1    = float(f1_score(y_test, y_pred, zero_division=0))
    cm    = confusion_matrix(y_test, y_pred)

    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   Precision: {prec*100:.2f}%")
    print(f"   Recall   : {rec*100:.2f}%")
    print(f"   F1 Score : {f1*100:.2f}%")

    # ── 5. Feature Importance (NumPy array) ───────────────────────────────────
    importances = rf.feature_importances_                    # NumPy array
    sorted_idx  = np.argsort(importances)[::-1]
    feat_imp = [
        {'feature': feature_names[i], 'importance': round(float(importances[i]), 4)}
        for i in sorted_idx
    ]

    # ── 6. Matplotlib Charts ──────────────────────────────────────────────────
    plt.style.use('dark_background')
    GOLD   = '#f59e0b'
    BLUE   = '#3b82f6'
    GREEN  = '#10b981'
    RED    = '#ef4444'
    BG     = '#0f172a'

    def save_fig(name):
        path = os.path.join(charts_dir, name)
        plt.savefig(path, dpi=100, bbox_inches='tight',
                    facecolor=BG, edgecolor='none')
        plt.close()
        print(f"   Saved chart -> {path}")

    # 6a. Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    labels = [['True Neg\n(TN)', 'False Pos\n(FP)'],
              ['False Neg\n(FN)', 'True Pos\n(TP)']]
    colors = [[BLUE, RED], ['#f97316', GREEN]]
    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle((c, r), 1, 1,
                         facecolor=mcolors.to_rgba(colors[r][c], 0.25),
                         edgecolor=colors[r][c], lw=1.5))
            ax.text(c+0.5, r+0.65, str(cm[r, c]),
                    ha='center', va='center', fontsize=26,
                    fontweight='bold', color='white')
            ax.text(c+0.5, r+0.25, labels[r][c],
                    ha='center', va='center', fontsize=9,
                    color='#94a3b8')
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Predicted 0 (No)', 'Predicted 1 (Yes)'],
                        color='#94a3b8', fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Actual 0 (No)', 'Actual 1 (Yes)'],
                        color='#94a3b8', fontsize=10)
    ax.set_title('Random Forest — Confusion Matrix', color=GOLD,
                 fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(length=0)
    save_fig('confusion_matrix.png')

    # 6b. Feature Importance bar chart
    n = len(feat_imp)
    fig, ax = plt.subplots(figsize=(7, max(4, n*0.45)), facecolor=BG)
    ax.set_facecolor(BG)
    names  = [f['feature'] for f in feat_imp]
    values = [f['importance']*100 for f in feat_imp]
    bars = ax.barh(names[::-1], values[::-1],
                   color=GOLD, alpha=0.85, height=0.6)
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}%', va='center', color='white', fontsize=9)
    ax.set_xlabel('Importance (%)', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.set_title('Feature Importance - Random Forest', color=GOLD,
                 fontsize=13, fontweight='bold', pad=10)
    ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color=(1,1,1,0.06), linestyle='--')
    save_fig('feature_importance.png')

    # 6c. Loan Acceptance Distribution (Pandas value_counts -> NumPy)
    counts = df[target].value_counts().sort_index().values  # NumPy array
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    wedges, _, autotexts = ax.pie(
        counts,
        labels=['Rejected (0)', 'Accepted (1)'],
        colors=[BLUE, GOLD],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': BG, 'linewidth': 2},
        pctdistance=0.75
    )
    for t in autotexts:
        t.set_color('white'); t.set_fontsize(11)
    ax.set_title('Personal Loan Acceptance Distribution', color=GOLD,
                 fontsize=12, fontweight='bold')
    centre = plt.Circle((0,0), 0.55, fc=BG)
    ax.add_artist(centre)
    save_fig('loan_distribution.png')

    # 6d. Income Distribution histogram using NumPy bins
    income_arr = df['Income'].values          # NumPy array
    bins = np.linspace(income_arr.min(), income_arr.max(), 25)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.hist(income_arr, bins=bins, color=BLUE, alpha=0.8,
            edgecolor=BG, linewidth=0.5)
    ax.set_xlabel('Annual Income ($000)', color='#94a3b8')
    ax.set_ylabel('Number of Customers', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.set_title('Income Distribution', color=GOLD,
                 fontsize=13, fontweight='bold', pad=10)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color=(1,1,1,0.06), linestyle='--')
    save_fig('income_distribution.png')

    # -- 7. Save model & metrics JSON ------------------------------------------
    model_path = os.path.join(model_dir, 'best_model.pkl')
    rf_path    = os.path.join(model_dir, 'random_forest.pkl')
    joblib.dump(rf, model_path)
    joblib.dump(rf, rf_path)
    print(f"   Saved model -> {model_path}")

    summary = {
        'best_model': 'Random Forest',
        'feature_names': feature_names,
        'metrics': {
            'Random Forest': {
                'accuracy'    : round(acc,  4),
                'precision'   : round(prec, 4),
                'recall'      : round(rec,  4),
                'f1_score'    : round(f1,   4),
                'accuracy_pct': round(acc*100,  2),
                'f1_pct'      : round(f1*100,   2),
                'confusion_matrix': cm.tolist()
            }
        },
        'feature_importance': feat_imp,
        'dataset_summary': {
            'total_records' : len(df),
            'total_features': len(feature_names),
            'train_records' : len(X_train),
            'test_records'  : len(X_test),
            'target_distribution': {
                '0': int((y == 0).sum()),
                '1': int((y == 1).sum())
            }
        }
    }
    metrics_path = os.path.join(model_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved metrics -> {metrics_path}")
    print("\nTraining complete!")

if __name__ == '__main__':
    train_and_evaluate()
