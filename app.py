import os
import json
import joblib
import numpy as np
import pandas as pd
from functools import wraps
from flask import Flask, render_template, request, jsonify, Response, send_from_directory, session, redirect, url_for

app = Flask(__name__)
app.secret_key = 'universal-bank-ml-portal-2026'

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'UniversalBank.csv')
MODEL_DIR    = os.path.join(BASE_DIR, 'model')
CHARTS_DIR   = os.path.join(BASE_DIR, 'static', 'charts')
BEST_MODEL   = os.path.join(MODEL_DIR, 'best_model.pkl')
SCALER_PATH  = os.path.join(MODEL_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'model_metrics.json')

# In-memory cache
_df      = None
_model   = None
_scaler  = None
_metrics = None

CHURN_DATASET = os.path.join(BASE_DIR, 'dataset', 'BankChurn.csv')
CHURN_MODEL   = os.path.join(MODEL_DIR, 'churn_model.pkl')
CHURN_METRICS = os.path.join(MODEL_DIR, 'churn_metrics.json')

_churn_df = None
_churn_model = None
_churn_metrics = None

def get_churn_df():
    global _churn_df
    if _churn_df is None:
        _churn_df = pd.read_csv(CHURN_DATASET)
    return _churn_df.copy()

def get_df():
    global _df
    if _df is None:
        _df = pd.read_csv(DATASET_PATH)
    return _df.copy()

def get_artifacts():
    global _model, _scaler, _metrics
    if _model is None and os.path.exists(BEST_MODEL):
        _model = joblib.load(BEST_MODEL)
    if _scaler is None and os.path.exists(SCALER_PATH):
        _scaler = joblib.load(SCALER_PATH)
    if _metrics is None and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    return _model, _scaler, _metrics

def get_churn_artifacts():
    global _churn_model, _churn_metrics
    if _churn_model is None and os.path.exists(CHURN_MODEL):
        _churn_model = joblib.load(CHURN_MODEL)
    if _churn_metrics is None and os.path.exists(CHURN_METRICS):
        with open(CHURN_METRICS) as f:
            _churn_metrics = json.load(f)
    return _churn_model, _churn_metrics

# ── Page routes ──────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('username') == 'admin' and request.form.get('password') == 'admin123':
            session['logged_in'] = True
            session['username'] = 'admin'
            return redirect(url_for('dashboard_page'))
        else:
            error = 'Invalid credentials'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard_page():
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/analysis')
@login_required
def analysis_page():
    return render_template('analysis.html', active_page='analysis')

@app.route('/customers')
@login_required
def customers_page():
    return render_template('customer_table.html', active_page='customers')

@app.route('/churn')
@login_required
def churn_page():
    return render_template('churn.html', active_page='churn')

@app.route('/churn-prediction')
@login_required
def churn_prediction_page():
    return render_template('churn_prediction.html', active_page='churn_prediction')

@app.route('/models')
@login_required
def models_page():
    return render_template('models.html', active_page='models')

@app.route('/prediction')
@login_required
def prediction_page():
    return render_template('prediction.html', active_page='prediction')

@app.route('/insights')
@login_required
def insights_page():
    return render_template('insights.html', active_page='insights')

@app.route('/about')
@login_required
def about_page():
    return render_template('about.html', active_page='about')

# ── Serve matplotlib chart images ────────────────────────────────────────────
@app.route('/charts/<path:filename>')
def serve_chart(filename):
    return send_from_directory(CHARTS_DIR, filename)

# ── REST APIs ─────────────────────────────────────────────────────────────────
@app.route('/api/dashboard-kpis')
@login_required
def api_dashboard_kpis():
    df = get_df()
    n  = len(df)
    return jsonify({
        'total_customers'    : int(n),
        'avg_age'            : round(float(df['Age'].mean()), 1),
        'avg_income'         : round(float(df['Income'].mean()), 1),
        'avg_mortgage'       : round(float(df['Mortgage'].mean()), 1),
        'loan_acceptance_rate': round(float(df['Personal Loan'].mean() * 100), 2),
        'online_banking_pct' : round(float(df['Online'].mean() * 100), 2),
        'credit_card_pct'    : round(float(df['CreditCard'].mean() * 100), 2),
    })

@app.route('/api/analysis-data')
@login_required
def api_analysis_data():
    df = get_df()

    # Apply optional filters
    min_age     = request.args.get('min_age',     type=int)
    max_age     = request.args.get('max_age',     type=int)
    education   = request.args.get('education',   type=int)
    loan_status = request.args.get('loan_status', type=int)

    if min_age     is not None: df = df[df['Age'] >= min_age]
    if max_age     is not None: df = df[df['Age'] <= max_age]
    if education   in [1,2,3]:  df = df[df['Education'] == education]
    if loan_status in [0, 1]:   df = df[df['Personal Loan'] == loan_status]

    n = len(df)

    # Helpers using Pandas + NumPy
    def bin_series(col, bins, labels):
        return pd.cut(df[col], bins=bins, labels=labels, right=False)\
                 .value_counts().sort_index().to_dict()

    age_dist  = bin_series('Age',        [20,30,40,50,60,70], ['20-29','30-39','40-49','50-59','60+'])
    inc_dist  = bin_series('Income',     [0,50,100,150,200,250], ['$0-50k','$50k-100k','$100k-150k','$150k-200k','$200k+'])
    exp_dist  = bin_series('Experience', [0,10,20,30,40,50],  ['0-9 yrs','10-19 yrs','20-29 yrs','30-39 yrs','40+ yrs'])

    edu_map  = {1:'Undergraduate', 2:'Graduate', 3:'Advanced/Professional'}
    edu_dist = df['Education'].map(edu_map).value_counts().to_dict()
    fam_dist = {f"{k} Person{'s' if k>1 else ''}": int(v)
                for k, v in df['Family'].value_counts().sort_index().items()}

    mortgage_dist = {
        'No Mortgage ($0)': int((df['Mortgage'] == 0).sum()),
        'Has Mortgage'    : int((df['Mortgage'] >  0).sum()),
    }
    loan_dist = {
        'Accepted (1)': int((df['Personal Loan'] == 1).sum()),
        'Rejected (0)': int((df['Personal Loan'] == 0).sum()),
    }
    binary_summary = {
        'Online Banking'    : int(df['Online'].sum()),
        'Securities Account': int(df['Securities Account'].sum()),
        'Credit Card'       : int(df['CreditCard'].sum()),
        'CD Account'        : int(df['CD Account'].sum()),
    }

    # Correlation matrix — NumPy under the hood via Pandas
    num_df = df.drop(columns=[c for c in ['ID','ZIP Code'] if c in df.columns])
    corr   = num_df.corr().round(2)
    corr_feats  = list(corr.columns)
    corr_matrix = corr.values.tolist()   # NumPy array → Python list

    # Scatter sample
    scatter = df.sample(n=min(600, n), random_state=42)[
        ['Income','CCAvg','Personal Loan']
    ].to_dict(orient='records')

    # Education vs Loan acceptance rate
    edu_loan = df.groupby('Education')['Personal Loan'].agg(['count','sum'])
    edu_loan_rate = {
        edu_map.get(k, str(k)):
        round(float(row['sum'] / row['count'] * 100), 2) if row['count'] else 0
        for k, row in edu_loan.iterrows()
    }

    return jsonify({
        'age_dist'    : age_dist,
        'income_dist' : inc_dist,
        'exp_dist'    : exp_dist,
        'edu_dist'    : edu_dist,
        'family_dist' : fam_dist,
        'mortgage_dist': mortgage_dist,
        'loan_dist'   : loan_dist,
        'binary_summary': binary_summary,
        'correlation' : {'features': corr_feats, 'matrix': corr_matrix},
        'scatter_data': scatter,
        'edu_loan_rate': edu_loan_rate,
        'filtered_total': n,
    })

@app.route('/api/customer-data')
@login_required
def api_customer_data():
    df      = get_df()
    search  = request.args.get('search', '').strip()
    page    = request.args.get('page',    1,  type=int)
    limit   = request.args.get('limit',   15, type=int)
    sort_by = request.args.get('sort_by', 'ID')
    sort_dir= request.args.get('sort_dir','asc')

    if search:
        mask = (
            df['ID'].astype(str).str.contains(search, case=False) |
            df['Age'].astype(str).str.contains(search) |
            df['Income'].astype(str).str.contains(search)
        )
        df = df[mask]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_dir == 'asc'))

    total   = len(df)
    pages   = max(1, (total + limit - 1) // limit)
    page    = max(1, min(page, pages))
    records = df.iloc[(page-1)*limit : page*limit].to_dict(orient='records')

    return jsonify({'data': records, 'page': page, 'limit': limit,
                    'total_records': total, 'total_pages': pages})

@app.route('/api/export-csv')
@login_required
def api_export_csv():
    csv_data = get_df().to_csv(index=False)
    return Response(csv_data, mimetype='text/csv',
                    headers={'Content-Disposition':
                             'attachment; filename=UniversalBank_Data.csv'})

@app.route('/api/model-performance')
@login_required
def api_model_performance():
    _, _, metrics = get_artifacts()
    if metrics is None:
        from model.train_model import train_and_evaluate
        train_and_evaluate()
        global _metrics
        _metrics = None
        _, _, metrics = get_artifacts()
    return jsonify(metrics)

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    data   = request.json or {}
    model, scaler, _ = get_artifacts()
    if model is None:
        return jsonify({'error': 'Model not trained yet.'}), 500

    try:
        # Must match training feature order exactly:
        # Age, Experience, Income, Family, CCAvg, Education, Mortgage,
        # Securities Account, CD Account, Online, CreditCard
        row = np.array([[
            float(data.get('age',         40)),
            float(data.get('experience',  15)),
            float(data.get('income',      80)),
            float(data.get('family',       2)),
            float(data.get('cc_avg',     2.0)),
            float(data.get('education',    2)),
            float(data.get('mortgage',     0)),
            float(data.get('securities',   0)),
            float(data.get('cd_account',   0)),
            float(data.get('online',       1)),
            float(data.get('credit_card',  0)),
        ]])

        prob = float(model.predict_proba(row)[0][1])
        pred = int(model.predict(row)[0])
        conf = round((prob if pred == 1 else 1 - prob) * 100, 2)

        if   prob >= 0.70: risk = 'High Acceptance Likelihood';     rec = 'Target immediately with premium personal loan campaign offers.'
        elif prob >= 0.40: risk = 'Moderate Potential';             rec = 'Offer customised interest rates or promotional cashback incentives.'
        else:              risk = 'Low Conversion Likelihood';      rec = 'Focus marketing on higher probability customer segments.'

        return jsonify({
            'prediction'      : pred,
            'prediction_label': 'ACCEPTED' if pred == 1 else 'REJECTED',
            'probability'     : prob,
            'probability_pct' : round(prob * 100, 2),
            'confidence_pct'  : conf,
            'risk_level'      : risk,
            'recommendation'  : rec,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/insights')
@login_required
def api_insights():
    df = get_df()
    acc = df[df['Personal Loan'] == 1]
    rej = df[df['Personal Loan'] == 0]
    total = len(df)

    insights = [
        {'category': 'Income Impact',          'title': 'High Annual Income Driver',
         'description': f"Loan acceptors average ${acc['Income'].mean():.1f}k vs ${rej['Income'].mean():.1f}k for non-acceptors.",
         'impact': 'High Positive Impact'},
        {'category': 'Credit Card Spending',   'title': 'Monthly Spending Correlation',
         'description': f"Acceptors spend ${acc['CCAvg'].mean():.2f}k/mo vs ${rej['CCAvg'].mean():.2f}k/mo for non-acceptors.",
         'impact': 'High Positive Impact'},
        {'category': 'CD Account',             'title': 'CD Account Synergy',
         'description': f"{df[df['CD Account']==1]['Personal Loan'].mean()*100:.1f}% of CD Account holders accept loans — the strongest binary predictor.",
         'impact': 'Critical Indicator'},
        {'category': 'Education Level',        'title': 'Higher Education Conversion',
         'description': 'Graduate and Professional degree holders convert at nearly 2x the rate of undergraduates.',
         'impact': 'Moderate Positive Impact'},
        {'category': 'Digital Adoption',       'title': 'Online Banking Engagement',
         'description': f"Online banking usage is {df['Online'].mean()*100:.1f}%, providing a key digital acquisition channel.",
         'impact': 'Strategic Channel'},
    ]

    return jsonify({
        'highest_income_grp': 'Income > $100k shows 3x higher loan conversion.',
        'top_edu'            : 'Advanced/Professional degree holders have highest loan uptake.',
        'avg_mortgage_val'   : f"${df['Mortgage'].mean():.1f}k",
        'overall_acceptance' : f"{df['Personal Loan'].mean()*100:.2f}%",
        'online_adoption'    : f"{df['Online'].mean()*100:.2f}%",
        'cc_ownership'       : f"{df['CreditCard'].mean()*100:.2f}%",
        'insights'           : insights,
    })

@app.route('/api/churn-data')
@login_required
def api_churn_data():
    df = get_churn_df()
    n = len(df)
    churn_rate = round(float(df['Exited'].mean() * 100), 2)
    avg_credit = round(float(df['CreditScore'].mean()), 1)
    avg_balance = round(float(df['Balance'].mean()), 2)
    
    def get_stats(group_col):
        res = {}
        for name, group in df.groupby(group_col):
            total = len(group)
            churned = int(group['Exited'].sum())
            rate = round(float(churned / total * 100), 2) if total > 0 else 0
            res[str(name)] = {'total': total, 'churned': churned, 'rate': rate}
        return res
        
    geo = get_stats('Geography')
    gender = get_stats('Gender')
    products = get_stats('NumOfProducts')
    
    bins = [18, 30, 40, 50, 60, 150]
    labels = ['18-29', '30-39', '40-49', '50-59', '60+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    age_group = get_stats('AgeGroup')
    
    tenure_stats = get_stats('Tenure')
    tenure = {k: v['rate'] for k, v in tenure_stats.items()}
    
    active = get_stats('IsActiveMember')
    activity = {
        'active': active.get('1', {'total':0, 'churned':0, 'rate':0}),
        'inactive': active.get('0', {'total':0, 'churned':0, 'rate':0})
    }
    
    return jsonify({
        'total_customers': n,
        'churn_rate': churn_rate,
        'avg_credit_score': avg_credit,
        'avg_balance': avg_balance,
        'churn_by_geography': geo,
        'churn_by_gender': gender,
        'churn_by_age_group': age_group,
        'churn_by_products': products,
        'churn_by_tenure': tenure,
        'churn_by_activity': activity
    })

@app.route('/api/churn-model-performance')
@login_required
def api_churn_model_performance():
    global _churn_metrics
    if _churn_metrics is None and os.path.exists(CHURN_METRICS):
        with open(CHURN_METRICS) as f:
            _churn_metrics = json.load(f)
    return jsonify(_churn_metrics)

@app.route('/api/churn-predict', methods=['POST'])
@login_required
def api_churn_predict():
    data = request.json or {}
    model, _ = get_churn_artifacts()
    if model is None:
        return jsonify({'error': 'Churn model not trained yet.'}), 500
    try:
        geography = data.get('geography', 'France')
        gender = data.get('gender', 'Female')
        row = {
            'CreditScore': float(data.get('credit_score', 650)),
            'Age': float(data.get('age', 40)),
            'Tenure': float(data.get('tenure', 5)),
            'Balance': float(data.get('balance', 0)),
            'NumOfProducts': float(data.get('num_products', 1)),
            'HasCrCard': float(data.get('has_credit_card', 1)),
            'IsActiveMember': float(data.get('is_active_member', 1)),
            'EstimatedSalary': float(data.get('estimated_salary', 60000)),
            'Geography_Germany': int(geography == 'Germany'),
            'Geography_Spain': int(geography == 'Spain'),
            'Gender_Male': int(gender == 'Male'),
        }
        feature_order = list(getattr(model, 'feature_names_in_', row.keys()))
        features = pd.DataFrame([row]).reindex(columns=feature_order, fill_value=0)
        probability = float(model.predict_proba(features)[0][1])
        prediction = int(model.predict(features)[0])
        confidence = round((probability if prediction else 1 - probability) * 100, 2)
        if probability >= 0.65:
            risk_level, recommendation = 'High Churn Risk', 'Contact this customer promptly with a tailored retention offer and service review.'
        elif probability >= 0.35:
            risk_level, recommendation = 'Moderate Churn Risk', 'Monitor engagement and offer a relevant product or loyalty incentive.'
        else:
            risk_level, recommendation = 'Low Churn Risk', 'Maintain the relationship through regular digital engagement.'
        return jsonify({'prediction': prediction, 'prediction_label': 'LIKELY TO CHURN' if prediction else 'LIKELY TO STAY', 'probability': probability, 'probability_pct': round(probability * 100, 2), 'confidence_pct': confidence, 'risk_level': risk_level, 'recommendation': recommendation})
    except (TypeError, ValueError) as error:
        return jsonify({'error': f'Invalid customer input: {error}'}), 400


if __name__ == '__main__':
    # Auto-train if model missing
    if not os.path.exists(BEST_MODEL):
        from model.train_model import train_and_evaluate
        train_and_evaluate()

    print("\n" + "="*70)
    print("[SERVER] Bank Customer Data Analysis & ML Web Portal is online!")
    print("[URL]    Open your browser at: http://127.0.0.1:5000")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
