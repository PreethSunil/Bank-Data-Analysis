import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 5000

# Features
customer_id = np.arange(1, n + 1)
credit_score = np.clip(np.random.normal(650, 50, n).astype(int), 300, 850)
geography = np.random.choice(['France', 'Germany', 'Spain'], n, p=[0.5, 0.25, 0.25])
gender = np.random.choice(['Male', 'Female'], n)

# Age: skewed towards 30-50
age = np.clip(np.random.normal(40, 10, n).astype(int), 18, 92)

tenure = np.random.randint(0, 11, n)

# Balance: many zeros
balance = np.random.uniform(10000, 250000, n)
zero_balance_mask = np.random.choice([True, False], n, p=[0.3, 0.7])
balance[zero_balance_mask] = 0

num_products = np.random.choice([1, 2, 3, 4], n, p=[0.5, 0.45, 0.04, 0.01])
has_crcard = np.random.choice([0, 1], n, p=[0.3, 0.7])
is_active_member = np.random.choice([0, 1], n, p=[0.5, 0.5])
estimated_salary = np.random.uniform(10000, 200000, n)

# Compute churn probability based on some logical rules
prob = np.zeros(n)
# Germany has higher churn
prob += np.where(geography == 'Germany', 0.15, 0.0)
# Older age has higher churn (up to a point, say 60)
prob += np.where((age > 40) & (age < 65), 0.1, 0.0)
# Low activity
prob += np.where(is_active_member == 0, 0.1, -0.1)
# Number of products
prob += np.where(num_products == 1, 0.05, 0.0)
prob += np.where(num_products > 2, 0.2, 0.0)
# High balance
prob += np.where(balance > 100000, 0.05, 0.0)
prob += np.where(balance == 0, -0.05, 0.0)

prob = prob + 0.05 # Base rate
prob = np.clip(prob, 0, 1)

exited = np.random.binomial(1, prob)

df = pd.DataFrame({
    'CustomerID': customer_id,
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_crcard,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
    'Exited': exited
})

os.makedirs('dataset', exist_ok=True)
df.to_csv('dataset/BankChurn.csv', index=False)
print("dataset/BankChurn.csv generated successfully.")
