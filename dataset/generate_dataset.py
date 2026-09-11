import os
import numpy as np
import pandas as pd

def generate_universal_bank_data(filename, num_records=5000, seed=42):
    np.random.seed(seed)
    
    # 1. ID
    ids = np.arange(1, num_records + 1)
    
    # 2. Age (23 to 67, mean ~45)
    age = np.random.randint(23, 68, size=num_records)
    
    # 3. Experience (typically Age - 20 - random offset, range 0 to 43)
    exp = age - 20 - np.random.randint(0, 5, size=num_records)
    exp = np.clip(exp, 0, 43)
    
    # 4. Income (in $000s, range 8 to 224, mean ~73)
    income = np.random.lognormal(mean=4.15, sigma=0.6, size=num_records)
    income = np.clip(income, 8, 224).round().astype(int)
    
    # 5. ZIP Code (5-digit integers around California zip codes e.g. 90001 - 96651)
    zip_code = np.random.choice([90001, 91107, 92007, 93023, 94102, 94305, 95112, 95616, 96001], size=num_records)
    
    # 6. Family (1 to 4, discrete)
    family = np.random.choice([1, 2, 3, 4], size=num_records, p=[0.29, 0.26, 0.20, 0.25])
    
    # 7. CCAvg (Monthly credit card spending in $000s, correlated with income)
    cc_avg = (income / 35.0) + np.random.exponential(scale=0.8, size=num_records)
    cc_avg = np.clip(cc_avg, 0.0, 10.0).round(2)
    
    # 8. Education (1: Undergrad, 2: Graduate, 3: Advanced/Professional)
    education = np.random.choice([1, 2, 3], size=num_records, p=[0.42, 0.28, 0.30])
    
    # 9. Mortgage (Value of house mortgage in $000s, ~70% zero, max ~635)
    has_mortgage = np.random.choice([0, 1], size=num_records, p=[0.70, 0.30])
    mortgage_val = np.random.randint(70, 635, size=num_records)
    mortgage = has_mortgage * mortgage_val
    
    # 10. Securities Account (0/1, ~10% yes)
    securities = np.random.choice([0, 1], size=num_records, p=[0.90, 0.10])
    
    # 11. CD Account (0/1, ~6% yes)
    cd_account = np.random.choice([0, 1], size=num_records, p=[0.94, 0.06])
    
    # 12. Online (0/1, ~60% yes)
    online = np.random.choice([0, 1], size=num_records, p=[0.40, 0.60])
    
    # 13. CreditCard (0/1, ~29% yes)
    credit_card = np.random.choice([0, 1], size=num_records, p=[0.71, 0.29])
    
    # 14. Personal Loan (Target variable, 0/1, ~9.6% acceptance rate)
    logit = (
        -8.5
        + 0.045 * income
        + 0.40 * cc_avg
        + 0.8 * (education - 1)
        + 0.4 * (family - 1)
        + 3.2 * cd_account
        + 0.5 * securities
        - 0.01 * age
    )
    prob = 1 / (1 + np.exp(-logit))
    personal_loan = (np.random.uniform(0, 1, size=num_records) < prob).astype(int)
    
    df = pd.DataFrame({
        'ID': ids,
        'Age': age,
        'Experience': exp,
        'Income': income,
        'ZIP Code': zip_code,
        'Family': family,
        'CCAvg': cc_avg,
        'Education': education,
        'Mortgage': mortgage,
        'Personal Loan': personal_loan,
        'Securities Account': securities,
        'CD Account': cd_account,
        'Online': online,
        'CreditCard': credit_card
    })
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Dataset successfully created at {filename} with shape {df.shape}")
    print(f"Personal Loan acceptance count: {df['Personal Loan'].sum()} ({df['Personal Loan'].mean()*100:.2f}%)")

if __name__ == '__main__':
    target_path = os.path.join(os.path.dirname(__file__), 'UniversalBank.csv')
    generate_universal_bank_data(target_path)
