import os
import urllib.request
import pandas as pd

def fetch_and_save_kaggle_dataset():
    # Public raw URLs hosting the exact Kaggle dataset (Bank_Personal_Loan_Modelling.csv / UniversalBank.csv)
    urls = [
        "https://raw.githubusercontent.com/aiwei/inst414-21s/main/Bank_Personal_Loan_Modelling.csv",
        "https://raw.githubusercontent.com/toby-all/Personal-Loan-Acceptance-Prediction/master/Bank_Personal_Loan_Modelling.csv",
        "https://raw.githubusercontent.com/karthik-hy/Bank-Personal-Loan-Modelling/master/Bank_Personal_Loan_Modelling.csv"
    ]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = os.path.join(base_dir, 'dataset', 'UniversalBank.csv')
    
    print(f"Downloading authentic Kaggle Universal Bank dataset to {target_path}...")
    success = False
    
    for url in urls:
        try:
            print(f"Attempting download from: {url}")
            df = pd.read_csv(url)
            print(f"Downloaded successfully! Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Normalize column names if needed (e.g., 'ZIP Code' vs 'ZIPCode')
            col_map = {
                'ZIPCode': 'ZIP Code',
                'ZipCode': 'ZIP Code',
                'PersonalLoan': 'Personal Loan',
                'SecuritiesAccount': 'Securities Account',
                'CDAccount': 'CD Account'
            }
            df = df.rename(columns=col_map)
            
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            df.to_csv(target_path, index=False)
            print(f"Saved dataset to {target_path}")
            print(f"Target 'Personal Loan' counts:\n{df['Personal Loan'].value_counts()}")
            success = True
            break
        except Exception as e:
            print(f"Failed to fetch from {url}: {e}")
            
    if not success:
        print("Could not download from remote mirrors.")

if __name__ == '__main__':
    fetch_and_save_kaggle_dataset()
