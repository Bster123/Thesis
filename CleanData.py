import pandas as pd 
import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

p_value_threshold = 0.05 
chi2_threshold = 200
r_threshold = 0.1

#Read Uncleaned Data set
df = pd.read_csv("accepted_2007_to_2018Q4.csv")

### Clean Data

# 1. Remove variables where at least 10% of the values are missing
threshold = 0.1 * len(df)
df = df.dropna(thresh=(len(df) - threshold), axis=1)

# 2. Replace inf values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 3. Remove rows with any NaN values
df.dropna(inplace=True)

#remove all loans that are current
filtered_df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]


###Feature Engineering
#Edit data to for better model and training performance. 

#Binary Encode loan default 
filtered_df.loc[filtered_df['loan_status'] == 'Fully Paid', 'loan_status'] = 1  #1 means good!
filtered_df.loc[filtered_df['loan_status'] == 'Charged Off', 'loan_status'] = 0 #Zero means bad :()

# Convert the 'issue_d' column to datetime format
filtered_df['issue_d'] = pd.to_datetime(filtered_df['issue_d'], format='%b-%Y')
filtered_df['issue_month'] = filtered_df['issue_d'].dt.month
filtered_df['issue_year'] = filtered_df['issue_d'].dt.year

#Comment out to keep loans from 2016 only 
# filtered_df = filtered_df.loc[(filtered_df['issue_year'] == 2016)]

#Comment out to keep loans from 2014-2016 only 
# filtered_df = filtered_df.loc[(filtered_df['issue_year'] >= 2014) & (filtered_df['issue_year'] <= 2016)]


# Add new variable of ratio between yearly income and monthly repayment 
filtered_df['repayment_ratio'] = (filtered_df['installment'] * 12) / filtered_df['annual_inc']


#Add Federal Interest Rates
fed_funds_df = pd.read_csv('FEDFUNDS.csv')
fed_funds_df['DATE'] = pd.to_datetime(fed_funds_df['DATE'])
fed_funds_df['fed_month'] = fed_funds_df['DATE'].dt.month
fed_funds_df['fed_year'] = fed_funds_df['DATE'].dt.year
filtered_df = pd.merge(filtered_df, fed_funds_df, left_on=['issue_month', 'issue_year'], right_on=['fed_month', 'fed_year'], how='left')
filtered_df.drop(['fed_month', 'fed_year', 'DATE'], axis=1, inplace=True)
filtered_df['int_FED_ratio'] = filtered_df['int_rate'] / filtered_df['FEDFUNDS']

#Add Federal deliquency rates at time of application
quarterly_data = pd.read_csv('DRALACBN.csv')
filtered_df['DATE'] = pd.to_datetime(filtered_df['issue_year'].astype(str) + '-' + filtered_df['issue_month'].astype(str))
quarterly_data['DATE'] = pd.to_datetime(quarterly_data['DATE'])
quarterly_data = quarterly_data.sort_values(by='DATE')
filtered_df = pd.merge_asof(filtered_df.sort_values('DATE'), quarterly_data, on='DATE', direction='backward').sort_index()
filtered_df.drop(columns=['DATE'], axis=1, inplace=True)

#Make all other variables numerical: 
# 'emp_title', 'purpose', 'title' Require Text analysis that is currently not performed
remove_columns = ['disbursement_method', 'issue_d','earliest_cr_line', 'zip_code', 'addr_state', 'last_pymnt_d', 'last_credit_pull_d', 'url', 'initial_list_status','emp_title', 'purpose', 'title']
nominal_columns = ['pymnt_plan', 'application_type', 'hardship_flag', 'debt_settlement_flag']  # Categorical data with No Order
ordinal_columns = ['term', 'grade','sub_grade', 'emp_length', 'verification_status', 'home_ownership']  # Cateogrial data with Order


label_encoder = LabelEncoder()
filtered_df.drop(remove_columns, axis=1, inplace=True)

#Encode columns with small number of unique values
for column in ordinal_columns:
    filtered_df[column] = label_encoder.fit_transform(filtered_df[column])

#Reduce Cardinality of columns with too many unique values: 
for feature in nominal_columns: 
    title_counts = filtered_df[feature].value_counts()
    mask = filtered_df[feature].isin(title_counts[title_counts < 10].index)
    filtered_df.loc[mask, feature] = 'Other'

filtered_df = pd.get_dummies(filtered_df, columns=nominal_columns)

#Normalise Data
scaler = StandardScaler()
columns_to_scale = [col for col in filtered_df.columns if col not in ['loan_status', 'total_pymnt', 'funded_amnt', 'term', 'int_rate','issue_month', 'issue_year']]
filtered_df[columns_to_scale] = scaler.fit_transform(filtered_df[columns_to_scale])

#Calculate/create profit or loss column to target profitablity 
filtered_df['profit_or_Loss'] = filtered_df['total_pymnt'] - filtered_df['funded_amnt']
filtered_df['profitable'] = (filtered_df['profit_or_Loss'] > 0).astype(int)


###Calculate Correlation of variables
 
results = []

#Variables that should not be used to train in the model as they rely on the outcome
trivialExplanators = [
    'last_pymnt_d',
    'recoveries',
    'collection_recovery_fee',
    'total_rec_late_fee',
    'total_pymnt',
    'total_pymnt_inv',
    'last_pymnt_amnt',
    'total_rec_prncp',
    'total_rec_int',
    ]


for column in filtered_df.columns:
    if column in ['loan_status']: #Add a column name here to force it to be in Feature set.
        continue  
    
    # Calculate Pearson correlation for variables with loan_status
    correlation, p_value = pearsonr(filtered_df['loan_status'], filtered_df[column])

    #Apply p-value and r-value thresholds and add to results table
    if abs(correlation) > r_threshold and p_value <= p_value_threshold and column not in trivialExplanators: 
        results.append({'Variable': column, 'Correlation': round(correlation,4), 'P-value': round(p_value,4)})
    else: 
        filtered_df = filtered_df.drop(column, axis=1)
        print('Removed')

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
###Save csv of sorted Correlations
results_df.to_csv('correlation_results.csv', index=False)


### Save Cleaned Data 
filtered_df.to_csv('Cleaned_Data.csv', index=False)

#Save top 25 rows to different file for visual inspection 
cut_df = filtered_df.head(25)
cut_df.to_csv('Top_rows.csv', index=False)
