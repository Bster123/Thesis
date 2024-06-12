import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import numpy as np
import matplotlib.pyplot as plt


use_EV = True # Set to True to enable EV calculations 

# Load the data
dataOG = pd.read_csv('cleaned_data.csv')

#Adjust this number to change total fraction of data used/ 
data_size = 1
data = dataOG.sample(frac=data_size, random_state=42)  # random_state for reproducibility

# Split the data into features and target
X = data.drop(['loan_status', 'profitable', 'profit_or_Loss'], axis=1)
y = data['loan_status'] #Set to 'profitable' to target profitable loans. 

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
print(pd.DataFrame(y_pred_proba).head())

X_test.to_csv('X_test_input.csv')
# Calculate profitability

# Reattach the original columns to for evaluation 
X_test_with_profit = X_test.copy()
X_test_with_profit['profit_or_Loss'] = data.loc[X_test.index, 'profit_or_Loss']
X_test_with_profit['int_rate'] = data.loc[X_test.index, 'int_rate']
X_test_with_profit['term'] = data.loc[X_test.index, 'term']
X_test_with_profit['profitable'] = data.loc[X_test.index, 'profitable']
X_test_with_profit['funded_amnt'] = data.loc[X_test.index, 'funded_amnt']
X_test_with_profit['ROI_per_annum'] = data.loc[X_test.index, 'ROI_per_annum']


def calculate_expected_value(row):
    expected_value = (row['P']) * (row['funded_amnt'] * row['int_rate']  * (np.where(row['term'] == 0, 3, 5)) / 100) + (1 - row['P']) * - row['funded_amnt']
    return expected_value

if use_EV: 
    X_test_with_profit['P'] = y_pred_proba
    X_test_with_profit['EV'] = X_test_with_profit.apply(calculate_expected_value, axis=1)
    y_pred = X_test_with_profit['EV'] > 0

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


# Append the predicted values to the test set
X_test_with_profit['Predicted_Profitable'] = y_pred
X_test_with_profit['loan_status'] = data.loc[X_test.index, 'loan_status']

# Calculate the total profit for the accepted loans
actual_loans_profit = X_test_with_profit['profit_or_Loss'].sum()
actual_cash_invested = X_test_with_profit['funded_amnt'].sum()
actual_return = (actual_loans_profit*100)/ actual_cash_invested
model_loans_profit = X_test_with_profit.loc[X_test_with_profit['Predicted_Profitable'] == 1, 'profit_or_Loss'].sum()
model_cash_invested = X_test_with_profit.loc[X_test_with_profit['Predicted_Profitable'] == 1, 'funded_amnt'].sum()
model_return = (model_loans_profit)*100 / model_cash_invested

print(f"Actual: Total Profit from accepted loans: {actual_loans_profit}")
print(f"Actual cash invested: {actual_cash_invested} ")
print(f"Actual Return: {actual_return:.2f}%")
print(f"Model: Total Profit from accepted loans: {model_loans_profit}")
print(f"Model cash invested: {model_cash_invested} ")

# Calculation of metrics
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)  # also known as recall
specificity = tn / (tn + fp)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
g_mean = np.sqrt(sensitivity * specificity)

# Output the metrics
print(f"Number of True Positives: {tp}")
print(f"Number of False Positives: {fp}")
print(f"Number of True Negatives: {tn}")
print(f"Number of False Negatives: {fn}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"G-mean: {g_mean:.2f}")
print(f"F1-SCORE: {f1:.2f}")

# Additional metrics based on business impact or other calculations could be added here
# Example for Model Return (assumes a placeholder calculation)
print(f"Model Return: {model_return:.2f}%")


def calculate_pscore(row):
    total_interest = row['funded_amnt'] * row['int_rate']  * (np.where(row['term'] == 0, 3, 5)) / 100

    if row['Predicted_Profitable'] == 1 and row['loan_status'] == 1:
        return total_interest / 100000
    elif row['Predicted_Profitable'] == 1 and row['loan_status'] == 0:
        return -row['funded_amnt']/ 100000
    elif row['Predicted_Profitable'] == 0 and row['loan_status'] == 1:
        return - total_interest/ 100000
    elif row['Predicted_Profitable'] == 0 and row['loan_status'] == 0:
        return row['funded_amnt']/ 100000
    else:
        return 0

#Obtain P-Score for model
X_test_with_profit['PScore'] = X_test_with_profit.apply(calculate_pscore, axis=1)
Pscore_sum = X_test_with_profit['PScore'].sum() 
print(f'Model Pscore: {Pscore_sum}')

#Obtain P-Score for original data
X_test_with_profit['Predicted_Profitable'] = 1 #Set all values to 1 as they were actually invested in reality.
X_test_with_profit['PScore'] = X_test_with_profit.apply(calculate_pscore, axis=1)
Pscore_sum = X_test_with_profit['PScore'].sum() 
print(f'Original Pscore: {Pscore_sum}')
