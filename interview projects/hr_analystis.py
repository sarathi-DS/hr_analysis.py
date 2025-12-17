# %%


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score


# Load dataset
HR_analystis = pd.read_csv("HR_Data_Analytics - HR_Data_Analytics.csv")


# Check missing values
print(HR_analystis.isna().sum())


# Check duplicates
# I checked the dataset for duplicate records to ensure every entry is unique. Fortunately, there were no duplicates, so no rows needed to be removed.
#This confirms that the dataset is clean and ready for further analysis
print("Duplicate rows:", HR_analystis.duplicated().sum())


# Convert Hire_Date to datetime
HR_analystis['Hire_Date'] = pd.to_datetime(HR_analystis['Hire_Date'])
# Create numeric features
HR_analystis['Hire_Year'] = HR_analystis['Hire_Date'].dt.year
HR_analystis['Hire_Month'] = HR_analystis['Hire_Date'].dt.month
HR_analystis['Hire_Day'] = HR_analystis['Hire_Date'].dt.day

HR_analystis.drop('Hire_Date', axis=1, inplace=True)
# Drop irrelevant columns
HR_analystis = HR_analystis.drop(columns=['Unnamed: 0', 'Employee_ID', 'Full_Name', 'Location'], errors='ignore')
# Outlier Handling using IQR
def cap_outliers_iqr(df, cols):
    for col in cols:
        q1, q3 = HR_analystis[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        HR_analystis[col] = HR_analystis[col].clip(lower, upper)
    return df

HR_analystis = cap_outliers_iqr(HR_analystis, ['Salary_INR', 'Experience_Years'])
# Descriptive statistics
print(HR_analystis.describe())
# Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_df = HR_analystis.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=True)
plt.show()
# Salary distribution by Job Title
plt.figure(figsize=(25, 8))
sns.barplot(x="Job_Title", y="Salary_INR", data=HR_analystis)
plt.title("Pay Gaps Across Different Roles")
plt.xticks(rotation=30)
plt.show()
# Salary distribution by Department
grouped = HR_analystis.groupby('Department')['Salary_INR'].sum().sort_values(ascending=False)
plt.pie(grouped, labels=grouped.index, autopct='%1.1f%%', startangle=90)
plt.title('Salary distribution by Department')
plt.axis('equal')
plt.show()
# Encode categorical variables
def encode_labels(HR_analystis, columns):
    for col in columns:
        if HR_analystis[col].dtype == 'object':
            le = LabelEncoder()
            HR_analystis[col] = le.fit_transform(HR_analystis[col])
    return HR_analystis

HR_analystis = encode_labels(HR_analystis, HR_analystis.columns)
# Features and target
X = HR_analystis.drop('Status', axis=1)
y = HR_analystis['Status']
# Scale numeric features for Logistic Regression
num_cols = ['Experience_Years', 'Salary_INR', 'Performance_Rating']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
# Train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
# Random Forest with ROC-AUC & F1
def random_model(x_train, y_train, x_test, y_test, X_full=None, y_full=None):
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    model.fit(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    # Evaluation metrics
    print("\n Random Forest Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))
    print("F1 Score (weighted):", f1_score(y_test, y_pred, average='weighted'))
    print("ROC-AUC (ovr):", roc_auc_score(y_test, y_prob, multi_class='ovr'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Cross-validation if full dataset provided
    if X_full is not None and y_full is not None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='f1_weighted')
        print("Mean CV F1 (weighted):", cv_scores.mean())

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=x_train.columns).sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(y=importances.index[:15], x=importances.values[:15])
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.show()

# Run models
random_model(x_train, y_train, x_test, y_test, X_full=None, y_full=None)