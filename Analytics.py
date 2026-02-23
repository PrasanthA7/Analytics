

"""Exploratory Data Analysis of Sales Performance Using Python"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("Salesdata.csv")

print("Dataset Shape:", df.shape)
print("\nDataset Information:")
df.info()

print("\nSummary of the Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

print("\nSales Statistics:")
print("Mean:", df['Sales'].mean())
print("Median:", df['Sales'].median())
print("Standard Deviation:", df['Sales'].std())

# EDA
print("\nCategory-wise Sales & Profit:")
category_analysis = df.groupby('Category')[['Sales', 'Profit']].sum()
print(category_analysis)

print("\nRegion-wise Sales:")
print(df.groupby('Region')['Sales'].sum())

fig, axes = plt.subplots(3, 2, figsize=(14, 18))

category_analysis['Sales'].plot(
    kind='bar',
    ax=axes[0, 0],
    edgecolor='black',
    linewidth=1.5
)
axes[0, 0].set_title("Sales by Category")
axes[0, 0].set_ylabel("Sales")
axes[0, 1].hist(df['Profit'], bins=20, edgecolor='black', linewidth=1.2)
axes[0, 1].set_title("Profit Distribution")
axes[0, 1].set_xlabel("Profit")
axes[0, 1].set_ylabel("Frequency")


monthly_sales = df.resample('M', on='Order Date')['Sales'].sum()
axes[1, 0].plot(monthly_sales, marker='o', linestyle='-', linewidth=2)
axes[1, 0].set_title("Monthly Sales Trend")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Sales")


sns.heatmap(
    df[['Sales', 'Profit', 'Discount']].corr(),
    annot=True,
    linewidths=1,
    linecolor='black',
    ax=axes[1, 1]
)
axes[1, 1].set_title("Correlation Heatmap")

#Sales Distribution
axes[2, 0].hist(df['Sales'], bins=15, edgecolor='black', linewidth=1.2)
axes[2, 0].set_title("Sales Distribution")
axes[2, 0].set_xlabel("Sales")
axes[2, 0].set_ylabel("Frequency")

#Sales vs Profit
axes[2, 1].scatter(df['Sales'], df['Profit'], alpha=0.6)
axes[2, 1].set_title("Sales vs Profit")
axes[2, 1].set_xlabel("Sales")
axes[2, 1].set_ylabel("Profit")
plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.show()



df['Sales_normalized'] = (df['Sales'] - df['Sales'].min()) / (df['Sales'].max() - df['Sales'].min())
df['Sales_zscore'] = stats.zscore(df['Sales'])

print("\nNormalized Sales (first 5 rows):")
print(df['Sales_normalized'].head())

print("\nStandardized Sales (first 5 rows):")
print(df['Sales_zscore'].head())

print("\nLoss-making orders:")
loss_orders = df[df['Profit'] < 0]
print(loss_orders[['Category', 'Sub-Category', 'Profit']])

# Insights
print("\nINSIGHTS:")
print("\nSales data is not normally distributed")
print("\nSome categories generate losses")
print("\nNormalization and standardization help data comparison")

#Salary Analysis of Employees using Exploratory Data Analysis.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


df = pd.read_csv("Salary.csv")

print("The Given DataSet is:")
print(df.head())

print("\nInfo of the DataSet:")
print(df.info())

print("\nSummary of the dataset:")
print(df.describe())

print("\nFinding Missing Values:")
print(df.isnull().sum())

print("\nDisplaying Duplicate Values:")
print(df.duplicated().sum())

#Data Cleaning-
df = df.drop_duplicates()
df = df.dropna()

#Statistical Analysis
print("\nResults of the Statistical Analysis:")
print("Mean Salary:", df['Salary'].mean())
print("Median Salary:", df['Salary'].median())
print("Standard Deviation:", df['Salary'].std())
print("Minimum Salary:", df['Salary'].min())
print("Maximum Salary:", df['Salary'].max())

#Analysis using Group Wise

if 'Job Roles' in df.columns:
    print("\nAverage Salary by Job Role:")
    print(df.groupby('Job Roles')['Salary'].mean())

if 'Location' in df.columns:
    print("\nAverage Salary by Location (Top 10):")
    print(df.groupby('Location')['Salary'].mean().nlargest(10))

#Visualizations Using the Performance of Exploratory Data Analysis

fig, axes = plt.subplots(3, 2, figsize=(14, 18))

# 1. Salary Distribution
axes[0, 0].hist(df['Salary'], bins=20, edgecolor='black')
axes[0, 0].set_title("Salary Distribution")
axes[0, 0].set_xlabel("Salary")
axes[0, 0].set_ylabel("Frequency")

# 2. Salary Boxplot
axes[0, 1].boxplot(df['Salary'])
axes[0, 1].set_title("Salary Outliers")

# 3. Rating vs Salary (Using existing 'Rating' column)
if 'Rating' in df.columns:
    axes[1, 0].scatter(df['Rating'], df['Salary'], alpha=0.6)
    axes[1, 0].set_title("Rating vs Salary")
    axes[1, 0].set_xlabel("Rating")
    axes[1, 0].set_ylabel("Salary")
else:
    axes[1, 0].set_title("Rating column not found")

# 4. Salaries Reported vs Salary (Using existing 'Salaries Reported' column)
if 'Salaries Reported' in df.columns:
    sns.boxplot(x='Salaries Reported', y='Salary', data=df, ax=axes[1, 1])
    axes[1, 1].set_title("Salaries Reported vs Salary")
    axes[1, 1].set_xlabel("Number of Salaries Reported")
    axes[1, 1].set_ylabel("Salary")
else:
    axes[1, 1].set_title("'Salaries Reported' column not found")

# 5. Location vs Salary (Using existing 'Location' column)
if 'Location' in df.columns:
    top_locations = df['Location'].value_counts().nlargest(5).index
    sns.boxplot(x='Location', y='Salary', data=df[df['Location'].isin(top_locations)], ax=axes[2, 0])
    axes[2, 0].set_title("Salary by Top Locations")
    axes[2, 0].set_xlabel("Location")
    axes[2, 0].set_ylabel("Salary")
    axes[2, 0].tick_params(axis='x', rotation=45)
else:
    axes[2, 0].set_title("Location column not found")

# 6. Job Roles vs Salary
if 'Job Roles' in df.columns:

    top_job_roles = df['Job Roles'].value_counts().nlargest(5).index
    sns.boxplot(x='Job Roles', y='Salary', data=df[df['Job Roles'].isin(top_job_roles)], ax=axes[2, 1])
    axes[2, 1].set_title("Salary by Top Job Roles")
    axes[2, 1].set_xlabel("Job Role")
    axes[2, 1].set_ylabel("Salary")
    axes[2, 1].tick_params(axis='x', rotation=45)
else:
    axes[2, 1].set_title("'Job Roles' column not found")

plt.tight_layout()
plt.show()

# Combine Pie Chart and Correlation Heatmap in one figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Pie Chart for Employment Status Distribution
if 'Employment Status' in df.columns:
    employment_status_counts = df['Employment Status'].value_counts()
    ax1.pie(employment_status_counts, labels=employment_status_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Distribution of Employment Status")
    ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
else:
    ax1.set_title("'Employment Status' column not found")

# Correlation Analysis for numerical columns
numeric_cols_for_corr = df.select_dtypes(include=['int64','float64'])

sns.heatmap(numeric_cols_for_corr.corr(), annot=True, linewidths=1, linecolor='black', ax=ax2)
ax2.set_title("Correlation Heatmap")

plt.tight_layout() # Use tight_layout for single figures too
plt.show()
plt.close()


# Normalization & Standardization

if 'Salary' in df.columns:
    df['Salary_normalized'] = (df['Salary'] - df['Salary'].min()) / (df['Salary'].max() - df['Salary'].min())
    df['Salary_zscore'] = stats.zscore(df['Salary'])

print("\nNormalized Salary:")
print(df['Salary_normalized'].head())

print("\nStandardized Salary:")
print(df['Salary_zscore'].head())


# Outlier Detection
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['Salary'] < Q1 - 1.5*IQR) | (df['Salary'] > Q3 + 1.5*IQR)]
print("\nOUTLIERS:")
print(outliers)

#Final Insights from The Visualization:
print("\nINSIGHTS:")
print("1. Salary distribution is right-skewed indicating high-income outliers.")
print("2. Rating and 'Salaries Reported' can influence Salary.")
print("3. Salary varies significantly across different Locations and Job Roles.")
print("4. Normalization and standardization help in ML modeling.")
print("5. Outliers represent extremely high-paying roles.")
print("6. Correlation heatmap shows strongest factors affecting salary.")
