import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../Log_Data.csv')

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

print("Missing values:")
print(df.isnull().sum())
df = df.dropna()

print("\nDescriptive statistics:")
print(df.describe())

print("\nData types:")
print(df.dtypes)

print("\nUnique values:")
for column in df.columns:
    print(f"{column}: {df[column].nunique()}")


plt.figure(figsize=(8, 6))
plt.hist(df['Behavioral Score'], bins=10, edgecolor='black')
plt.xlabel('Behavioral Score')
plt.ylabel('Frequency')
plt.title('Histogram of Behavioral Score')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x='User Role', y='Behavioral Score', data=df)
plt.xlabel('User Role')
plt.ylabel('Behavioral Score')
plt.title('Box Plot of Behavioral Score by User Role')
plt.show()


corr_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


print("\nMean Behavioral Score by User Role:")
print(df.groupby('User Role')['Behavioral Score'].mean())
print("\nMean Behavioral Score by Resource Sensitivity:")
print(df.groupby('Resource Sensitivity')['Behavioral Score'].mean())


print("\nCount of records by User Role:")
print(df.groupby('User Role')['Behavioral Score'].count())
print("\nCount of records by Resource Sensitivity:")
print(df.groupby('Resource Sensitivity')['Behavioral Score'].count())

plt.figure(figsize=(8, 6))
plt.scatter(df['User ID'], df['Behavioral Score'])
plt.xlabel('User ID')
plt.ylabel('Behavioral Score')
plt.title('Scatter Plot of User ID vs Behavioral Score')
plt.show()


plt.figure(figsize=(8, 6))
df.groupby('User Role')['Behavioral Score'].mean().plot(kind='bar')
plt.xlabel('User Role')
plt.ylabel('Mean Behavioral Score')
plt.title('Bar Chart of User Role vs Mean Behavioral Score')
plt.show()

plt.figure(figsize=(6, 4))
df['Access Granted'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Access Granted')
plt.ylabel('')
plt.show()