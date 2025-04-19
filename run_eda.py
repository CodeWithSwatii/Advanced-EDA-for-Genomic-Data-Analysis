import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

data_file = "genomic_data.csv"

if not os.path.exists(data_file):
    raise FileNotFoundError(f"The file '{data_file}' does not exist. Please provide the correct path.")

data = pd.read_csv(data_file)
print("Dataset Info:")
data.info()

print("\nFirst few rows of the dataset:")
print(data.head())

missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

print("\nSummary Statistics:")
print(data.describe())

numeric_cols = data.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

plt.figure(figsize=(10, 8))
corr_matrix = data[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

categorical_cols = data.select_dtypes(include=["object", "category"]).columns
print("\nCategorical Columns:")
print(categorical_cols)

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data[col])
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.show()

# Advanced visualization: Pairplot for numeric columns
if len(numeric_cols) > 1:
    sns.pairplot(data[numeric_cols], diag_kind="kde")
    plt.suptitle("Pairplot of Numeric Features", y=1.02)
    plt.show()

# Detecting skewness in numeric data
skewness = data[numeric_cols].skew()
print("\nSkewness of Numeric Columns:")
print(skewness)

if "GC_Content" in data.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data["GC_Content"], kde=True, bins=30)
    plt.title("Distribution of GC Content")
    plt.xlabel("GC Content (%)")
    plt.ylabel("Frequency")
    plt.show()

print("Advanced EDA for Genomic Data Analysis: Identifying Genetic Variations Through Visualization completed.")
