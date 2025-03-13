import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:/Users/ASUS/Downloads/test.csv"
df = pd.read_csv(file_path)

# Handling missing values
df["Age"] = df["Age"].fillna(df["Age"].median())  # Impute Age with median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())  # Impute Fare with median

# Dropping 'Cabin' column due to excessive missing values
df.drop(columns=["Cabin"], inplace=True)

# Removing duplicate records
df.drop_duplicates(inplace=True)

# Detecting and treating outliers using IQR method for Age and Fare
Q1 = df[["Age", "Fare"]].quantile(0.25)
Q3 = df[["Age", "Fare"]].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping outliers within the defined range
df["Age"] = df["Age"].clip(lower=lower_bound["Age"], upper=upper_bound["Age"])
df["Fare"] = df["Fare"].clip(lower=lower_bound["Fare"], upper=upper_bound["Fare"])

# Standardizing categorical values
df["Sex"] = df["Sex"].str.lower().str.strip()
df["Embarked"] = df["Embarked"].str.upper().str.strip()

# Summary statistics for numerical columns
summary_stats = df.describe().T

# Skewness for numerical columns
skewness = df.skew(numeric_only=True)

# Frequency distribution for categorical variables
categorical_columns = ["Sex", "Embarked", "Pclass"]
frequency_distributions = {col: df[col].value_counts() for col in categorical_columns}

# Plot histograms and box plots for numerical variables
numerical_columns = ["Age", "Fare"]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes = axes.flatten()

# Histograms
for i, col in enumerate(numerical_columns):
    sns.histplot(df[col], bins=20, kde=True, ax=axes[i])
    axes[i].set_title(f"Histogram of {col}")

# Box plots
for i, col in enumerate(numerical_columns):
    sns.boxplot(x=df[col], ax=axes[i+2])
    axes[i+2].set_title(f"Box plot of {col}")

plt.tight_layout()
plt.show()

# Bivariate Analysis
# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Scatter plots for continuous variable relationships
sns.pairplot(df, vars=["Age", "Fare"], hue="Pclass", palette="husl")
plt.show()

# Bar plots comparing categorical and numerical variables
sns.barplot(x="Pclass", y="Fare", data=df, errorbar=None)
plt.title("Average Fare by Passenger Class")
plt.show()

# Violin plot for Age distribution across Sex
sns.violinplot(x = "Sex", y="Age", data=df, palette="pastel",legend = False)
plt.title("Age Distribution by Sex")
plt.show()

# Box plot for Fare distribution across Embarked locations
sns.boxplot(x="Embarked", y="Fare", data=df, palette="muted",legend = False)
plt.title("Fare Distribution by Embarkation Point")
plt.show()

# Multivariate Analysis
# Pair plot to analyze multiple relationships
sns.pairplot(df, hue="Pclass", palette="husl")
plt.show()

# Heatmap to visualize correlations among multiple variables
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Enhanced Correlation Matrix")
plt.show()

# Grouped comparisons - Interaction between Pclass, Sex, and Fare
plt.figure(figsize=(8, 6))
sns.boxplot(x="Pclass", y="Fare", hue="Sex", data=df, palette="muted")
plt.title("Fare Distribution by Passenger Class and Sex")
plt.legend(title="Sex")
plt.show()

# Save the cleaned dataset
df.to_csv("C:/Users/ASUS/Downloads/cleaned_data.csv", index=False)

print("Data cleaning, univariate, bivariate, and multivariate EDA completed. Cleaned dataset saved as 'cleaned_data.csv'")

