import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


np.random.seed(42)

num_houses = 100
house_data = {
    'House_ID': np.arange(1, num_houses + 1),
    'Price': np.random.randint(100000, 500000, size=num_houses),
    'Size_sqft': np.random.randint(500, 3500, size=num_houses),
    'Bedrooms': np.random.randint(1, 6, size=num_houses),
    'Bathrooms': np.random.randint(1, 4, size=num_houses),
    'Year_Built': np.random.randint(1950, 2021, size=num_houses),
    'Garage': np.random.choice(['Yes', 'No'], size=num_houses)
}

# DataFrame
df = pd.DataFrame(house_data)

#first few rows
print(df.head(5))

# missing values
print(df.isnull().sum())

# missing values if any
df['Garage'].fillna('No', inplace=True)

# Summary statistics
print(df.describe())

# Price distribution
sns.histplot(df['Price'], bins=20, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Relationship between Price and Size
sns.scatterplot(x='Size_sqft', y='Price', data=df)
plt.title('Price vs. Size')
plt.xlabel('Size (sqft)')
plt.ylabel('Price')
plt.show()

# Boxplot of Prices by number of Bedrooms
sns.boxplot(x='Bedrooms', y='Price', data=df)
plt.title('Price Distribution by Number of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Price')
plt.show()

# Pairplot to see pairwise relationships
sns.pairplot(df[['Price', 'Size_sqft', 'Bedrooms', 'Bathrooms']])
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()

# Heatmap for correlation matrix
corr_matrix = df[['Price', 'Size_sqft', 'Bedrooms', 'Bathrooms', 'Year_Built']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()