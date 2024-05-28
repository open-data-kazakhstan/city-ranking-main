import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer

# Import the CSV
df = pd.read_csv('./archive/merged_to_normalize.csv')
df.drop(df.tail(1).index,inplace=True)

# Select the numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns

# Initialize KNNImputer with k=5 (number of nearest neighbors to consider)
imputer = KNNImputer(n_neighbors=5)

# Fit and transform the numerical columns using KNNImputer
df_filled = pd.DataFrame(imputer.fit_transform(df[numerical_cols]),
                         columns=numerical_cols)

# Update the original DataFrame with filled values
df[numerical_cols] = df_filled

# Extract the region column for later use
regions = df.pop('Region')

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[numerical_cols])
scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols)
scaled_df.insert(0, 'Region', regions)

# Select the columns with negative variables
negative_columns = ['air_pollution_index', 'clearance_rate,' 'crime_amount',
                    'household_spending_per_month', 'unemployment_rate']

# Make the negative columns show up as such
scaled_df[numerical_cols] = scaled_df[numerical_cols].apply(lambda x: x * -1
                                                            if x.name in negative_columns
                                                            else x)

# Sum up the individual ratings for each region to get their total score
scaled_df['Total_Score'] = scaled_df[numerical_cols].sum(axis=1)

# Sort the DataFrame based on 'Total_Score' in descending order
df_sorted = scaled_df.sort_values(by='Total_Score', ascending=False)

# Display the sorted DataFrame
print(df_sorted[['Region', 'Total_Score']])

# Plot
plt.figure(figsize=(10, 6))
bars = plt.barh(df_sorted['Region'],
                df_sorted['Total_Score'],
                color='skyblue')

plt.xlabel('Total_Score')
plt.title('Cities and Their Total_Scores')
plt.gca().invert_yaxis()

# Annotate bars with their values
for i, bar in enumerate(bars):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{df_sorted.iloc[i]["Total_Score"]:.2f}',
             va='center', ha='left')

plt.tight_layout()
plt.show()

# Second graph

# Normalize Total_Score to a scale of 0 to 100
max_score = df_sorted['Total_Score'].max()
df_sorted['Normalized_Score'] = (df_sorted['Total_Score'] / max_score) * 100

# Plot
plt.figure(figsize=(11, 6))
bars = plt.barh(df_sorted['Region'],
                df_sorted['Normalized_Score'])

plt.xlabel('Normalized_Score')
plt.title('Cities and Their Scores Scaled to 100')
plt.gca().invert_yaxis()

# Annotate bars with their values
for i, bar in enumerate(bars):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f'{df_sorted.iloc[i]["Normalized_Score"]:.1f}',
             va='center', ha='left')

plt.tight_layout()
plt.show()
