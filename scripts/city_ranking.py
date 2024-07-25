import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from adjustText import adjust_text

# Import the CSV
df = pd.read_csv('archive/merged-calculated.csv')
df.drop(df.tail(1).index, inplace=True)

# Select the numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns

# Initialize KNNImputer with k=5 (number of nearest neighbors to consider)
imputer = KNNImputer(n_neighbors=5)

# Fit and transform the numerical columns using KNNImputer
df_filled = pd.DataFrame(imputer.fit_transform(df[numerical_cols]),
                         columns=numerical_cols)

# Update the original DataFrame with filled values
df[numerical_cols] = df_filled

#df.to_csv("data/fillout.csv")


# Extract the region column for later use
regions = df.pop('Region')

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[numerical_cols])
scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols)
scaled_df.insert(0, 'Region', regions)

# Select the columns with negative variables
negative_columns = ['air_pollution_index', 'clearance-rate-divided,' 'crime_amount',
                    'household_spending_per_month', 'car-amount-rate', 'unemployment-rate']

# Make the negative columns show up as such
scaled_df[numerical_cols] = scaled_df[numerical_cols].apply(lambda x: x * -1
                                                            if x.name in negative_columns
                                                            else x)
#scaled_df.to_csv("data/scaled.csv")

# Sum up the individual ratings for each region to get their total score
scaled_df['Total_Score'] = scaled_df[numerical_cols].sum(axis=1)

# Sort the DataFrame based on 'Total_Score' in descending order
df_sorted = scaled_df.sort_values(by='Total_Score', ascending=False)

# Display the sorted DataFrame
df_sorted[['Region', 'Total_Score']].to_csv('final_rating.csv')

# Rating bar chart
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

# Normalize Total_Score to a scale of 0 to 100
max_score = df_sorted['Total_Score'].max()
df_sorted['Normalized_Score'] = (df_sorted['Total_Score'] / max_score) * 100

# Normalized rating bar chart
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

# Prepare the data for clustering
X = scaled_df.drop(columns=['Region', 'Total_Score'])

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
scaled_df['Cluster'] = kmeans.fit_predict(X)

# Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

# Create a DataFrame for the principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pc_df['Cluster'] = scaled_df['Cluster']
pc_df['Region'] = regions

# Create mesh grid for decision boundary
h = .005
x_min, x_max = pc_df['PC1'].min() - 1, pc_df['PC1'].max() + 1
y_min, y_max = pc_df['PC2'].min() - 1, pc_df['PC2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh using the trained model
Z = kmeans.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Plot the decision boundary by assigning a color to each point in the mesh
plt.figure(figsize=(14, 10))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['red', 'yellow', 'green']))

# Plot the clusters
colors = ['red', 'yellow', 'green']
labels = ['Lower life quality', 'Average life quality', 'Higher life quality']
for cluster in range(3):
    clustered_data = pc_df[pc_df['Cluster'] == cluster]
    plt.scatter(clustered_data['PC1'], clustered_data['PC2'],
                label=f'{labels[cluster]}', color=colors[cluster])

# Add region labels to the plot
texts = []
for i in range(pc_df.shape[0]):
    texts.append(plt.text(pc_df.loc[i, 'PC1'], pc_df.loc[i, 'PC2'], pc_df.loc[i, 'Region'],
                          fontsize=7, ha='right'))

# Adjust text to avoid overlap
adjust_text(texts, arrowprops=dict(arrowstyle='->'))

plt.xlabel('Principal Component 1 (most variance)')
plt.ylabel('Principal Component 2 (second most variance)')
plt.title('Clusters Visualization')
plt.legend()
plt.show()

# Prepare data for bar plot
df_sorted = scaled_df.sort_values(by='Total_Score', ascending=False)

# Drop the last two columns for clustering
X = df_sorted.drop(columns=['Region', 'Total_Score'])

# Perform KMeans clustering again
kmeans = KMeans(n_clusters=3, random_state=42)
df_sorted['Cluster'] = kmeans.fit_predict(X)

# # Define the labels
labels_bar = ['Average life quality', 'Lower life quality', 'Higher life quality']

# Define the colors for each cluster
colors_bar = {0: 'yellow', 1: 'red', 2: 'green'}

# Plotting the bar chart
plt.figure(figsize=(10, 6))

# Loop through each row and plot a bar with the corresponding color
for idx, row in df_sorted.iterrows():
    plt.barh(row['Region'], row['Total_Score'], color=colors_bar[row['Cluster']])

# Set labels and legend
plt.xlabel('Total Score')
plt.title('Total Score with Colored Clusters')
plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color) for color in colors_bar.values()],
           labels=labels_bar,
           loc='lower right')

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

