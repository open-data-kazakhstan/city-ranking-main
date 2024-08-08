import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from adjustText import adjust_text
from tabulate import tabulate

import seaborn as sns

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


# Extract the region column for future use
regions = df.pop('Region')


# Update the original DataFrame with filled values
df[numerical_cols] = df_filled

# Extract indicators for separate processing
gdp_per_capita_diff_process = df_filled.pop('gdp_per_capita')
air_pollution_index_diff_process = df_filled.pop('air_pollution_index')
df_filled['crime_rate'] = (df_filled['crime_amount'] / df_filled['Population']) * 100000
#print(df_filled['crime_rate'])
crime_rate_diff_process = df_filled.pop('crime_rate')
life_expectancy_diff_process = df_filled.pop('life_expectancy')

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[numerical_cols])
scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols)
scaled_df.insert(0, 'Region', regions)


# Process indicator GDP - logarithmic transformation - then maxmin
scaled_df.insert(5, 'gdp_per_capita_diff_process', gdp_per_capita_diff_process)

scaled_df['log_gdp_per_capita'] = np.log(scaled_df['gdp_per_capita_diff_process'])

# Min-Max scaling
min_val = scaled_df['log_gdp_per_capita'].min()
max_val = scaled_df['log_gdp_per_capita'].max()
scaled_df['normalized_gdp'] = (scaled_df['log_gdp_per_capita'] - min_val) / (max_val - min_val)

# Process indicator Air Pollution Index -  piecewise normalization approach based on World Health Organization (WHO)
scaled_df.insert(1, 'air_pollution_index_diff_process', air_pollution_index_diff_process)
def normalize_air_pollution(x, threshold=50):  # Assuming WHO threshold of 50
    if x <= threshold:
        return x / (2 * threshold)
    else:
        return 0.5 + 0.5 * (1 - np.exp(-(x - threshold) / threshold))
scaled_df['normalized_air_pollution'] = scaled_df['air_pollution_index_diff_process'].apply(normalize_air_pollution)

# Process crime amount - Square root transformation and Min-Max scaling
# Calculate crime rate per 100,000 inhabitants
scaled_df.insert(6, 'crime_rate_diff_process', crime_rate_diff_process)

# Square root transformation and Min-Max scaling
scaled_df['sqrt_crime_rate'] = np.sqrt(scaled_df['crime_rate_diff_process'])
min_val = scaled_df['sqrt_crime_rate'].min()
max_val = scaled_df['sqrt_crime_rate'].max()
scaled_df['normalized_crime_rate'] = 1 - (scaled_df['sqrt_crime_rate'] - min_val) / (max_val - min_val)


# Process life expectancey - with Global lower benchmark to 65
scaled_df.insert(6, 'life_expectancy_diff_process', life_expectancy_diff_process)
global_min = 65  # Global lower benchmark
observed_max = scaled_df['life_expectancy_diff_process'].max()

scaled_df['normalized_life_expectancy'] = (scaled_df['life_expectancy_diff_process'] - global_min) / (observed_max - global_min)

# Normalized final dataset

df_scaled_final = pd.DataFrame().assign(region=scaled_df['Region'], 
air_pollution=scaled_df['normalized_air_pollution'], 
avg_salary=scaled_df['avg_salary'],
population=scaled_df['Population'],
clearance_rate=scaled_df['clearance-rate-divided'],
life_expectancy=scaled_df['normalized_life_expectancy'],
crime_rate=scaled_df['normalized_crime_rate'], 
entertainment_places_rate=scaled_df['Entertainment-places-rate'],
gdp=scaled_df['normalized_gdp'],
postgraduate_education=scaled_df['Postgraduate_Education'],
higher_education=scaled_df['Higher_Education'],
household_spending=scaled_df['household_spending_per_month'],
med_institution_rate=scaled_df['Med-institution-rate'],
car_amount_rate=scaled_df['car-amount-rate'],
school_number_rate=scaled_df['School-number-rate'],
transport_quanity_rate=scaled_df['Transport-quanity-rate'],
unemployment_rate=scaled_df['unemployment-rate']
)

# Transform negative indicators
df_scaled_final['crime_rate'] = 1 - df_scaled_final['crime_rate']
df_scaled_final['unemployment_rate'] = 1 - df_scaled_final['unemployment_rate']
df_scaled_final['air_pollution'] = 1 - df_scaled_final['air_pollution']
df_scaled_final['car_amount_rate'] = 1 - df_scaled_final['car_amount_rate']

# Aggregation - weighting 
# Set 'region' as the index
# Create a mapping of original index to region names
index_to_region = {idx: region for idx, region in enumerate(df_scaled_final['region'])}

# Set 'region' as the index
df_scaled_final.set_index('region', inplace=True)
df_scaled_final.to_csv("data/scaled_final.csv")
print("Summary of Normalized Indicators:")
print(tabulate(df_scaled_final, headers='keys', tablefmt='psql'))
print(df_scaled_final.describe())

print("\nCorrelation Matrix:")
print(df_scaled_final.corr())
print(tabulate(df_scaled_final.corr(), headers='keys', tablefmt='psql'))


# Define the hierarchy
criteria = [
    "Economic", "Social", "Environmental", "Infrastructure"
]

sub_criteria = {
    "Economic": ["avg_salary", "gdp", "household_spending"],
    "Social": ["population", "life_expectancy", "crime_rate", "higher_education", "unemployment_rate","clearance_rate", "postgraduate_education"],
    "Environmental": ["air_pollution"],
    "Infrastructure": ["entertainment_places_rate", "med_institution_rate", "car_amount_rate", "school_number_rate", "transport_quanity_rate"]
}

# Identify indicators with negative effects
negative_effects = ["crime_rate", "unemployment_rate", "air_pollution", "car_amount_rate"]

''' This is for equal weight
# Predefined pairwise comparisons for main criteria
predefined_main_comparisons = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
]

# Predefined pairwise comparisons for each set of sub-criteria
predefined_sub_comparisons = {
    "Economic": [
        [1, 1, 1],  # avg_salary compared to gdp, household_spending
        [1, 1, 1],  # gdp compared to avg_salary, household_spending
        [1, 1, 1]
        ],
    "Social": [
        [1, 1, 1, 1, 1,1,1],  
    [1, 1, 1, 1, 1,1,1], 
    [1, 1, 1, 1, 1,1,1],  
    [1, 1, 1, 1, 1,1,1],  
    [1, 1, 1, 1, 1,1,1],
    [1, 1, 1, 1, 1,1,1] ,
    [1, 1, 1, 1, 1,1,1]  
    ],
    "Environmental": [
        [1]
    ],
    "Infrastructure": [
        [1, 1, 1, 1, 1],  # entertainment_places_rate compared to med_institution_rate, car_amount_rate, school_number_rate, transport_quanity_rate
    [1, 1, 1, 1, 1],  # med_institution_rate compared to entertainment_places_rate, car_amount_rate, school_number_rate, transport_quanity_rate
    [1, 1, 1, 1, 1],  # car_amount_rate compared to entertainment_places_rate, med_institution_rate, school_number_rate, transport_quanity_rate
    [1, 1, 1, 1, 1],  # school_number_rate compared to entertainment_places_rate, med_institution_rate, car_amount_rate, transport_quanity_rate
    [1, 1, 1, 1, 1] 
    ]
}
'''
# https://simplemaps.com/custom/country/hqwiYSyU#finish

# Predefined pairwise comparisons for main criteria
predefined_main_comparisons = [
    #       Economic  Social  Environmental  Infrastructure
    [1,       3,        1/2,            4],  # Economic
    [1/3,        1,          1/5,            2],    # Social
    [2,        5,          1,              6],    # Environmental
    [1/4,        1/2,        1/6,            1]     # Infrastructure
]
predefined_sub_comparisons = {
    "Economic": [
        [1, 2, 3],
        [1/2, 1, 2],
        [1/3, 1/2, 1]
        ],
    "Social": [
        [1, 1/3, 3, 1/2, 2, 1/4, 1/5],
        [3, 1, 5, 2, 4, 1/2, 1/3],
        [1/3, 1/5, 1, 1/4, 1/2, 1/6, 1/7],
        [2, 1/2, 4, 1, 3, 1/3, 1/2],
        [1/2, 1/4, 2, 1/3, 1, 1/5, 1/6],
        [4, 2, 6, 3, 5, 1, 2],
        [5, 3, 7, 2, 6, 1/2, 1]
    ],
    "Environmental": [
        [1]
    ],
    "Infrastructure": [
        [1, 3, 1/2, 2, 4],
        [1/3, 1, 1/4, 1/2, 3],
        [2, 4, 1, 3, 5],
        [1/2, 2, 1/3, 1, 2],
        [1/4, 1/3, 1/5, 1/2, 1]
    ]
}

# Create pairwise comparison matrices
def create_comparison_matrix(n, comparisons):
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            matrix[i][j] = comparisons[i][j]
            matrix[j][i] = 1 / comparisons[i][j]
    return matrix

main_matrix = create_comparison_matrix(len(criteria), predefined_main_comparisons)

# Calculate weights
def calculate_weights(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_index = np.argmax(eigvals)
    eigvec = eigvecs[:, max_index].real
    return eigvec / np.sum(eigvec)

main_weights = calculate_weights(main_matrix)

# Check consistency
def consistency_ratio(matrix, weights):
    n = len(matrix)
    lambda_max = np.sum(np.dot(matrix, weights) / weights) / n
    consistency_index = (lambda_max - n) / (n - 1)
    random_index = {3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    return consistency_index / random_index[n]

cr = consistency_ratio(main_matrix, main_weights)
print(f"Consistency Ratio: {cr}")

# Repeat for sub-criteria
sub_weights = {}
for criterion in criteria:
    if len(sub_criteria[criterion]) > 1:
        print(f"\nComparing sub-criteria for {criterion}")
        sub_matrix = create_comparison_matrix(len(sub_criteria[criterion]), predefined_sub_comparisons[criterion])
        #print(sub_matrix)
        sub_weights[criterion] = calculate_weights(sub_matrix)
    else:
        sub_weights[criterion] = [1.0]

print(f"Main Weights: {main_weights}")
print(f"Sub Weights: {sub_weights}")


# Plotting the weights
def plot_weights(criteria, weights, title):
    plt.figure(figsize=(10, 6))
    plt.bar(criteria, weights, color='skyblue')
    plt.xlabel('Criteria')
    plt.ylabel('Weights')
    plt.title(title)
    plt.show()

# Plot main weights
plot_weights(criteria, main_weights, "Main Criteria Weights")

# Plot sub-criteria weights
for criterion in criteria:
    if len(sub_criteria[criterion]) > 1:
        plot_weights(sub_criteria[criterion], sub_weights[criterion], f"Sub-Criteria Weights for {criterion}")

# Calculate global weights:
global_weights = {}
for i, criterion in enumerate(criteria):
    for j, sub_criterion in enumerate(sub_criteria[criterion]):
        global_weights[sub_criterion] = main_weights[i] * sub_weights[criterion][j]

#print("\nGlobal Weights:")
for criterion, weight in global_weights.items():
    print(f"{criterion}: {weight:.4f}")
    
# Aggregation of Indicators
weighted_scores = pd.DataFrame()
for indicator, weight in global_weights.items():
    if indicator in df_scaled_final.columns:
        weighted_scores[indicator] = df_scaled_final[indicator] * weight

# Calculate overall sustainability score
df_scaled_final['sustainability_score'] = weighted_scores.sum(axis=1)

print("Top 5 regions by sustainability score:")
print(df_scaled_final['sustainability_score'].nlargest(5))

print("\nBottom 5 regions by sustainability score:")
print(df_scaled_final['sustainability_score'].nsmallest(5))


# Visualization of Aggregated Indicators

df_scaled_final.sort_values(by ='sustainability_score',ascending=False).to_csv("data/weight_with_sustainability_score_equal_weight.csv")
plt.figure(figsize=(12, 6))
ax = df_scaled_final['sustainability_score'].sort_values(ascending=False).plot(kind='bar')

# Adding titles and labels
plt.title('Regional Rankings Based on Weighted Sustainability Score')
plt.xlabel('Region')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')

# Adding exact values on the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points')

plt.tight_layout()
plt.show()




def ranking_clustering():
    # Select the columns with negative variables
    negative_columns = ['air_pollution_index', 'clearance-rate-divided,' 'crime_amount',
                        'household_spending_per_month', 'car-amount-rate', 'unemployment-rate']

    # Make the negative columns show up as such
    scaled_df[numerical_cols] = scaled_df[numerical_cols].apply(lambda x: x * -1
                                                                if x.name in negative_columns
                                                                else x)

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

