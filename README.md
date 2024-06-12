# city-ranking-main

This project provides an analysis of various cities based on multiple factors such as air pollution, crime rate, household spending, and unemployment rate. It uses a combination of data imputation, normalization, and visualization techniques to rank cities according to their overall scores.

## Source

All of the data is originally sourced from stat.gov.kz and later merged and cleaned for use in this project.

## Data

The dataset `merged_to_normalize.csv` contains various metrics for different cities. In total, 15 different parameters from our previous research were included. 

We have also added some metadata such as column descriptions and [data packaged](https://specs.frictionlessdata.io/data-package/) it.

### Data Cleaning:
- The last row of the CSV is dropped as it might contain summary or unwanted data.
- Only numerical columns are selected for further processing.

### Imputation:
- Missing values in the numerical columns are imputed using K-Nearest Neighbors (KNN) with `n_neighbors=5`. This method takes the missing value, evaluates all of the other variables aside from it and then approximates the missing value based on 5 nearest matches from other regions.

### Normalization:
- Numerical data is scaled using `MinMaxScaler` to bring all values into the range [0, 1].

### Negative Indicators:
- Several columns representing negative indicators are adjusted so that higher values represent worse conditions. Namely, the following parameters have been concluded to negatively effect a region's rating:

- **air_pollution_index**: A metric indicating the level of air pollution
- **clearance_rate**: Rate at which crimes are solved
- **crime_amount**: Total number of crimes
- **household_spending_per_month**: Average household spending per month
- **unemployment_rate**: Unemployment rate in the city

### Positive Indicators:
- The rest of the columns that represent positive indicators are as follows:

- **Region**: The name of the region or city.
- **avg_salary**: Average salary of the residents in the region.
- **Total_city_population**: Total population of the region.
- **gdp_per_capita_tenge**: Gross Domestic Product (GDP) per capita in Tenge.
- **Postgraduate_Education**: Proportion of the population with postgraduate education.
- **Higher_Education**: Proportion of the population with higher education.
- **life_expectancy**: Average life expectancy of the residents.
- **med_institutions_amount**: Number of medical institutions in the region.
- **number_of_schools**: Total number of schools in the region.
- **public_transport_quantity**: Quantity of public transportation available in the region.
- **number of entertainment places**: Number of entertainment venues in the region.
- **Total_Score**: An overall score or index derived from various metrics to assess the region's quality or performance.

### Scoring:
- A total score for each city is calculated by summing the normalized values of the selected columns.
- Cities are then ranked based on their total scores.

### Clustering:
- KMeans clustering is performed on the normalized data (excluding the region and total score columns) to categorize cities into clusters based on their overall characteristics.
- PCA (Principal Component Analysis) is used to reduce the dimensions of the data for visualization purposes. This helps to visualize the clusters in a 2D space.
- Three clusters are used, each representing different levels of life quality: average, higher, and lower.


### Visualization:
- First, a horizontal bar plot is created to display the scores of each city:

![image](https://github.com/open-data-kazakhstan/city-ranking-main/assets/109875855/cc425d58-768c-465d-af19-e50e0ba04fbc)

- Then, the 'Total score' column is scaled again to represent values from 0 to 100, and a second graph is made for better clarity.

![image](https://github.com/open-data-kazakhstan/city-ranking-main/assets/109875855/475546b4-d39c-4511-b14a-2d9e55b87e6c)

- After that, a PCA plot is created to visualize the clusters formed by KMeans. Each city is represented in a 2D space, colored according to its cluster:

![image](https://github.com/open-data-kazakhstan/city-ranking-main/assets/109875855/1fcbdd0a-f0b2-4015-b608-26c11158fb56)

- Finally, a bar chart is created where each city's bar is colored according to its cluster, providing a visual representation of both the total score and the cluster classification:

![image](https://github.com/open-data-kazakhstan/city-ranking-main/assets/109875855/623fd685-9617-448f-a19e-09b417fa279d)


## Results

The script outputs a DataFrame showing the regions and their total scores, sorted in descending order. A bar plot is also generated to visually represent the rankings.

## License

This dataset is licensed under the Open Data Commons [Public Domain and Dedication License][pddl].

[pddl]: https://www.opendatacommons.org/licenses/pddl/1-0/
