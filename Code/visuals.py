

#%% regression analysis
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import rgb2hex


# Define the color scheme
fhnw_colour = (252/255, 230/255, 14/255)


#rgb colours
rgb_colors = [
    (0.9882352941176471, 0.9019607843137255, 0.054901960784313725), # Original Yellow
    (0.7905882352941177, 0.7215686274509805, 0.04392156862745098),
    (0.5929411764705882, 0.5411764705882353, 0.03294117647058823),
    (0.39529411764705885, 0.36078431372549025, 0.02196078431372549),
    (0.19764705882352937, 0.18039215686274507, 0.010980392156862742),
    (0.0, 0.0, 0.0) # Black
]

# hex colors
hex_colors = [rgb2hex(color) for color in rgb_colors]

# Create a colormap
fhnw_colourmap = LinearSegmentedColormap.from_list("custom_colormap", rgb_colors)



#%%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Convert the RGBA values from 0-255 range to 0-1 range
rgba_color = (252/255, 230/255, 14/255, 255/255)


# Set the time period
start_year = 2000
end_year = 2020
dates = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='D')

# Yearly Trend Component
yearly_trend = np.linspace(0, 1, len(dates)) * 5

# Seasonal Component
seasonal_component = np.sin(np.linspace(0, 2 * np.pi, len(dates)))

# Random Component
random_component = np.random.normal(0, 0.5, len(dates))

# Introduce specific events
# Dot-com Bubble Burst in 2001
dot_com_bubble = ((dates.year == 2001) & (dates.month >= 3)) * -2

# Financial Crisis in 2008
financial_crisis = ((dates.year == 2008) & ((dates.month >= 9) & (dates.month <= 12))) * -3

# COVID-19 Pandemic Early 2020
covid_pandemic = ((dates.year == 2020) & (dates.month <= 6)) * -4

# Combine all components with the events
time_series = yearly_trend + seasonal_component + random_component + dot_com_bubble + financial_crisis + covid_pandemic

# Convert to a pandas DataFrame
time_series_df = pd.DataFrame(time_series, index=dates, columns=['Value'])


# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(time_series_df.index, time_series_df['Value'], color=rgba_color)
plt.title('Time Series with Yearly, Seasonal, Random Components, and Real-Life Events')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

#%%


# Creating a random time series data
np.random.seed(0)  # For reproducibility
n_points = 1000
x = np.linspace(0, 10, n_points)
y = np.sin(x) + np.random.normal(scale=0.5, size=n_points)  # Sine wave with noise

# Convert to DataFrame for convenience
data = pd.DataFrame({'x': x, 'y': y})

# Visualizing the original data
plt.figure(figsize=(14, 7))
plt.scatter(data['x'], data['y'], label='Original Data', color=rgb_colors[4], alpha=0.7)

# Splitting the data into train and test (here we use all data for simplicity in demonstration)
X = data['x'].values.reshape(-1, 1)
y = data['y'].values

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Polynomial Regression Model
poly_reg = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
poly_reg.fit(X, y)
y_pred_poly = poly_reg.predict(X)

# Random Forests Model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(X, y)
y_pred_rf = rf_reg.predict(X)

# Plotting the predictions
plt.plot(data['x'], y_pred_lin, label='Linear Regression Predictions', linewidth = 5, color=rgb_colors[1])
plt.plot(data['x'], y_pred_poly, label='Polynomial Regression Predictions', linewidth = 5, color=rgb_colors[2])
plt.plot(data['x'], y_pred_rf, label='Random Forest Predictions', linewidth = 1, color = rgb_colors[3])

# Annotations
plt.title('Comparison of Regression Models')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Print out the RMSE for each model
print(f"Linear Regression RMSE: {mean_squared_error(y, y_pred_lin, squared=False):.2f}")
print(f"Polynomial Regression RMSE: {mean_squared_error(y, y_pred_poly, squared=False):.2f}")
print(f"Random Forest Regression RMSE: {mean_squared_error(y, y_pred_rf, squared=False):.2f}")




#%% clustering

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generating synthetic data with 3 clusters
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Applying KMeans clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plotting the centroids
centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X')
plt.title('K-means Clustering Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#%%

import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from mpl_toolkits.mplot3d import Axes3D

# Generating a Swiss roll dataset
X, _ = make_swiss_roll(n_samples=800, noise=0.5, random_state=0)

# Clustering with different algorithms
# 1. KMeans clustering
kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(X)

# 2. Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
y_agglo = agglo.fit_predict(X)

# 3. DBSCAN clustering
dbscan = DBSCAN(eps=1.5, min_samples=10)
y_dbscan = dbscan.fit_predict(X)

# Plotting the results
fig = plt.figure(figsize=(18, 5))

# KMeans
ax = fig.add_subplot(131, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_kmeans, cmap='viridis')
ax.set_title('KMeans Clustering')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Agglomerative Clustering
ax = fig.add_subplot(132, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_agglo, cmap='viridis')
ax.set_title('Agglomerative Clustering')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# DBSCAN
ax = fig.add_subplot(133, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_dbscan, cmap='viridis')
ax.set_title('DBSCAN Clustering')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()
plt.show()

#%%


import numpy as np
import matplotlib.pyplot as plt

# Generate random sample observations
np.random.seed(0)
observations = np.random.rand(10, 2)  # 10 observations in 2D

# Choose a reference point (first point in this case)
reference_point = observations[0]

# Calculate distances from the reference point to all other points
distances = np.sqrt(np.sum((observations - reference_point)**2, axis=1))

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(observations[:, 0], observations[:, 1], color='blue', label='Observations')
plt.scatter(reference_point[0], reference_point[1], color='red', label='Reference Point')

# Draw lines from the reference point to other points
for point, distance in zip(observations, distances):
    plt.plot([reference_point[0], point[0]], [reference_point[1], point[1]], 'k--', alpha=0.5)
    plt.text((reference_point[0] + point[0]) / 2, (reference_point[1] + point[1]) / 2, 
             f'{distance:.2f}', color='green')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Distances from Reference Point to Other Points')
plt.legend()
plt.grid(True)
plt.show()


#%% 

from fhnw_colourmap import fhnw_colourmap

# Visualizing the original data with the new color scheme
plt.figure(figsize=(14, 7))
plt.scatter(data['x'], data['y'], label='Original Data', color=fhnw_colour, alpha=0.7)

# Splitting the data into train and test (here we use all data for simplicity in demonstration)
X = data['x'].values.reshape(-1, 1)
y = data['y'].values

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Polynomial Regression Model
poly_reg = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
poly_reg.fit(X, y)
y_pred_poly = poly_reg.predict(X)

# Random Forests Model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(X, y)
y_pred_rf = rf_reg.predict(X)

# Plotting the predictions using the custom colormap
plt.plot(data['x'], y_pred_lin, label='Linear Regression Predictions', color=rgb_colors[1])
plt.plot(data['x'], y_pred_poly, label='Polynomial Regression Predictions', color=rgb_colors[3])
plt.plot(data['x'], y_pred_rf, label='Random Forest Predictions', color=rgb_colors[5])

# Annotations
plt.title('Comparison of Regression Models')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Print out the RMSE for each model
rmse_lin = mean_squared_error(y, y_pred_lin, squared=False)
rmse_poly = mean_squared_error(y, y_pred_poly, squared=False)
rmse_rf = mean_squared_error(y, y_pred_rf, squared=False)

rmse_lin, rmse_poly, rmse_rf

#%% Benfords law

import pandas as pd

# Benford's Law Expected Distribution
digits = range(1, 10)
benford_law_probs = [np.log10(1 + 1/d) for d in digits]

# Generate a sample real-world dataset (e.g., random financial transaction values)
np.random.seed(0)
sample_data = np.random.uniform(1, 1000, 1000)  # 1000 transactions between $1 and $1000

# Extracting the first digit from each number in the dataset
first_digits = [int(str(number)[0]) for number in sample_data]

# Actual distribution of first digits
actual_distribution = [first_digits.count(digit) / len(sample_data) for digit in digits]

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(digits, benford_law_probs, width=0.4, align='center', alpha=0.6, label='Benford\'s Law')
plt.bar(np.array(digits) + 0.4, actual_distribution, width=0.4, align='center', alpha=0.6, color='orange', label='Actual Distribution')

plt.xlabel('First Digit')
plt.ylabel('Proportion')
plt.xticks(digits)
plt.title('Comparison of Benford\'s Law with Actual Distribution of First Digits')
plt.legend()
plt.show()

#%% swiss roll

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import TSNE

# Generate a Swiss roll dataset
X, color = make_swiss_roll(n_samples=1000)

# Use t-SNE to project the Swiss roll to 2D
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
X_2d = tsne.fit_transform(X)

# Plot
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=color, cmap=fhnw_colourmap)
plt.title("2D projection of a Swiss Roll")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

#%%

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with 4 clusters
X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.2, random_state=40)

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap= fhnw_colourmap)
plt.title('Scatter Plot with 4 Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
# plt.grid(True)
plt.show()

#%%
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np

# Step 1: Generate Synthetic Dataset
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Step 2: Apply K-means Clustering
kmeans = KMeans(n_clusters=4, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Step 3: Apply Agglomerative Hierarchical Clustering
agg_cluster = AgglomerativeClustering(n_clusters=4)
y_agg = agg_cluster.fit_predict(X)

# Custom function for Polythetic Divisive Hierarchical Clustering
def divisive_clustering(X, n_clusters=4):
    # Initially treat the entire dataset as one cluster
    clusters = [(X, range(X.shape[0]))]  # Store both data and indices
    labels = np.zeros(X.shape[0], dtype=int)

    while len(clusters) < n_clusters:
        # Find the largest cluster to split
        largest_cluster_index = np.argmax([len(cluster[0]) for cluster in clusters])
        cluster_to_split, indices_to_split = clusters.pop(largest_cluster_index)

        # Apply K-means with k=2 to split the largest cluster
        kmeans = KMeans(n_clusters=2, random_state=0).fit(cluster_to_split)
        split_labels = kmeans.labels_

        # Split the cluster into two based on the labels
        cluster1 = cluster_to_split[split_labels == 0]
        cluster2 = cluster_to_split[split_labels == 1]

        indices1 = [indices_to_split[i] for i in range(len(split_labels)) if split_labels[i] == 0]
        indices2 = [indices_to_split[i] for i in range(len(split_labels)) if split_labels[i] == 1]

        # Add the new clusters back to the list
        clusters.append((cluster1, indices1))
        clusters.append((cluster2, indices2))

        # Update labels
        for index in indices1:
            labels[index] = len(clusters) - 2
        for index in indices2:
            labels[index] = len(clusters) - 1

    return labels


# Step 4: Apply Divisive Clustering
y_divisive = divisive_clustering(X, n_clusters=4)

# Step 5: Plotting the Results
plt.figure(figsize=(15, 5))

# Plotting K-means Clustering
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.title("K-means Clustering")

# Plotting Agglomerative Hierarchical Clustering
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_agg, s=50, cmap='viridis')
plt.title("Agglomerative Hierarchical Clustering")

# Plotting Divisive Hierarchical Clustering
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=y_divisive, s=50, cmap='viridis')
plt.title("Divisive Hierarchical Clustering")

plt.show()
