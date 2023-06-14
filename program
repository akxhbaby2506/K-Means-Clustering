import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from google.colab import files
uploaded = files.upload()
customer_data = pd.read_csv(io.BytesIO(uploaded['Mall_Customers.csv']))
print (customer_data.info())

customer_data.head()

customer_data.describe()

customer_data.shape

customer_data.isnull().sum()

X=customer_data.iloc[:,[3,4]].values
print(X)

WCSS=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
  kmeans.fit(X)
WCSS.append(kmeans.inertia_)
  
sns.set()
plt.plot(range(1,11),WCSS)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
#return a label for each data point based on their cluster
Y=kmeans.fit_predict(X)
print(Y)

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0],X[Y==0,1],s=50,c='green',label='Cluster 1')
plt.scatter(X[Y==1,0],X[Y==1,1],s=50,c='red',label='Cluster 2')
plt.scatter(X[Y==2,0],X[Y==2,1],s=50,c='yellow',label='Cluster 3')
plt.scatter(X[Y==3,0],X[Y==3,1],s=50,c='violet',label='Cluster 4')
plt.scatter(X[Y==4,0],X[Y==4,1],s=50,c='blue',label='Cluster 5')
#plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='cyan',label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')

# Fit KMeans with 5 clusters
kmeans = KMeans(n_clusters=5)
kmeans.fit(customer_data[['Annual Income (k$)', 'Spending Score (1-100)']])
labels = kmeans.labels_
# Add the cluster labels to the dataset
customer_data['Cluster'] = labels
# Count the number of data points in each cluster
cluster_counts = customer_data['Cluster'].value_counts()
# Create a bar graph of cluster counts
plt.bar(cluster_counts.index, cluster_counts.values)
plt.title('K-Means Clustering Results')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.xticks(cluster_counts.index)
plt.show()

age_groups = ['18-29', '30-39', '40-49', '50-59', '60-70']
# Group the data by age group and compute the mean spending score for each group
grouped_data = customer_data.groupby(pd.cut(customer_data['Age'], bins=[18, 29, 39, 49, 59, 70], labels=age_groups))['Spending Score (1-100)'].mean()
# Plot the stacked bar graph
plt.bar(age_groups, grouped_data, label='Spending Score', color='blue')
plt.xlabel('Age Group')
plt.ylabel('Average Spending Score')
plt.title('Average Spending Score by Age Group')
plt.legend()
plt.show()
