# Partition based,heirarchical based,density based
#Used for customer segmentation
# Import packages from sklearn
from sklearn.cluster import KMeans
import numpy as np

# Create array with numpy
X = np.array([[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]])

kmeans =  KMeans(n_clusters=2,random_state=0).fit(X)
kmeans.labels_

kmeans.predict([[0,0],[12,3]])
kmeans.cluster_centers_
