########################################
# RFM Customer Segmentation Using K-Means
#######################################

# 1. Import & Reading
# 2. Data Understanding
# 3. Data Preparation
# 4. Calculating RFM Metrics
# 5. K-Means
# 6. Visualization of Clusters
# 7. Determining the Optimum Number of Clusters
# 8. Creating Final Clusters
# 9. Comparison of Scores for RFM Divided by KMeans and Segments

#######################
# 1. Import & Reading
#######################

########### IMOPORT LIBRARIES ###########
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)  # show max rows in rows
pd.set_option('display.max_row', None)  # show max column in columns
pd.set_option('display.float_format', lambda x: ' %.2f' % x)  # show 2 numbers after the comma

########### READING THE DATASET ###########
df_ = pd.read_excel('datasets/csv_path/w3/online_retail_II.xlsx',
                    sheet_name="Year 2010-2011")  # 2010-2011 page is taken

# Data is copied against data corruption
df = df_.copy()

#######################
# 2. Data Understanding
#######################

# Are there any missing observations in the dataset? If so, how many missing observations are there?
df.isnull().sum()

# Remove the missing observations from the data set.
df = df.dropna()

#######################
# 3. Data Preparation
#######################

# Descriptive statistics of the dataset
df.describe().T

# Unique number of StockCode
df["StockCode"].nunique()

# How many of each product
df["StockCode"].value_counts()

# Sorting from the 5 most ordered products to the lowest
df.groupby("StockCode").agg({"Quantity": "sum"}).sort_values(by='Quantity', ascending=False).head()

# The 'C' in the invoice shows the canceled transactions. Canceled transactions leave the dataset.
df = df[~df["Invoice"].astype(str).str.contains('C', na=False)]  # na (blank observation) means disregard

# Earnings per invoice
df["Total Price"] = df["Quantity"] * df["Price"]

#######################
# 4. RFM Metrics
#######################

"""
Recency: Time from customer's last purchase to date

Frequency: Total number of purchases

Monetary: The total spend by the customer
"""

import time
from datetime import datetime, timedelta
# Recency account is selected after 2 days to avoid problems.
today_date = df['InvoiceDate'].max() + timedelta(days=2)

# Recency, frequency and monetary values were calculated
rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today_date - date.max()).days,
                                     # last purchase - today's purchase and .days -> returns the format in days
                                     "Invoice": lambda num: num.nunique(),  # nunique -> number of unique invoices
                                     "Total Price": lambda total_price: total_price.sum()})

# Column names changed
rfm.columns = ["Recency", "Frequency", "Monetary"]

################################
# 5. K-Means
################################

# Convert data between 0-1 with MinMaxScaler to avoid measurement problems.
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(rfm)
df[0:5]

kmeans = KMeans()
k_fit = kmeans.fit(df)

k_fit.get_params()
# max_iter = repeats for a certain number of iterations (eg new centers are determined)
# The location with the lowest SSE point becomes the focal point

k_fit.n_clusters
k_fit.cluster_centers_ # Center of 8 different clusters
k_fit.labels_ # Labels belonging to 8 clusters
k_fit.inertia_ # total error value # total SSE value

################################
# 6. Visualization of Clusters
################################

k_means = KMeans(n_clusters=2).fit(df)
clusters_ = k_means.labels_
type(df)
df = pd.DataFrame(df)

plt.scatter(df.iloc[:, 0], # get whole row get variable at 0.index
            df.iloc[:, 1], # get whole row get variable in 1st index
            c=clusters_,
            s=50,
            cmap="viridis")
plt.show()

# marking of centers
centers_ = k_means.cluster_centers_

plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=clusters_,
            s=50,
            cmap="viridis")

plt.scatter(centers_[:, 0],
            centers_[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()

# In order to visualize the data set in 2 dimensions, 2 variables must be selected.
# If it is desired to be done considering all the variables: PCA is done.
# eg there are 40 dimensions Reduce size with PCA, reduce to two prime components after reduction

################################
# 7. Determining the Optimum Number of Clusters
################################

# model object
kmeans = KMeans()

# sum of square distance
ssd = []

K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df) # fit kmeans per k
    ssd.append(kmeans.inertia_) # get the total error value of kmeans

ssd # sum of each of the sets' errors
# 2 clusters, 3 clusters... total
# errors are over standardized values.


# The number of clusters is determined by the ELbov method.
# Elbov method elbow method
# 5 points where errors change the most can be selected
plt.plot(K, ssd, "bx-")
plt.xlabel("Distance Residual Sums Against Different K Values")
plt.title("Elbow Method for Optimum Number of Clusters")
plt.show()

# is used when you want to look at it mathematically
# green train times
# distortion score = sse
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k=(2, 20))
visu.fit(df)
visu.show();

elbow.elbow_value_ # 6

################################
# 8. Creating Final Clusters
################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

clusters_ = kmeans.labels_ # cluster numbers

pd.DataFrame({"States": df.index, "clusters_": clusters_})

rfm["cluster_no"] = clusters_
# cluster_ids are set to start from 1
rfm["cluster_no"] = df["cluster_no"] + 1
rfm.head(10)

################################
# 9. Comparison of Scores for RFM Divided by KMeans and Segments
################################

"""
             Recency  Frequency  Monetary  cluster_no
Customer ID                                          
12346.00         327          1  77183.60           2
12347.00           3          7   4310.00           1
12348.00          76          4   1797.24           4
12349.00          20          1   1757.55           1
12350.00         311          1    334.40           2
12352.00          37          8   2506.04           1
12353.00         205          1     89.00           3
12354.00         233          1   1079.40           3
12355.00         215          1    459.40           3
12356.00          24          3   2811.43           1
"""

"""
           monetory_score RFM_SCORE              segment  
Customer ID                                                
12346.00                 5        11           hibreating  
12347.00                 5        55            champions  
12348.00                 4        24              at_Risk  
12349.00                 4        41            promising  
12350.00                 2        11           hibreating  
12352.00                 5        35      loyal_customers  
12353.00                 1        11           hibreating  
12354.00                 4        11           hibreating  
12355.00                 2        11           hibreating  
12356.00                 5        43  potential_loyalists  

"""






