#######################################
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

# values that are zero are discarded
rfm = rfm[rfm["Monetary"] > 0]

#######################
# 5. Calculating RFM Scores
#######################

# The lowest recency value is the most valuable
# So in order from largest to smallest
rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels = [5, 4, 3, 2, 1])

# Higher frequency indicates more frequent shopping.
# Sort from smallest to largest
# If the same quarter is still observed when going to different quarters after sorting, mehtod first is used because this causes a problem.
# duplicates -> gives value error
rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels = [1,2,3,4,5])

# A high Monetary value indicates that the total amount paid is high.
# Sort from smallest to largest
rfm["monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels = [1,2,3,4,5])

rfm["RFM_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

#######################
# 6. Creating & Analysing RFM Segments
#######################

# Defining RFM scores as segments
segment_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At_Risk',
    r'[1-2]5': 'Cant_Loose',
    r'3[1-2]': 'About_to_Sleep',
    r'33': 'Need_Attention',
    r'[3-4][4-5]': 'Loyal_Customers',
    r'41': 'Promising',
    r'51': 'New_Customers',
    r'[4-5][2-3]': 'Potential_Loyalists',
    r'5[4-5]': 'Champions'
}

# Regular expression: r'[1-2]->recency [1-2]->frequency => r'[1-2][1-2] => gives naming according to values
rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex = True)

# The metrics created at the beginning are selected, not the scores
rfm[["segment", "Recency", "Frequency", "Monetary"]].groupby("segment").agg(["mean", "count"])

"""
                         Recency       Frequency       Monetary      
                       mean count      mean count     mean count
segment                                                         
Hibernating          218.90  1065      1.10  1065   487.71  1065
about_to_sleep        54.50   351      1.16   351   461.06   351
at_Risk              156.06   580      2.87   580  1076.51   580
cant_loose           133.43    63      8.38    63  2796.16    63
champions              6.88   633     12.42   633  6857.96   633
loyal_customers       34.47   827      6.46   827  2856.72   827
need_attention        54.06   186      2.33   186   889.23   186
new_customers          7.86    42      1.00    42   388.21    42
potential_loyalists   18.12   492      2.01   492  1034.91   492
promising             24.44    99      1.00    99   355.35    99
"""

################################
# 5. K-Means
################################

# Dataframe is reloaded without segmentation in here.

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

# !pip install -U yellowbrick

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

kmeans = KMeans(n_clusters=10).fit(df)

clusters_ = kmeans.labels_ # cluster numbers

pd.DataFrame({"States": df.index, "clusters_": clusters_})

rfm["cluster_no"] = clusters_
# cluster_ids are set to start from 1
rfm["cluster_no"] = rfm["cluster_no"] + 1
rfm.head(10)

# Grouping by cluster recency, frequency, monetary
rfm.groupby("cluster_no").agg(['mean', "count"]).reset_index()

"""
cluster_no Recency       Frequency       Monetary      
                mean count      mean count     mean count
0          1    6.77   571     13.33   571  7852.57   571
1          2  107.93   399      3.82   399  1576.20   399
2          3   19.19   291      1.24   291   259.58   291
3          4  279.33   609      1.06   609   246.45   609
4          5   33.84   664      7.13   664  3456.84   664
5          6   17.41   487      2.76   487   758.60   487
6          7   68.63   348      2.46   348   474.55   348
7          8   86.28   238      1.36   238  1037.70   238
8          9   96.05   483      1.11   483   228.37   483
9         10  247.47   248      2.42   248  1555.98   248
"""
################################
# 9. Comparison of Scores for RFM Divided by KMeans and Segments
################################

# SEGMENT
"""
                   Recency       Frequency       Monetary      
                       mean count      mean count     mean count
segment                                                         
Hibernating          218.90  1065      1.10  1065   487.71  1065
about_to_sleep        54.50   351      1.16   351   461.06   351
at_Risk              156.06   580      2.87   580  1076.51   580
cant_loose           133.43    63      8.38    63  2796.16    63
champions              6.88   633     12.42   633  6857.96   633
loyal_customers       34.47   827      6.46   827  2856.72   827
need_attention        54.06   186      2.33   186   889.23   186
new_customers          7.86    42      1.00    42   388.21    42
potential_loyalists   18.12   492      2.01   492  1034.91   492
promising             24.44    99      1.00    99   355.35    99
"""

# K-MEANS

"""
 cluster_no       Recency       Frequency       Monetary      
                mean count      mean count     mean count
0          1    6.77   571     13.33   571  7852.57   571
1          2  107.93   399      3.82   399  1576.20   399
2          3   19.19   291      1.24   291   259.58   291
3          4  279.33   609      1.06   609   246.45   609
4          5   33.84   664      7.13   664  3456.84   664
5          6   17.41   487      2.76   487   758.60   487
6          7   68.63   348      2.46   348   474.55   348
7          8   86.28   238      1.36   238  1037.70   238
8          9   96.05   483      1.11   483   228.37   483
9         10  247.47   248      2.42   248  1555.98   248
"""


"""
Result: While performing segmentation in RFM, evaluation was made 
by considering recency and frequency values. While doing K-means, 
calculations were made by considering all the variables. In addition, 
clusters showed a more regular distribution in k_mean.
"""

