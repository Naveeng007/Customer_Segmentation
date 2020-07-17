from datetime import datetime,timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


tx_data=pd.read_csv('OnlineRetail.csv',encoding = "ISO-8859-1")#using to handle presence of special latin characters link:https://mincong.io/2019/04/07/understanding-iso-8859-1-and-utf-8/#:~:text=ISO%208859%2D1%20encodes%20what,romanizations%20of%20East%2DAsian%20languages.


tx_data['InvoiceDate']=pd.to_datetime(tx_data['InvoiceDate'])
tx_uk=tx_data.query("Country=='United Kingdom'").reset_index(drop=True)

###########calculating recency##########

tx_user=pd.DataFrame(tx_data['CustomerID'].unique())#only extracting unique customers
tx_user.columns=['CustomerID']#assigning tx_user  a column name

#get the max purchase date for each customer and create a dataframe with it
tx_max_purchase=tx_uk.groupby('CustomerID').InvoiceDate.max().reset_index()#finding last date when customer did transaction. with reset index we assigning new index to new dataframe
tx_max_purchase.columns=['CustomerID','MaxPurchaseDate']#assigning columns as customer ID and last purchase data

#we take our observation point as the max invoice date in our dataset
tx_max_purchase['Recency']=(tx_max_purchase['MaxPurchaseDate'].max()-tx_max_purchase['MaxPurchaseDate']).dt.days #maximum of all present data - maximum of particular customer will give recency value

#merge this dataframe to our new dataframe
tx_user=pd.merge(tx_user,tx_max_purchase[['CustomerID','Recency']],on='CustomerID')#permorming a join operation on customerID and then assigning column names in tx_user

#drawing Histogram
plt.title('Recency')
plt.xlabel('Recency Customers(bin)')
plt.ylabel('No of  Customers(frequency)')
plt.hist(tx_user['Recency'])
plt.show()

tx_user.Recency.describe()#will give info about data such as mean ,std etc

#finding best clusters
X=tx_user[['Recency']]
wcss=[]
for i in range(1,11):
    from sklearn.cluster import KMeans
    Kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)#for same cluster the value must be reproducible hence random_state is used. Kmeans++ is advance form of kmeans whhich uses some algo to assing inital centroid value
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)#storing goodness of fit values i.e sigma(1 to n) sum of sq of with every element of cluster with its centroid
plt.plot(range(1,11),wcss)
plt.show()
plt.title('The Elbow Method')#for checking how many cluster we must have for better results
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#from this graph comes out to be around (4 or 3) so we will take 4

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

########################for ordering clusters ################################
def order_cluster(cluster_field_name,target_field_name,df,ascending):#basicall it is replacing old cluster name which was randomly assign to any group so this method will replace with new as, 0 will least value and 2 will be most 
    
    df_new=df.groupby(cluster_field_name)[target_field_name].mean().reset_index()#finding the mean of cluster_feild name then reseting index link:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html
    df_new=df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)#after sorting target values either des or asc depends on bool value then reseting index and dropping old index
    df_new['index']=df_new.index
    df_final=pd.merge(df,df_new[[cluster_field_name,'index']],on=cluster_field_name)#performing join operation on cluster_field_name and assigning new column name as index
    df_final=df_final.drop([cluster_field_name],axis=1)#dropping the cluster_field_name as it will be replaced by new cluster field 
    df_final=df_final.rename(columns={"index":cluster_field_name})#replacing index value to cluster_field name
    return df_final

#calclating frequency of each customers in the databse for the store
tx_frequency = tx_uk.groupby('CustomerID').InvoiceDate.count().reset_index()#by counting number of date customer visited will give frequency
tx_frequency.columns = ['CustomerID','Frequency']#assigning column name

###order clusters in descending order of recency
#THEN
tx_user.groupby('RecencyCluster')['Recency'].describe()
tx_user=order_cluster('RecencyCluster','Recency',tx_user,False)#here it will sort in descending order as false has been sent
#NOW
tx_user.groupby('RecencyCluster')['Recency'].describe()
#add this data to our main dataframe
tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')

#drawing histogram for frequency of customers visted store 
x=tx_user.query('Frequency<1000')['Frequency']
plt.xlabel('Number of customers')
plt.ylabel('Having Frequency')
plt.title('Number of Users having Frequency is less than 1000')
plt.hist(x)
plt.show()

Kmeans=KMeans(n_clusters=4)
Kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster']=Kmeans.predict(tx_user[['Frequency']])

#THEN
tx_user.groupby('FrequencyCluster')['Frequency'].describe()
####order frequency in order of ascending order

tx_user=order_cluster('FrequencyCluster','Frequency',tx_user,True)
#Now
tx_user.groupby('FrequencyCluster')['Frequency'].describe()

#Now turn for Revenue
tx_uk['Revenue'] = tx_uk['UnitPrice'] * tx_uk['Quantity']
tx_revenue = tx_uk.groupby('CustomerID').Revenue.sum().reset_index()

#merge it with our main dataframe
tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')
#turn for freq histogram
xr=tx_user.query('Revenue<10000')['Revenue']
plt.title('Revenue')
plt.xlabel('Money Spend less than 1000 ')
plt.ylabel('Number of Customers spend')
plt.hist(xr)
plt.show()
#clustering freq
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Revenue']])
tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
y_kmeans=tx_user['RevenueCluster']

#THEN
tx_user.groupby('RevenueCluster')['Revenue'].describe()
#order revenue in order of ascending order
tx_user=order_cluster('RevenueCluster','Revenue',tx_user,True)
#NOW
tx_user.groupby('RevenueCluster')['Revenue'].describe()#grouping by revenuecluster then corresponding to that finding revenue mean


tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
tx_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()

tx_user['Segment'] = 'Low-Value'#if value less than 2 marked as low value
tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value'


plt.xlabel('Frequency')
plt.ylabel('Revenue')
tx_graph=tx_user.query('Revenue<50000 and Frequency<2000')
plt.scatter(tx_graph.query("Segment == 'Low-Value'")['Frequency'],tx_graph.query("Segment == 'Low-Value'")['Revenue'], s = 100, c = 'red', label = 'Cluster 1')#applying query to find segments then ['recency'] means selecting only this column, s=100 means scatter size 
plt.scatter(tx_graph.query("Segment == 'Mid-Value'")['Frequency'],tx_graph.query("Segment == 'Mid-Value'")['Revenue'], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(tx_graph.query("Segment == 'High-Value'")['Frequency'],tx_graph.query("Segment == 'High-Value'")['Revenue'], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(tx_graph.query("Segment == 'Low-Value'")['Recency'],tx_graph.query("Segment == 'Low-Value'")['Revenue'], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(tx_graph.query("Segment == 'Mid-Value'")['Recency'],tx_graph.query("Segment == 'Mid-Value'")['Revenue'], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(tx_graph.query("Segment == 'High-Value'")['Recency'],tx_graph.query("Segment == 'High-Value'")['Revenue'], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(tx_graph.query("Segment == 'Low-Value'")['Frequency'],tx_graph.query("Segment == 'Low-Value'")['Recency'], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(tx_graph.query("Segment == 'Mid-Value'")['Frequency'],tx_graph.query("Segment == 'Mid-Value'")['Recency'], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(tx_graph.query("Segment == 'High-Value'")['Frequency'],tx_graph.query("Segment == 'High-Value'")['Recency'], s = 100, c = 'green', label = 'Cluster 3')

