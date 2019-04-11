#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Scrape data from wikipedia url using read html


# # Use read_html

# In[2]:


import requests
import re


# In[3]:


import pandas as pd
df2 = pd.DataFrame()
df2 = pd.read_html('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M')


# In[4]:


type(df2)
type(df2[1])


# In[5]:


df = df2[0]


# In[6]:


df = df.rename(columns=df.iloc[0]).drop(df.index[0])


# In[7]:


df


# # Drop rows where Borough == 'Not assigned'

# In[8]:


df2 = df.drop(df[(df['Borough'] == 'Not assigned')].index)
len(df2)


# # Assign Borough value to Neighbourhood value for rows with Neighbourhood == 'Not assigned

# In[9]:


df2.loc[df['Neighbourhood'] == 'Not assigned', 'Neighbourhood'] = df2['Borough']
df2


# # Merge repeating postal codes and have Neighbourhood values assigned as one separated by a comma(s)

# In[10]:


df3 = df2.groupby(['Postcode','Borough'])['Neighbourhood'].apply(', '.join).reset_index()


# In[11]:


df3


# In[12]:


df3.shape


# In[13]:


import pandas as pd
import numpy as np


# In[14]:


data = pd.read_csv("https://cocl.us/Geospatial_data")
data


# In[15]:


#Rename column Postal Code to Postcode so we can merge this data frame with the 
#previous data frame
data.rename(columns={'Postal Code': 'Postcode'}, inplace=True)
data


# In[16]:


df4 = pd.merge(df3, data, on='Postcode')
df4


# # Begin Analysis of Neighborhoods in Toronto

# In[17]:


df4['Borough'].unique()


# ## Filter data based on substring Toronto to get data frame of only the boroughs in Toronto

# In[18]:


toronto_data = df4[df4['Borough'].str.contains("Toronto")]
toronto_data


# In[19]:


mlatitude = (toronto_data['Latitude']).mean()
mlongitude = (toronto_data['Longitude']).mean()


# # Create map of Toronto

# In[20]:


import folium
map_toronto = folium.Map(location = [mlatitude,mlongitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(toronto_data['Latitude'], toronto_data['Longitude'], toronto_data['Neighbourhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# # Define Foursquare Credential Variables

# In[21]:


CLIENT_ID = 'SCUVPBOLN02KP2IAVPPN4QCH5H0VQVXQ4TOBZGQ5PNTH5NBA' # your Foursquare ID
CLIENT_SECRET = 'R3V4JBEWJDFBT0FYNWQG5ZLTGCEYMMAK4ISDGWKIADBODLR1' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# # Let's look at the venues in The Beaches neighborhood
# # Define latitude, longitude, and name of the first neighborhood: The Beaches

# In[22]:


nlatitude = toronto_data['Latitude'].iloc[0]
nlongitude = toronto_data['Longitude'].iloc[0]
name = toronto_data['Neighbourhood'].iloc[0]

print('Latitude and longitude values of {} are {}, {}.'.format(nlatitude, 
                                                               nlongitude, 
                                                               name))


# In[92]:


#Define Limit and radius
#We are looking at 100 venus within a 500 mile radius of The Beaches
LIMIT = 100
radius = 500


# In[24]:


#Define url from foursquare api
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    nlatitude, 
    nlongitude, 
    radius, 
    LIMIT)
url # display URL


# In[93]:


#Define results so we can start finding our venues
results = requests.get(url).json()
results


# In[26]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[28]:


from pandas.io.json import json_normalize
import requests

venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[29]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# In[30]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()['response']['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[31]:


toronto_venues = getNearbyVenues(names=toronto_data['Neighbourhood'],
                                   latitudes=toronto_data['Latitude'],
                                   longitudes=toronto_data['Longitude']
                                  )


# In[32]:


print(toronto_venues.shape)
toronto_venues.head()


# In[33]:


toronto_venues.groupby('Neighborhood').count()


# In[34]:


print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# In[35]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# In[36]:


toronto_onehot.shape


# In[37]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# In[38]:


toronto_grouped.shape


# In[39]:


num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[40]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[41]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted


# In[94]:


import sklearn
from sklearn.cluster import KMeans
import numpy as np

# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[90]:


# add clustering labels
#neighborhoods_venues_sorted['(0, 'Cluster Labels', kmeans.labels_)']

toronto_merged = toronto_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighbourhood')

toronto_merged = toronto_merged.drop('College Town', 1)
toronto_merged
# check the last columns!


# In[71]:


import matplotlib.colors as colors
import matplotlib.cm as cm


# In[72]:


# create map
map_clusters = folium.Map(location=[mlatitude, mlongitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# # So now we have a map of the neighborhoods in Toronto split into clusters based on similar frequency of venues that appear in each neighborhood

# In[73]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[74]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[75]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[76]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[77]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[ ]:




