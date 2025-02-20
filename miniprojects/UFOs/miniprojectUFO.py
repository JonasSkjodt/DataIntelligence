import pandas as pd
import numpy as np

# visualization with matplotlib
import matplotlib.pyplot as plt

#testing matplotlib world map
from mpl_toolkits.basemap import Basemap

# low_memory=False to avoid warnings
df = pd.read_csv("ufo.csv", low_memory=False)

# get the 5 first rows (head), 5 last rows (tail) and the summary statistics (describe) of the dataset
head = df.head(5)
tail = df.tail(5)
describe = df.describe()

#print(head)

########## isualize the number of sightings by shape ##########

# plot the visualization
# df["shape"].fillna("not sighted").astype("category").value_counts().plot.bar()

# Displays the plot
# plt.show()

########## where are all the ufo sightings on a world map ##########

# what are the columns actually called??
# print(df.columns)

# # Remove the spaces from column names
# df.columns = df.columns.str.strip()

# # # there are errors in the dataset where some latitude and longitude values are strings (and should be numeric)
# # # Convert latitude and longitude to numeric, forcing errors to NaN cause wtf else to do
# df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
# df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# # # Drop rows with NaN values in latitude or longitude
# df = df.dropna(subset=['latitude', 'longitude'])

# # # Basemap instance
# m = Basemap(projection="merc", llcrnrlat=-60, urcrnrlat=85,
#             llcrnrlon=-180, urcrnrlon=180, resolution="c") # resolution of map can be set to c = crude, l = low, i = intermediate

# # # Draw map features
# m.drawcoastlines()
# m.drawcountries()
# m.fillcontinents(color="lightgray", lake_color="aqua")

# # # Convert latitude and longitude to map coordinates
# x, y = m(df["longitude"].values, df["latitude"].values)

# # # Plot the data points with a constant size
# m.scatter(x, y, s=2, c="red", alpha=0.5, zorder=5)

# # # Add a title
# plt.title("UFO on World Map")
# # # show the plot with matplotlib
# plt.show()

def convert_float(string):
    if string is None:
      return None
    try:
      return float(string)
    except ValueError:
      return None

# print(df.columns)

# Index(['datetime', 'city', 'state', 'country', 'shape', 'duration (seconds)',
#        'duration (hours/min)', 'comments', 'date posted', 'latitude',
#        'longitude '],

########## how long is the average ufo sighting // answer, around 2 and a half hours per sighting ##########
# dfmask = df["duration (seconds)"].map(convert_float).fillna(0).mean()
# print(dfmask)


########## how long is the average ufo sighting per country ##########

df["duration (seconds)"] = df["duration (seconds)"].map(convert_float).fillna(0)
# # # another way of doing it: avgTimeByCountry = df.groupby("country")["duration (seconds)"].apply(lambda x: x.map(convert_float).mean())
avgTimeByCountry = df.groupby("country")["duration (seconds)"].mean()
# print(avgTimeByCountry)

########## the number of ufo sightings per country ##########
# df["country"].value_counts().plot.bar()
# plt.show()

# how many sightings per country
# print(df["country"].fillna("unknown country").value_counts()) 

########## how the hell is gb avg time 66061 sec? GB greatly outnumbers the other countries in length of UFO sightings are there any outliers? ##########
# 
# df.groupby("country").idxmax("gb").plot.box()
pd.set_option('display.max_colwidth', None)
maskGB = df["country"] == "gb"
smalldf = df[maskGB]
print(smalldf["duration (seconds)"].max()) # 97836000.0 seconds (over 3 years) of 1 ufo sighting

#########################
# question:
# when you have so many different outliers (change the number below to see the different outliers) then, what do you do to get a "good" data result?
#########################

# smalldf[smalldf["duration (seconds)"] < 1000.0]["duration (seconds)"].plot.box()
# plt.show()

########## what is the comment to the highest duration sighting ##########
# 
# print(df[df["duration (seconds)"] == 97836000.0]) # full columns with values of that specific outlier
# print(df[df["duration (seconds)"] == 97836000.0]["comments"]) # only the comments of that specific outlier
df[df["duration (seconds)"] == 97836000.0]

########## trim the outliers from the data ##########
import scipy.stats
def trimmed_mean_for_gb():
    trim_mean = lambda x, *args: scipy.stats.trim_mean(x, 0.1)
    trim_mean.__name__ = "trim_mean"
    return smalldf.groupby("country").agg({"duration (seconds)": [trim_mean]})
  
print(trimmed_mean_for_gb())

########## trend time per sighting ##########

########## trend ufo sightings over time ##########

########## when were the dates posted compared to the datetime of the sighting ##########