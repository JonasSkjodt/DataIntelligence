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

####
# try to visualize the number of sightings by shape
####

# .dropna drops the missing values
# .astype('category') is used to convert the column to a categorical type
# .value_counts() is used to count the number of sightings by shape
# .plot.bar() is used to plot the bar chart
# df["shape"].dropna().astype('category').value_counts().plot.bar()

# .fillna fills the missing values with unknown
# plot the visualization
# df["shape"].fillna("not sighted").astype("category").value_counts().plot.bar()

# Displays the plot
# plt.show()

####
#  world map test
####
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

# how long is the average ufo sighting // answer, around 2 and a half hours per sighting
# dfmask = df["duration (seconds)"].map(convert_float).fillna(0).mean()
# print(dfmask)


# how long is the average ufo sighting per country
################# Question #################
# we've done an .apply to get the mean of the duration of the sightings per country, but how can we use .map with .fillna to achieve the same thing

#dfmask = df["duration (seconds)"].map(convert_float).fillna(0)
# dfmaskdel = dfmask["country"].dropna()
# dfCountry = df.groupby("country")["duration (seconds)"].mean()
# print(dfCountry)
# wut = df.groupby("country")["duration (seconds)"].apply(lambda x: x.map(convert_float).mean())

# wut = df.groupby("country")["duration (seconds)"]
wut = df[["country", "duration (seconds)"]].corr()
# wut = df.groupby(["country"]).agg({"duration (seconds)": "mean"})
print(wut)

# the number of ufo sightings per country
# trend time per sighting
# trend ufo sightings over time
# when were the dates posted compared to the datetime of the sighting
