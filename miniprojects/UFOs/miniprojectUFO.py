# When answering each question, make summary statistics on each.

# Summary statistics is a part of descriptive statistics that summarizes and provides the gist of
# information about the sample data

# When using summary statistics on the following answer, for instance,
# show the ufo sightings by shape.
# 1: make sure to answer the question based on the data
# 2: show the data as a screenshot
# 3: show the mean, median, standard deviation, visualizations, etc.

# This particular dataset has the following column names:
# Index(['datetime', 'city', 'state', 'country', 'shape', 'duration (seconds)',
#         'duration (hours/min)', 'comments', 'date posted', 'latitude',
#         'longitude '],

# logarithmic problem
# we encountered an issue with the data when converting some of the duration (seconds) to log, it became minus numbers
# This was only a problem with the data that was 0.0, since log(0) is undefined, and for instance, log(0.00001) is -11.5 ...
# The problem was in duration (seconds) for the country us.
# we made a variable called dfextra to handle that specific data

# Questions for next time
# can sources be cited in the exam report if common sense fails?
# How much are you able to investigate about the subject youre getting outside of the dataset?
# How long is the exam report actually? Can i get the trial exam or an exam report bitte
# we've done a normal boxplot, a logarithmic boxplot, how should we do one for percentage?

import pandas as pd
import numpy as np

# visualization with matplotlib
import matplotlib.pyplot as plt

#testing matplotlib world map
from mpl_toolkits.basemap import Basemap

# for logaritmic
import math
from math import log

# convert to float
def convert_float(string):
    if string is None:
      return None
    try:
      return float(string)
    except ValueError:
      return None

# check convert
def check_convert(string):
    if string is None:
      return False
    try:
      return True
    except ValueError:
      return False

# low_memory=False to avoid warnings
df = pd.read_csv("ufo.csv", low_memory=False)
# convert the duration (seconds) column to float and fill missing values with 0 (read comment below)
dfextra = df
df["duration (seconds)"] = df["duration (seconds)"].map(convert_float).fillna(0.00001) # it cant be 0.0, since that makes it impossible to log on it
# bug fixing why 0.0 didnt work
# print(df[(df["duration (seconds)"] > 0.001) & (df["duration (seconds)"] < 1.0)]["duration (seconds)"].apply(log))

# get the 5 first rows (head), 5 last rows (tail) and the summary statistics (describe) of the dataset
def describe_dataset():
  describe = df.describe()
  print("describe: \n", describe)

##########
# What are are the most common UFO shape sightings?
##########

# plot the visualization
def find_shape_in_plot():
   df["shape"].fillna("not sighted").astype("category").value_counts().plot.bar()
   plt.show()

##########
# where are all the ufo sightings on a world map
##########

# what are the columns actually called??
def print_columns():
    print(df.columns)

# Remove the spaces from column names
def remove_spaces_from_columns():
    df.columns = df.columns.str.strip()

# there are errors in the dataset where some latitude and longitude values are strings (and should be numeric)
# Convert latitude and longitude to numeric, forcing errors to NaN cause wtf else to do
def basemap():
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    #Drop rows with NaN values in latitude or longitude
    df = df.dropna(subset=['latitude', 'longitude'])

    # Basemap instance
    m = Basemap(projection="merc", llcrnrlat=-60, urcrnrlat=85,
           llcrnrlon=-180, urcrnrlon=180, resolution="c") # resolution of map can be set to c = crude, l = low, i = intermediate

    # Draw map features
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color="lightgray", lake_color="aqua")

    # Convert latitude and longitude to map coordinates
    x, y = m(df["longitude"].values, df["latitude"].values)

    # Plot the data points with a constant size
    m.scatter(x, y, s=2, c="red", alpha=0.5, zorder=5)

    # Add a title
    plt.title("UFO on World Map")
    # show the plot with matplotlib
    plt.show()

##########
# how long is the average ufo sighting // answer, around 2 and a half hours per sighting
##########
def avg_sighting_time():
    dfmask = df["duration (seconds)"].map(convert_float).fillna(0).mean()
    print(dfmask)


##########
# how long is the average ufo sighting per country
##########
def average_sighting_time_per_country():
    avgTimeByCountry = df.groupby("country")["duration (seconds)"].mean()
    print(avgTimeByCountry)


##########
# the number of ufo sightings per country
##########
def show_sighting_foreach_country():
  df["country"].value_counts().plot.box()
  plt.show()


# how many sightings per country
def print_sighting_foreach_country():
  print(df["country"].fillna("unknown country").value_counts()) 

##########
# how the hell is gb avg time 66061 sec? GB greatly outnumbers the other countries in length of UFO sightings are there any outliers?
##########
def show_box_for_gb_for_sightings_duration():
  dfextra["duration (seconds)"] = dfextra["duration (seconds)"].map(convert_float).fillna(1)
  dfextra["duration (seconds)"] = dfextra["duration (seconds)"].map(convert_less_then_zero_numbers_to_one)
  maskGB = dfextra["country"] == "gb"
  smalldf = dfextra[maskGB]
  print(smalldf)
  # print(smalldf["duration (seconds)"].max()) # 97836000.0 seconds (over 3 years) of 1 ufo sighting

  
# #########################
# question:
# when you have so many different outliers (change the number below to see the different outliers) then, what do you do to get a "good" data result?
# Answer, refer to common sense
# try logarithmic
# try procentage // why would we do procentage?
# #########################
  # normal
  # smalldf[smalldf["duration (seconds)"] < 90000.0]["duration (seconds)"].plot.box()
  # logarithmic
  # smalldf[smalldf["duration (seconds)"] < 90000.0]["duration (seconds)"].apply(log).plot.box()
  # plt.show()
  # procentage
  
  countryLog = smalldf[smalldf["duration (seconds)"] < 10000000.0]["duration (seconds)"].apply(log)
  print("log min: ", countryLog.min())
  print("log max: ", countryLog.max())
  print("log mean: ", countryLog.mean())
  print("log median: ", countryLog.median())

  median_duration_seconds = math.exp(countryLog.median())
  print("The sighting duration (seconds) for the country was:", median_duration_seconds, "median seconds")

# this is a helper function to convert less then zero numbers to one
def convert_less_then_zero_numbers_to_one(fl):
  if fl < 1.0:
    return 1.0
  else:
    return fl
  

##########
# what is the comment to the highest duration sighting
##########
def comments_of_the_highest_duration_sighting():
    print(df[df["duration (seconds)"] == 97836000.0]) # full columns with values of that specific outlier
    print(df[df["duration (seconds)"] == 97836000.0]["comments"]) # only the comments of that specific outlier
    df[df["duration (seconds)"] == 97836000.0]



##########
# trim the outliers from the data
##########
import scipy.stats
def trimmed_mean_for_gb():
    trim_mean = lambda x, *args: scipy.stats.trim_mean(x, 0.1)
    trim_mean.__name__ = "trim_mean"
    maskGB = df["country"] == "gb"
    smalldf = df[maskGB]
    return smalldf.groupby("country").agg({"duration (seconds)": [trim_mean]})


# print(df[(df["duration (seconds)"] > 0.001) & (df["duration (seconds)"] < 1.0)]["duration (seconds)"].apply(log))
# print(df[df["duration (seconds)"] < 9000.0]["duration (seconds)"].apply(log))


########## Can you do things like mean on a column like country? ##########

########## trend time per sighting ##########

########## trend ufo sightings over datetime and date posted ##########
# Index(['datetime', 'city', 'state', 'country', 'shape', 'duration (seconds)',
#         'duration (hours/min)', 'comments', 'date posted', 'latitude',
#         'longitude '],
def trend_ufo_sightings_over_datetime():
    df["datetime"]    = pd.to_datetime(df["datetime"], errors="coerce") #should probably figure out how to handle the errors rather than ignoring them
    df["date posted"] = pd.to_datetime(df["date posted"], errors="coerce")

    # datetime
    df["year"]   = df["datetime"].dt.year
    df["month"]  = df["datetime"].dt.month
    df["day"]    = df["datetime"].dt.day
    df["hour"]   = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["second"] = df["datetime"].dt.second

    # date posted 
    df["date posted year"]  = df["date posted"].dt.year
    df["date posted month"] = df["date posted"].dt.month
    df["date posted day"]   = df["date posted"].dt.day

    print(df["datetime"].dt.year.value_counts().sort_index()) # takes the datetime column and then counts the values by year and then sorts them by index. dt is a datetime accessor built into pandas
    print(df["date posted"].dt.year.value_counts().sort_index())

    #print the earliest and latest sightings
    print("Earliest sighting:", df["datetime"].min())
    print("Latest sighting:", df["datetime"].max())

    # df["datetime"].dt.year.value_counts().sort_index().plot()
    # df["date posted"].dt.year.value_counts().sort_index().plot()
    # plt.show()

########## when were the dates posted compared to the datetime of the sighting ##########

########## compare more than one plots to each other ##########

########## years vs seconds ##########

########## what are the most common UFO shapes per country? ##########
def country_trend_to_ufo_shape():
    df["shape"] = df["shape"].fillna("not sighted")

    most_seen   = df.groupby("country")["shape"].agg(lambda x: x.mode().iloc[0]) # mode() is built into pandas, it returns the most frequent values in a series apparently 
    second_seen = df.groupby("country")["shape"].agg(lambda x: x.value_counts().index[1])
    third_seen  = df.groupby("country")["shape"].agg(lambda x: x.value_counts().index[2])

    print("Most Seen UFO Shape Per Country:")
    print(most_seen, "\n")
    print("Second Seen UFO Shape Per Country:")
    print(second_seen, "\n")
    print("Second Seen UFO Shape Per Country:")
    print(third_seen)

########## did the comments become weirder over time ##########

########## are there trends in the comments which shows the same style / facets / words / descriptions of the ufos ##########


# describe_dataset()

# find_shape_in_plot()

# print_columns()

# remove_spaces_from_columns()

# basemap()

# # convert_float(string)

# avg_sighting_time()

# average_sighting_time_per_country()

# show_sighting_foreach_country()

# print_sighting_foreach_country()

# show_box_for_gb_for_sightings_duration()

# comments_of_the_highest_duration_sighting()

# trimmed_mean_for_gb()

# country_trend_to_ufo_shape()

trend_ufo_sightings_over_datetime()
