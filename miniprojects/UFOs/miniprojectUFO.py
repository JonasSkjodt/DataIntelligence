# questions
# when talking about modeling data via inferential statistics, does that mean stuff like regression and class...
# I dont understand the true positive and false negitive in class... in things like recall
# Whats the best way to do a random forest plot when doing a classification report? Can you show us an example?
# is the decision tree even needed to visualize, compared to the classification report?
# Can you do things like "mean" on a column with names and such? 


# TODOS
# make more classification
# make more regression plots 
# make more seaborn
# make more than one plot (draw the plot in another window with the same data visualized) for the same function



# ----------------------------------------
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


import pandas as pd
import numpy as np

# visualization with matplotlib
import matplotlib.pyplot as plt

# data manipulation with pandas
import pandas as pd

# visualization with seaborn
import seaborn as sns

# visualization with plotnine
import plotnine as p9
from plotnine import options

#testing matplotlib world map
from mpl_toolkits.basemap import Basemap

# for logaritmic
import math
from math import log

#preprocessing
import sklearn.preprocessing

# for regex
import re

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
  # https://www.geeksforgeeks.org/how-to-calculate-the-percentage-of-a-column-in-pandas/
  
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

    # print(df["datetime"].dt.year.value_counts().sort_index()) # takes the datetime column and then counts the values by year and then sorts them by index. dt is a datetime accessor built into pandas
    # print(df["date posted"].dt.year.value_counts().sort_index())

    # #print the earliest and latest sightings
    # print("Earliest sighting:", df["datetime"].min())
    # print("Latest sighting:", df["datetime"].max())

    # matplotlib
    # df["datetime"].dt.year.value_counts().sort_index().plot()
    # df["date posted"].dt.year.value_counts().sort_index().plot()
    # plt.show()

    # pandas

    # seaborn
    # sns.countplot(data=df, x="year")
    # plt.xticks(rotation=90)
    # plt.show()

    # plotnine
    # p9plot = p9.ggplot(df, p9.aes(x="year")) + p9.geom_bar(fill="lightblue")
    # # print(p9plot)

    # options.figure_size = (8, 6)
    # print("Drawing plot...")
    # p9plot.draw()
    # print("Plot drawn.")
    # p9plot.save("ufo_sightings_by_year.png")


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
def comments_over_time():
  df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
  df["date posted"] = pd.to_datetime(df["date posted"], errors="coerce")

  # Extract year from datetime
  df["year"] = df["datetime"].dt.year

  # Group by year and concatenate comments
  comments_by_year = df.groupby("year")["comments"].apply(lambda x: ' '.join(x.dropna()))

  # Print comments by year
  for year, comments in comments_by_year.items():
    print(f"Year: {year}")
    print(f"Comments: {comments[:100]}...")  # Print first 200 characters for brevity
    print("\n")





# YES Continue looking at pandas methods for descriptive statistics and data handling: aggregation using .groupby, handling missing values using .fillna/.isna/.dropna
# YES Find a couple of features in the dataset that you want to examine
# Identify whether a regression or classification best answers your question
# Try to model the features using inferential statistics
# YES Evaluate how well your model explains the features
########## Is there coming more sighting in the fellowing years in our dataset

# scikit-learn is imported by the name sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics

###### REGRESSION ######
## Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
# X:  80332
# y:  80332
# array.reshape(-1, 1): Reshapes a single feature into a column vector.
# array.reshape(1, -1): Reshapes a single sample into a row vector.
rng = np.random.RandomState(0)
def split_data(x, y):
   X = df[x].values.reshape(-1, 1)
   y = df[y].values
   X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=rng)
   return X_train, X_test, y_train, y_test
 
def split_data2(x, y):
   X = df[x].values
   y = df[y].values
   X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=rng)
   return X_train, X_test, y_train, y_test

def single_feature_linear_regression(x, y):
    X_train, X_test, y_train, y_test = split_data(x, y)
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return sklearn.metrics.r2_score(y_test, preds)


def more_sightings_in_the_following_years_with_regression():
  trend_ufo_sightings_over_datetime()
  single_feature_linear_regression("year", "date posted year")

def sighting_over_datetime_plot():
  trend_ufo_sightings_over_datetime()

  yearly_counts = df["year"].value_counts().sort_index()
  X = yearly_counts.index.values.reshape(-1, 1)
  y = yearly_counts.values
  
  # this is not need for seaborn to work
  model = sklearn.linear_model.LinearRegression()
  model.fit(X, y)
  preds = model.predict(X)
  r2_score = sklearn.metrics.r2_score(y, preds)

  # seaborn
  sns.regplot(x=X, y=y, data=df, ci=None, line_kws={"color": "red"}) # scatter_kws={"color": "black"}
  plt.text(x=1950, y=6000, s=f"R^2 Score: {r2_score}", fontsize=10, color='red')
  plt.xlim(1900, 2020)
  plt.ylim(0, 8000)
  # plt.plot([0, 4], [1.5, 0], linewidth=2)

  # sklearn and matplotlib
  # plt.scatter(X, y, color='blue', label='Actual')
  # plt.plot(X, preds, color='red', linewidth=2, label='Predicted')
  # plt.xlabel("year")
  # plt.ylabel("number og sightings")
  # plt.title(f'Linear Regression: oirjgre')
  # plt.legend()
  # plt.ylim(0, 8000) # set the y-axis limits
  # plt.xlim(1900, 2020)
  # plt.text(1940, 7000, f"R^2 Score: {r2_score}")
  # plt.text(1940, 6000, f"coefficient: {model.coef_}")
  
  plt.show()

###### CLASSIFICATION ######

from sklearn.tree import plot_tree

def sightings_over_time_with_classification():
  trend_ufo_sightings_over_datetime()

  input_features = ["year"]
  X_train, X_test, y_train, y_test = split_data2(input_features, "date posted year")
  model = sklearn.ensemble.RandomForestClassifier(random_state=rng)
  model.fit(X_train, y_train)
  preds = model.predict(X_test)
  test_report = sklearn.metrics.classification_report(y_test, preds)
  print(test_report) # report says f1 has a weighted avg og 74%
  feature_importances = "\n\t".join(f"{feature}: {importance:.2f}" for feature, importance in zip(input_features, model.feature_importances_))
  print(f"Feature importances:\n\t{feature_importances}")

  # trying to do a decision tree plot
  # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
  h = sklearn.tree.plot_tree(model.estimators_[0],
      feature_names=["year"],
      class_names=df["date posted year"], filled=True)
  
  plt.figure(figsize=(20, 10))
  plot_tree(h, feature_names=["year"], filled=True, rounded=True, fontsize=10)
  plt.title("Decision Tree from Random Forest")
  plt.show()

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt

# # Assuming regressor is your trained Random Forest model
# # Pick one tree from the forest, e.g., the first tree (index 0)
# tree_to_plot = regressor.estimators_[0]

# # Plot the decision tree
# plt.figure(figsize=(20, 10))
# plot_tree(tree_to_plot, feature_names=df.columns.tolist(), filled=True, rounded=True, fontsize=10)
# plt.title("Decision Tree from Random Forest")
# plt.show()
  
########## are there trends in the comments which shows the same style / facets / words / descriptions of the ufos ##########

########## ##########

########## trend time per sighting ##########

########## when were the dates posted compared to the datetime of the sighting ##########

########## compare more than one plots to each other ##########

########## years vs seconds ##########

#duration of seconds vs hours with scaling
def scale(df):
  scaler = sklearn.preprocessing.StandardScaler()
  return scaler.fit_transform(df)

# Function to convert time strings to seconds
# def convert_to_seconds(time_str):
#     if pd.isna(time_str):  # Handle NaN or None values
#         return 0
    
#     # Regex to extract numeric value and unit 1/2
#     #match = re.search(r"(\d'+|\d+\.?\d*)\s*\-?\s*(\d+\.?\d*)?\s*(hrs?|hour?s?|min(?:ute)?s?|sec(?:ond)?s?)", time_str, re.IGNORECASE)
#     match = re.search(r"((?:\d+\/\d+|\d+(?:\.\d+)?|\d+')?)\s*\-?\s*((?:\d+\/\d+|\d+(?:\.\d+)?)?)\s*(hrs?|hour?s?|min(?:ute)?s?|sec(?:ond)?s?|hour)", time_str, re.IGNORECASE)
#     # (|\d'+|\d+\.?\d*)\s*\-?\s*(\/?\d*\d+\.?\d*)?\s*(hrs?|hour?s?|min(?:ute)?s?|sec(?:ond)?s?) gives (1) (/2) (hours)
#     if not match:
#         return 0  # If no match, return 0 or handle as needed
    
#     # Extract numeric value and unit
#     value1 = float(match.group(1))
#     value2 = float(match.group(2)) if match.group(2) else value1  # Handle ranges like "1-2 hrs"
#     unit = match.group(3).lower()  # Normalize unit to lowercase
    
#     # Average the range (e.g., "1-2 hrs" becomes 1.5 hrs)
#     avg_value = (value1 + value2) / 2
    
#     # Convert to seconds based on unit
#     if unit.startswith("hr"):
#         return avg_value * 3600
#     elif unit.startswith("min"):
#         return avg_value * 60
#     elif unit.startswith("sec"):
#         return avg_value
#     else:
#         return 0  # Handle unknown units

def convert_to_seconds2(time_str):
    if pd.isna(time_str):  # Handle NaN or None values
        return 0

    # Regex to extract numeric value and unit, including 1/2 and "hour"
    match = re.search(r"((?:\d+\/\d+|\d+(?:\.\d+)?|\d+')?)\s*\-?\s*((?:\d+\/\d+|\d+(?:\.\d+)?)?)\s*(hrs?|hour?s?|min(?:ute)?s?|sec(?:ond)?s?|hour)", time_str, re.IGNORECASE)

    if not match:
        return 0  # If no match, return timestamp or handle as needed

    # Extract numeric value and unit
    value1_str = match.group(1)
    value2_str = match.group(2)
    unit = match.group(3).lower()  # Normalize unit to lowercase

    def parse_value(value_str):
        if not value_str:
            return float(1)
        if '/' in value_str:
            num, den = map(float, value_str.split('/'))
            return num / den
        else:
            try:
                return float(value_str)
            except ValueError:
                return None

    value1 = parse_value(value1_str)
    value2 = parse_value(value2_str) if value2_str else value1

    if value1 is None:
      return 0

    # Handle if value 2 is None, if value 2 is none, then value 1 is used.
    if value2 is None:
        avg_value = value1
    else:
        avg_value = (value1 + value2) / 2


    # hour.min sec
    # 00.00 00
    # 1 hour / 25 min / 02 sec
    # "1" + "." + "25" + "02"
    # 1.2502
    
    # Convert to seconds based on unit
    if unit.startswith("hr") or unit == "hour":
        return avg_value * 3600
    elif unit.startswith("min"):
        return avg_value * 60
    elif unit.startswith("sec"):
        return avg_value
    else:
        return 0  # Handle unknown units
    
    
from sklearn.preprocessing import MinMaxScaler  
def seconds_and_hours_with_scaling():

  df["duration (hours/min)"] = df["duration (hours/min)"]# .apply(convert_to_seconds2)

  #print(df["duration (seconds)"])
  print(df["duration (hours/min)"])
  


def jitter_plot():
  # Create example datasets
  data1 = {'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50]}  # High numbers
  data2 = {'x': [1, 2, 3, 4, 5], 'y': [10000, 20000, 30000, 40000, 50000]}       # Low numbers

  # Convert to DataFrame
  df1 = pd.DataFrame(data1)
  df2 = pd.DataFrame(data2)

  # Normalize the y-values using MinMaxScaler
  scaler = MinMaxScaler()
  
  df1['y_scaled'] = scaler.fit_transform(df1[['y']])
  df2['y_scaled'] = scaler.fit_transform(df2[['y']])

  # Add jitter to the scaled values
  jitter_range = 0.05  # Adjust this value for more or less jitter
  df1['y_jitter'] = df1['y_scaled'] + np.random.uniform(-jitter_range, jitter_range, size=len(df1))
  df2['y_jitter'] = df2['y_scaled'] + np.random.uniform(-jitter_range, jitter_range, size=len(df2))


  # Combine the data for plotting
  df1['label'] = 'High Numbers'
  df2['label'] = 'Low Numbers'

  combined_df = pd.concat([df1, df2])

  # Plot using seaborn
  plt.figure(figsize=(8, 6))
  sns.scatterplot(data=combined_df, x='x', y='y_jitter', hue='label', style='label', s=100)
  
  # Add labels and title
  plt.title('Scatter Plots with Scaled Y-Axis', fontsize=14)
  plt.xlabel('X-axis', fontsize=12)
  plt.ylabel('Scaled Y-axis', fontsize=12)
  plt.legend(title='Legend')
  plt.show()
 
print(df["duration (hours/min)"].head(60))

# seconds_and_hours_with_scaling()
# trend_ufo_sightings_over_datetime()

# sighting_over_datetime_plot()

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

#comments_over_time()

# more_sightings_in_the_following_years_with_regression()

# sightings_over_time_with_classification()
