import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.preprocessing


# First read the file "eggs.csv" into a pandas dataframe. Use the specified `column_names` as the column names.
# Hint: the value for skiprows should be larger than 1
column_names = ["cage-eggs", "free-range-eggs", "cage-free-eggs", "voliere-eggs", "organic-eggs", "sale-price-cage-free-eggs", "sale-price-organic-eggs"]
def read_csv():
    # Fill in the function to read in the csv file
    return pd.read_csv("eggs.csv", skiprows=3, names=column_names)

def convert_date(date):
    # Fill in the function to convert the date string into a datetime object
    try:
      m = {"1": 1, "2":4, "3": 7, "4":10}
      year, quarter = date.split("K")
      if quarter not in m:
        return None
      d = datetime.datetime(int(year), m[quarter], 1)
      return d
    # from teacher
    except (AttributeError, ValueError) as e:
        print(f"Warning: couldn't convert {date}: {e}", file=sys.stderr)


def convert_index(dataframe):
    # Fill in the function to convert the index of the given dataframe into a datetime index. Return the complete dataframe.
    dataframe.index = dataframe.index.map(convert_date)
    return dataframe

def drop_rows_with_missing_indices(dataframe):
    # Fill in the function to return a dataframe with the rows with missing indices dropped
    return dataframe.dropna(how="all")

# Make a function to replace all the ".." values with np.nan in the given dataframe
def convert_missing_values(dataframe):
    # Fill in the function to convert all the cells in the given dataframe which have ".." to nan. Return the complete dataframe.
    return dataframe.replace("..", np.nan)

def convert_to_float(f):
    if f is None:
        return None
    try:
        return float(f)
    except ValueError:
        return None
    
def convert_dataframe_to_floats(dataframe):
    # Fill in the function to convert all the values in the dataframe to floats. Return the complete dataframe.
    return dataframe.applymap(convert_to_float)

# missing values

# Make a function which fills the missing values of the specified column of the given dataframe by the mean of the same column.
# Return the complete dataframe from the function.
def fill_mean(dataframe, column):
    # Fill in the function to replace missing values with the mean value of the column. Return the complete dataframe.
    dataframe[column] = dataframe[column].fillna(dataframe[column].mean())
    return dataframe

def plot_difference(old_data, new_data, column):
    plot_df = pd.DataFrame(index=old_data.index)
    # Add some noise to the data to make the data points of
    # the two different series not have the exact same coordinates
    # while still having the same distributions.
    plot_df[f"{column}-pre"] = old_data[column] + (np.random.RandomState(0).rand(len(old_data[column])) * (old_data[column].std() / 3))
    plot_df[f"{column}-post"] = new_data[column]
    sns.scatterplot(x="index", y=f"{column}-pre", data=plot_df.reset_index(), color="red", label="pre")
    sns.scatterplot(x="index", y=f"{column}-post", data=plot_df.reset_index(), color="blue", label="post")
    plt.ylabel(column)

def fill_missing_values_bfill(dataframe, column):
    # Fill in the function to fill missing values using the bfill method. Return the complete dataframe.
    dataframe[column] = dataframe[column].bfill()
    return dataframe

def interpolate_values(dataframe, column):
    # Fill in the method to replace missing values in the given column of the dataframe using the `interpolate` method. Return the complete dataframe.
    dataframe[column] = dataframe[column].interpolate(method="linear", limit_direction="both")
    return dataframe

# outliers and scaling
# The data for sale-price-cage-free-eggs has a couple of outliers
sns.scatterplot(x="index", y="sale-price-cage-free-eggs", data=df.reset_index())

def lowest_five_percent(dataframe, column):
    # Fill out the function to return the fifth percentile
    return dataframe[column].quantile(0.05)

def highest_five_percent(dataframe, column):
    # Fill out the function to return the 95th percentile
    return dataframe[column].quantile(0.95)

ninety_fifth_percentile = highest_five_percent(df, "sale-price-cage-free-eggs")
assert ninety_fifth_percentile is not None, "Fifth percentile was None. Did you forget to return a value from your function?"
assert np.isclose(ninety_fifth_percentile, 1090.19), "Ninety-fifth percentile was not the expected 1090.19. Did you return the correct percentile?"
print("Ninety-fifth percentile tests succeeded!")