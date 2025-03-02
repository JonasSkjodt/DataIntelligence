import pandas as pd
import numpy as np

def convert_percentage(string):
    number_as_string = string[:-1]
    number = float(number_as_string)
    return number / 100

column_names = ["beverage-category", "beverage", "beverage-prep", "calories",
    "fat-total", "fat-trans", "fat-saturated", "sodium", "carbohydrates",
    "cholesterol", "fibre", "sugars", "protein", "vitamin-a", "vitamin-c",
    "calcium", "iron", "caffeine"]

df = pd.read_csv("starbucks.csv", skiprows=1, names=column_names)

for col in ["vitamin-a", "vitamin-c", "calcium", "iron"]:
    df[col] = df[col].map(convert_percentage)

print(df['fat-total'])

# make a function which converts the input to a float and returns None if the input cannot be converted.
def convert_float(string):
    # fill in the function here
    if string is None:
      return None
    try:
      return float(string)
    except ValueError:
      return None

# Make a function which converts values to floats and handles missing values
def convert_floats_and_fill_missing_values(dataframe, column_name):
    # fill in function to convert the column specified by `column_name` in the given dataframe to float and replace missing values by 0
    # return the converted dataframe from your function
    dataframe[column_name] = dataframe[column_name].map(convert_float).fillna(0)
    return dataframe

# make a function which returns unique values from the given column
def get_unique_values(dataframe, column_name):
    # fill in the function to return unique values from the given column of the dataframe
    return dataframe[column_name].unique()

# make a function which groups the given dataframe first by the "beverage-prep"
# and and secondly by the "beverage-category" columns and returns the means based on this grouping.
def group_and_get_mean(dataframe):
    # fill in the function here
    # df.groupby(["mammal", "species"]).sum(numeric_only=True)
    return dataframe.groupby(["beverage-prep", "beverage-category"]).mean(numeric_only=True)

# Write a function which groups by beverage-prep column and gets each row which has the
# largest calories value of each group. You should return a dataframe with all the
# relevant rows.
def get_max_calories(dataframe):
    # Fill in the function here
    # df.loc[tmp_df.groupby("species")["height"].idxmax()]
    return dataframe.loc[dataframe.groupby("beverage-prep")["calories"].idxmax()]

# Write a function which groups the data based on the beverage-category column and then
# aggregates the mean, median and trimmed mean for the calories column. To do this you
# can import the scipy.stats module and use the trim_mean method. Trim away 10% of the data.
def mean_median_and_trimmed_mean(dataframe):
    # Fill in the function here
    import scipy.stats
    # trim_mean = lambda x, *args: x.max() - x.min()
    trim_mean = lambda x, *args: scipy.stats.trim_mean(x, 0.1)
    trim_mean.__name__ = "trim_mean"
    return dataframe.groupby("beverage-category").agg({"calories": ["mean", "median", trim_mean]})

## visualize the data

# Fill in this block to plot a histogram of the calories feature
df["calories"].plot.hist()

# Fill in this block to plot the `beverage-category` feature as a bar plot.
df["beverage-category"].value_counts().plot.bar()

# Fill in this block to make a scatter plot of the `sugars` and `calories` features.
df.plot.scatter(x="sugars", y="calories")

# Fill in this block to make a line plot of the values of the `Cielab b*` feature against the `year`. Remember to use the new `df_paper` variable.
# df_paper.plot(x="year", y="Cielab b*")