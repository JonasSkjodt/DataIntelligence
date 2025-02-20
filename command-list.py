# install pandas
!pip install pandas

# Fetch the dataset
!curl -L "https://docs.google.com/spreadsheets/d/e/2PACX-1vSN0mU23u8L-al7IBmp7zVplxPe-mPw5cMdpDDpHddgewWWgYwfonSrVmoWHOlImE8A8zL1o2jT7rg1/pub?gid=1037190964&single=true&output=csv" -o starbucks.csv

# import pandas
import pandas as pd
import numpy as np

# now define the dataset
df = pd.read_csv("starbucks.csv")

#this will remove the some of the stupid fails in a dataset, like whitespaces
df = pd.read_csv("path to dataset", skiprows=1, names=column_names)

df.describe()  # this will give you a good metric overview of the dataset

# there is a problem with the dataset where a data is 3 2 instead of 3.2
# so we could write a function that checks if some data can be converted to a float
def is_float(v):
    try:
        float(v)
        return True
    except ValueError:
        return False

mask = df["fat-total"].map(is_float) #check the whole row

df[mask] #prints out all the row that could be converted to float

df[~mask] #prints out all the row that couldnt be converted to float

df.loc[~mask, "fat-total"] = 3.2 
# this will then find the value(s) that couldnt be converted in a row
# the = will the replace the value there was before

df["fat-total"] = df["fat-total"].astype(float) 
#converts the newly change row to a float now

# if i wanted to find the corrolation between two columns
df["Calories"].corr(df["Protein (g)"])

# DataFrames and Series have a collection of methods to calculate summary statistics from the data. We'll start by looking at .mean(), .median(), .std(), and .corr().

# .mean() gives you then mean
# .median() gives you the median
# .std() gives you the standard deviation. This is explained in detail below.
# .corr() gives you a correlation matrix. This is also explained further below.