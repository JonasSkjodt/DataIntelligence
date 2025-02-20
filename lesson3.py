import pandas as pd
import numpy as np

df = pd.read_csv("starbucks.csv")

# get the 5 first rows (head), 5 last rows (tail) and the summary statistics (describe) of the dataset
head = df.head(5)
tail = df.tail(5)
describe = df.describe()

# print(head)
# print(tail)
# print(describe)

# Define a function which selects and returns the column containing saturated fat from the starbucks dataset
def get_saturated_fat():
    # fill in the function
     return df["Saturated Fat (g)"]

column_names = ["beverage-category", "beverage", "beverage-prep", "calories",
    "fat-total", "fat-trans", "fat-saturated", "sodium", "carbohydrates",
    "cholesterol", "fibre", "sugars", "protein", "vitamin-a", "vitamin-c",
    "calcium", "iron", "caffeine"]

def rename_dataframe_columns():
    # load the starbucks dataset once again, this time using `column_names` as the column names
    return pd.read_csv("starbucks.csv", names=column_names, skiprows=1)

def get_fatty_values():
    # Fill in the function to select all the rows where `fat-total` is larger than 3, check if all of the values in fat-total are floats
    def is_float(v):
      try:
          float(v)
          return True
      except ValueError:
          return False
    mask = df["fat-total"].map(is_float)
    df[~mask]