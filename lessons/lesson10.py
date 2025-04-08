# example of how to deal with messy data and then tidy it  (in the example, the column header titles are hard to understand)
import pandas as pd

import dask.dataframe as dd
df = pd.read_csv("data.csv")

df.melt() #get all of the data into a single column, this is easier to work with when you have messy tidy.

# Only certain columns should be melted.

df_melted = df.melt(id_vars=['iso2', 'years'], value_name='cases')

df_melted["variable"]

# extract column headers from the melted data frame and put them into a new column called 'gender'.

df_melted["gender"] = df_melted["variable"].map(lambda v: v[7])

df_melted["gender"]

# now for the other column header, we make a function to convert the column header into a more readable format.

def convert_age(age_string):
    age_string = age_string[8:]
    l = len(age_string)
    if l == 2:
        return age_string
    elif l == 3:
        return "0-14"
    elif l == 4:
        return age_string[:2] + "-" + age_string[2:]
    
convert_age("column_name") # testing the function

df_melted["age"] = df_melted["variable"].map(convert_age)

# clean up
del df_melted["variable"] # drop the variable column, we don't need it anymore




