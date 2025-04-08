import pandas as pd
import numpy as np


df = pd.read_csv('shark-incidents.csv')

#print(df.head())

#
# [5 rows x 24 columns]
# Index(['Date', 'Year', 'Type', 'Country', 'State', 'Location', 'Activity',
#        'Name', 'Sex', 'Age', 'Injury', 'Unnamed: 11', 'Time', 'Species ',
#        'Source', 'pdf', 'href formula', 'href', 'Case Number', 'Case Number.1',
#        'original order', 'Unnamed: 21', 'Unnamed: 22', 'fatal'],
#       dtype='object')
#print(df.columns)

print(df.describe)

# whats the underlying tendencies of the people who are injured by sharks?

# skeleton of exam report:


# introduction
# loading and overview
### explain the steps i do and why ive used the blocks of code.
# exploration of the data
### motivate, then show plot (data), then interpret data
