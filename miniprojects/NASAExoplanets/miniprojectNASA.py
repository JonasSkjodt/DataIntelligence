# NASA Exoplanets Data Analysis
# The NASA Exoplanet Archive is a database that contains information on all known exoplanets
#  (planets outside our solar system) discovered by NASA's various space missions,
# ground-based observatories, and other sources. The dataset includes information
# such as the planet's name, mass, radius, distance from its host star, orbital period,
# and other physical characteristics. The dataset also includes information on the host star,
# such as its name, mass, and radius. The archive is updated regularly as new exoplanets are
# discovered, and it is a valuable resource for astronomers studying the properties and
# distribution of exoplanets in our galaxy.

# mass of the planet = mass_multiplier * mass_wrt(planet)
# radius of the planet = radius_multiplier * radius_wrt(planet)

# name
# Name of the planet as per given by NASA

# distance
# distance of the planet from earth in light years

# stellar_magnitude
# Brightness of the planet, the brighter the planet the lower number is assigned to the planet

# planet_type
# Type of the planet, these types are derived from our solar system planets

# discovery_year
# Year in which planet got discovered

# mass_multiplier
# mass multiplier of the planet with mass_wrt planet

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
df = pd.read_csv("cleaned_5250.csv", low_memory=False)

def describe_dataset():
  describe = df.describe()
  print("describe: \n", describe)

# name of columns
# Index(['name', 'distance', 'stellar_magnitude', 'planet_type',
#        'discovery_year', 'mass_multiplier', 'mass_wrt', 'radius_multiplier',
#        'radius_wrt', 'orbital_radius', 'orbital_period', 'eccentricity',
#        'detection_method'],
#       dtype='object')
def print_columns():
    print(df.columns)

# what is the most common planet type? (neptune-like)
def most_common_planet_type():
    planet_types = df['planet_type'].value_counts()
    print("Most common planet type: \n", planet_types)

# plot of the most common planet type with seaborn
def plot_most_common_planet_type():
    planet_types = df['planet_type'].value_counts()
    sns.barplot(x=planet_types.index, y=planet_types.values)
    plt.title('Most Common Planet Type')
    plt.xlabel('Planet Type')
    plt.ylabel('Count')
    plt.xticks(rotation=90) # becase readability is shit otherwise
    plt.show()

# How has the number of exoplanet discoveries evolved over time (yea and decade)?
# its high in 2014 and 2016 because of the kepler Space Telescope Data validations + better machine learning
# which made it easier to shift through all the data
def discoveries_over_time():
    # Group by year and count discoveries
    yearly_discoveries = df['discovery_year'].value_counts().sort_index()
    print(yearly_discoveries)
    # Plot
    plt.figure(figsize=(12, 6))
    yearly_discoveries.plot(kind='bar', color='skyblue')
    plt.title('Exoplanet Discoveries Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Discoveries')
    plt.grid(axis='y', linestyle='--')
    plt.show()
    
    # Decade
    df['decade'] = (df['discovery_year'] // 10) * 10
    decade_counts = df['decade'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 5))
    decade_counts.plot(kind='bar', color='teal')
    plt.title('Exoplanet Discoveries by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Number of Discoveries')
    plt.xticks(rotation=0)
    plt.show()

# 2014 and 2016 has a lot of discoveries, what were the detections methods used here?
# the print shows how the detection method "Transit" was used a lot in 2014 and 2016.
# Transit is a method from the Kepler Space Telescope.
def check_2014_2016_discoveries():
    # Filter for 2014 and 2016
    df_2014 = df[df['discovery_year'] == 2014]
    df_2016 = df[df['discovery_year'] == 2016]
    
    print("2014 Detection Methods:")
    print(df_2014['detection_method'].value_counts())
    
    print("\n2016 Detection Methods:")
    print(df_2016['detection_method'].value_counts())
    
    # Check if Kepler was involved
    kepler_2014 = df_2014['name'].str.contains('Kepler').sum()
    kepler_2016 = df_2016['name'].str.contains('Kepler').sum()
    print(f"\nKepler-named planets in 2014: {kepler_2014}")
    print(f"Kepler-named planets in 2016: {kepler_2016}")

# Did technology enhance the ability to see more exoplanets over the years or was it just kepler?
# The plot shows the trend towards outphasing older methods and using transit more and more over the years for the discovery of exoplanets
def did_technology_enhance_ability_to_see_exoplanets():
    # Calculate proportion of methods per year
    method_proportions = (
        df.groupby(['discovery_year', 'detection_method'])
        .size()
        .unstack()
        .apply(lambda x: x / x.sum(), axis=1)
    ) # normalize the data to get rid of the outliers
    
    # Plot line chart (shows absolute method introductions)
    plt.figure(figsize=(14, 7))
    for method in method_proportions.columns:
        plt.plot(method_proportions.index, method_proportions[method], 
                label=method, marker='o', linestyle='-')
    plt.title('Detection Method Adoption Over Time')
    plt.xlabel('Year')
    plt.ylabel('Proportion of Discoveries')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()
    
    # Print first use of each method
    method_first_use = df.groupby('detection_method')['discovery_year'].min()
    print("First recorded use of each detection method:")
    print(method_first_use.sort_values())


# What detection type is most common?
# just trying out different plots to again show its transit thats the most common detection method
def common_detection_methods():
    # Count detection methods
    method_counts = df['detection_method'].value_counts()
    
    # Plot
    plt.figure(figsize=(10, 6))
    method_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Detection Methods')
    plt.ylabel('')
    plt.show()
    
    # Alternative bar plot
    plt.figure(figsize=(10, 5))
    sns.barplot(x=method_counts.index, y=method_counts.values, palette='rocket')
    plt.title('Most Common Detection Methods')
    plt.xlabel('Method')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Does the detection method influence the type of planets found?
# trying out heatmap to see the planet types discovered by which detection method
# the heatmap shows how bigger gas giants were discovered by older exoplanet detection methods and newer smaller planets were discovered via transit
def method_vs_planet_type():
    # Cross-tabulation
    cross_tab = pd.crosstab(df['detection_method'], df['planet_type']) # crosstab is a pandas function. It takes two columns and creates a table with the counts of each combination of values in those columns.
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrBr')
    plt.title('Detection Method vs Planet Type')
    plt.xlabel('Planet Type')
    plt.ylabel('Detection Method')
    plt.show()
    
    # Normalized view with procentages
    cross_tab_norm = cross_tab.div(cross_tab.sum(1), axis=0) # normalize the data to get rid of the outliers
    plt.figure(figsize=(12, 8))
    sns.heatmap(cross_tab_norm, annot=True, cmap='YlOrBr') # YlOrBr is a color palette
    plt.title('Detection Method vs Planet Type (Normalized)')
    plt.title('Normalized: Detection Method vs Planet Type')
    plt.xlabel('Planet Type')
    plt.ylabel('Detection Method')
    plt.show()

# Is there a relationship between planet_type and orbital_period?
# the boxplot showed how the super earths are in a specific range of orbits around the sun, which is interesting because it shows how they are in a specific range of orbits around the sun
def planet_type_vs_orbital_period():
    # Filter out extreme outliers for better visualization
    filtered = df[df['orbital_period'] < df['orbital_period'].quantile(0.99)]
    
    # Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='planet_type', y='orbital_period', data=filtered, showfliers=False)
    plt.yscale('log')  # Log scale due to wide range
    plt.title('Orbital Period by Planet Type (log scale)')
    plt.xlabel('Planet Type')
    plt.ylabel('Orbital Period (days)')
    plt.xticks(rotation=45)
    plt.show()
    
    # Statistical test
    # Kruskal-Wallis is a non-parametric method for testing whether samples originate from the same distribution (from chatGPT)
    from scipy.stats import kruskal
    groups = [group['orbital_period'].values for name, group in filtered.groupby('planet_type')]
    h_stat, p_val = kruskal(*groups)
    print(f"\nKruskal-Wallis Test Results:")
    print(f"H-statistic: {h_stat:.2f}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Significant differences exist between planet types.")
    else:
        print("No significant differences detected.")

# describe_dataset()
# print_columns()
# most_common_planet_type()
# plot_most_common_planet_type()
# discoveries_over_time()
# check_2014_2016_discoveries()
# did_technology_enhance_ability_to_see_exoplanets()
# common_detection_methods()
# method_vs_planet_type()
planet_type_vs_orbital_period()