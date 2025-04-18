# import shap
import pandas as pd
import numpy as np
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error
# from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# - Date: Date of the incident
# - Year: Year of the incident
# - Type: Whether the incident was provoked (i.e. the human involved speared, hooked or captured the shark first), unprovoked (e.g. when the shark bit out of curiosity), involved a watercraft (when the shark bit or rammed a boat), involved a disaster (i.e. a capsizing boat or a crashing plane) or is questionable (i.e. there is not enough data to determine whether a shark was involved in the actual incident)
# - Country: The country the incident happened in
# - State: The state in the country in which the incident happened
# - Location: The placename of the area in which the incident happened
# - Activity: The activity of the human immediately before the incident
# - Name: The name of the human involved in the incident
# - Sex: The gender of the human involved in the incident
# - Age: The age of the human involved in the incident
# - Injury: A textual description of the resulting injury to the human
# - Unnamed: 11: Unexplained data                                           - Removed 
# - Time: Time of day of the incident
# - Species: Species of the shark involved in the incident
# - Source: Source of the reporting of the incident
# - pdf: Filename of the report of the incident                             - Removed
# - href formula: Links to reports, no longer functional                    - Removed
# - href: Same as above                                                     - Removed
# - Case Number: Internal reference
# - Case Number.1: Same as above
# - original order: Ordering in the original system
# - Unnamed: 21: Unknown data                                               - Removed
# - Unnamed: 22: Unknown data                                               - Removed
# - fatal: Whether the incident resulted in the human dying

df = pd.read_csv("dataExam/shark-incidents.csv")

# print(df.describe)

def check_case_nan():
    filt = df[["Case Number", "Case Number.1"]]
    mask = filt.isna().any(axis=1)
    # print(mask.value_counts())
    # print(df[mask])
    
    # 
    nan_count_per_row = df[mask].isna().sum(axis=1)
    # print(nan_count_per_row)
    return df[df.isna().sum(axis=1) < 20]
# removes rows which has almost nothing in them
df = check_case_nan()

def check_Unamed_nan():
    mask = df[["Unnamed: 11", "Unnamed: 21", "Unnamed: 22"]].isna().any(axis=1)
    # print(mask.value_counts())
    return df.drop(columns=["Unnamed: 21", "Unnamed: 22"])
# removes column which are almost not used
df = check_Unamed_nan()


def check_href():
    # print(df["href"].value_counts())
    # print(df["href formula"].value_counts())
    return df.drop(columns=["href", "href formula"])
df = check_href()

def check_pdf():
    print(df["pdf"].isna().value_counts())
    return df.drop(columns=["pdf"])
    
df = check_pdf()

print(df)