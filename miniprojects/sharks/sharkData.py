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

df = pd.read_csv("dataExam/shark-incidents.csv")

print(df)