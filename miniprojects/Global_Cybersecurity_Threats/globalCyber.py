import shap
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv("Global_Cybersecurity_Threats_2015-2024.csv")

def is_column_float(column):
    istrue = True
    for value in df[column]:
        if type(value) is not float:
            istrue = False
    return istrue

# print(df.columns)
# Country - ints
# Year - ints
# Attack Type - Strings
# Target Industry - Strings
# Financial Loss (in Million $) - floats
# Number of Affected Users - ints
# Attack Source - Strings
# Security Vulnerability Type - Strings
# Defense Mechanism Used - Strings
# Incident Resolution Time (in Hours) - ints

# print(df["Attack Type"].value_counts())
# DDoS                 531
# Phishing             529
# SQL Injection        503
# Ransomware           493
# Malware              485
# Man-in-the-Middle    459

# print(df["Country"].value_counts())
# UK           321
# Brazil       310
# India        308
# Japan        305
# France       305
# Australia    297
# Russia       295
# Germany      291
# USA          287
# China        281

def class_test():
    # Dont know why, the site said so
    X = df.drop("Financial Loss (in Million $)", axis=1)
    y = df["Financial Loss (in Million $)"]

    # this is all categorical columns there is
    categorical_cols = [
        "Country",
        "Attack Type",
        "Target Industry",
        "Attack Source",
        "Security Vulnerability Type",
        "Defense Mechanism Used"
    ]

    # all numerical columns
    numerical_cols = [
        "Year",
        "Number of Affected Users",
        "Incident Resolution Time (in Hours)"
    ]

    # Seems to work, but not sure how yet
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    # same
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    # show tree
    rf_model = model.named_steps['regressor']
    tree = rf_model.estimators_[0]

    plt.figure(figsize=(20, 10))
    plot_tree(tree,
            filled=True,
            feature_names=model.named_steps['preprocessor'].get_feature_names_out(),
            max_depth=3,
            fontsize=10)
    plt.title("Visualization of a Random Forest Tree")
    plt.show()
    # --------
    
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    print(f"Mean Squared Error: {mse:.2f}")
class_test()

def df_des():
    print(df.describe())
# df_des()

#print(df.isna().value_counts())
#3000 false, so should be good

#UK has the highest number of attacks, what are they? - phishing
def uk_attacks():
    uknum = df[df["Country"] == "UK"]
    # uknum = uknum.groupby("Attack Type")["Defense Mechanism Used"].agg(lambda x: x.mode().iloc[0])
    print("Most used attack: ")
    attacks = uknum["Attack Type"].value_counts()
    print(attacks)
    print("\n")

    print("Most used Defense: ")
    defense = uknum["Defense Mechanism Used"].value_counts()
    print(defense)
    print("\n")
    
    # df["Attack Type"].corr(df["Defense Mechanism Used"])
    print("Most used Defense per Attack: ")
    uk = uknum.groupby("Attack Type")["Defense Mechanism Used"].value_counts()
    print(uk)
    print("\n")

    print("Most used Attack per Defense: ")
    uk = uknum.groupby("Defense Mechanism Used")["Attack Type"].value_counts()
    print(uk)
    print("\n")
# uk_attacks()

# which country has the biggest loss? - UK
def fin_loss():
    fin = df.groupby("Country").agg({"Financial Loss (in Million $)": "sum"})
    print(fin.sort_values(by="Financial Loss (in Million $)", ascending=False))
# fin_loss()

# What makes the most amount of damage?
def Atk_damage():
    fin = df.groupby("Attack Type").agg({"Financial Loss (in Million $)": "sum"})
    print(fin.sort_values(by="Financial Loss (in Million $)", ascending=False))
# Atk_damage()

# what is the avg time to fix attack?
def Atk_time():
    fin = df.groupby("Attack Type").agg({"Incident Resolution Time (in Hours)": "mean"})
    print(fin.sort_values(by="Incident Resolution Time (in Hours)", ascending=False))
# Atk_time()


# is there an attack that some more the most?
def atk_overall():
    over = df.groupby("Attack Type")["Country"].value_counts()
    print(over)
# atk_overall()

# has there been 



# SHAP Values Training
# https://www.kaggle.com/code/vikumsw/explaining-random-forest-model-with-shapely-values
# yeah fuck that for now
def shap_tree():
    rng = np.random.RandomState(0)

    # get_dummies makes 
    # print(pd.get_dummies(df["Attack Type"]))
    
    
    
    X = df[["Incident Resolution Time (in Hours)", "Financial Loss (in Million $)"]]
    Y = df["Year"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    
    model = RandomForestClassifier(random_state=rng)
    model.fit(X_train, y_train)
    # preds = model.predict(X_test)
    # test_report = classification_report(y_test, preds)
    # print(test_report)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # print(shap_values)
    shap.initjs()
    # shap.summary_plot(shap_values, X_test)
    shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)
# shap_tree()

# print(is_column_float("Incident Resolution Time (in Hours)"))