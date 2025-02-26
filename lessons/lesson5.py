import pandas as pd

def convert_percentage(string):
    number_as_string = string[:-1]
    number = float(number_as_string)
    return number / 100

def convert_float(string):
    if string is None:
        return None
    try:
        return float(string)
    except ValueError:
      return None

column_names = ["beverage-category", "beverage", "beverage-prep", "calories",
    "fat-total", "fat-trans", "fat-saturated", "sodium", "carbohydrates",
    "cholesterol", "fibre", "sugars", "protein", "vitamin-a", "vitamin-c",
    "calcium", "iron", "caffeine"]
df = pd.read_csv("starbucks.csv", skiprows=1, names=column_names)
df["fat-total"] = df["fat-total"].map(convert_float)
df["fat-total"] = df["fat-total"].fillna(0)
for col in ["vitamin-a", "vitamin-c", "calcium", "iron"]:
    df[col] = df[col].map(convert_percentage)

# scikit-learn is imported by the name sklearn
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

# First let's make some random data
# X are the features we'll use for inferencing
# y are the targets we'll predict
def generate_random_data(high):
    assert high <= 100
    X = np.arange(100)
    y = X + rng.uniform(0, high, 100)
    return X, y

# This function draws the calculated model and data
def plot_random_correlated_features(X, y, slope, intercept, subplot, r2_score):
    # This draws the data points
    subplot.scatter(X, y, label="random data", color="#0000ff77")
    # This draws the line which shows the tendency infered by the model
    subplot.axline((0, intercept), slope=slope, color="#00ff00ff")
    subplot.text(0, 150, f"r² score: {r2_score:.02f}")
    subplot.legend(loc="upper right")

# This function splits the data into a train section and a test section.
# The train section is shown to the model and the test section is withheld and used to simulate the broader population you want to know something about.
def __split_data(X, y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=rng)
    return X_train, X_test, y_train, y_test

# This function fits a linear regression to the data given by X_train, the independent variables, and y_train, the dependent variable
def __linear_regression(X_train, y_train):
    # When you train a model you should always put some data aside for testing. The model should never have seen this test data.
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    return model

# This function combines the above functions
def fit_and_plot():
    # Prepare four (2x2) places to draw figures
    fig, axs = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(12, 12))
    # reshape to get the figure plots into a list with a more handy size (4x1)
    axs = axs.reshape((4, 1)).squeeze()
    for ax, high in zip(axs, [10, 30, 60, 80]):
        X, y = generate_random_data(high)
        # x and y must be reshaped in this toy example because the model expects an array where each element is an array of feature values.
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

        X_train, X_test, y_train, y_test = __split_data(X, y)
        model = __linear_regression(X_train, y_train)
        preds = model.predict(X_test)
        r2_score = sklearn.metrics.r2_score(y_test, preds)
        plot_random_correlated_features(X, y, model.coef_[0, 0], model.intercept_[0], ax, r2_score)
    plt.show()

fit_and_plot()


def split_data(dataframe, predictors, target, test_size=0.1):
    # Fill in the function here
    x = dataframe[predictors]
    y = dataframe[target]
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1, random_state=rng)
    return x_train, x_test, y_train, y_test

def single_feature_linear_regression(dataframe):
    # Fill in the function here
    x_train, x_test, y_train, y_test = split_data(dataframe, ["fat-total"], "calories", test_size=0.1)
    model = sklearn.linear_model.LinearRegression()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return sklearn.metrics.r2_score(y_test, preds)

def multi_feature_linear_regression(dataframe):
    # Fill in the function here
    x_train, x_test, y_train, y_test = split_data(dataframe, ["fat-total", "sodium", "carbohydrates", "cholesterol"], "calories")
    model = sklearn.linear_model.LinearRegression()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return sklearn.metrics.r2_score(y_test, preds)

# Fill out this cell with code to select data for the X and y inputs,
# split this data into train and test sets, fit a model on the training data ()
# and make predictions based on the test data. Score your predictions using sklearn.metrics.classification_report.
# Also examine the feature importances using the `feature_importances_` attribute of your random forest model
# Return a tuple with the trained model, your predictions and the real test set labels.
def random_forest_classification(dataframe):
    rng = np.random.RandomState(0)
    input_features = ["fat-total", "sodium", "carbohydrates", "cholesterol"]
    X_train, X_test, y_train, y_test = split_data(dataframe, input_features, "beverage-prep")
    model = sklearn.ensemble.RandomForestClassifier(random_state=rng)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    test_report = sklearn.metrics.classification_report(y_test, preds)
    print(test_report)
    feature_importances = "\n\t".join(f"{feature}: {importance:.2f}" for feature, importance in zip(input_features, model.feature_importances_))
    print(f"Feature importances:\n\t{feature_importances}")
    return model, preds, y_test