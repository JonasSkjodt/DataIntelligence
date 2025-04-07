#!curl -L https://www.gutenberg.org/files/44747/44747-0.txt -o stendhal-red-and-black.txt

# !pip install datasets nltk pandas scikit-learn

import collections

import nltk
# Download a dataset used for tokenization, i.e. splitting our sentences into parts.
nltk.download("punkt")
import numpy as np
import pandas as pd
# This module is only used for loading the data
import datasets

d = datasets.load_dataset("sentiment140", split="train", streaming=True, trust_remote_code=True)

df = pd.DataFrame(((r["text"], r["sentiment"]) for r in d.shuffle(buffer_size=5_000_000, seed=0).take(50000)), columns=["tweet", "sentiment"])
df["sentiment"] = df["sentiment"].map({0: "negative", 4: "positive"})

plot = df["sentiment"].value_counts().plot.bar()
plot.set_ylabel("count")
plot.set_title("Tweet sentiment counts")

# start exercise
def get_words(string):
    # Fill in the function to split the string into individual tokens. Return a list of tokens.
    import nltk
    tokenizer = nltk.tokenize.TweetTokenizer()
    w = tokenizer.tokenize(string)
    return w

def remove_ats(words):
    # Fill in the function to remove adressee indicators from the given list of tokens. Return the filtered list of tokens.
    return [w for w in words if w[0] != "@"]

def remove_links(words):
    # Fill in the function to remove links from the token list. Return the filtered list of tokens.
    return [w for w in words if w[:4] != "http"]

def lowercase_tokens(words):
    # Fill in this function to lowercase the input series
    return words.str.lower()

def get_stopwords(dataframe, column, n_most_common):
    # Fill in the function to return a list of the words most often occurring. Include only the words, not the counts.
    counter = collections.Counter()
    dataframe[column].str.lower().map(get_words).apply(counter.update)
    return [w for w, count in counter.most_common(n_most_common)]

def remove_stopwords(words, stopwords):
    # Fill in the function to remove stopwords from the list of tokens.
    return [w for w in words if w not in stopwords]

def clean_strings(dataframe, column_to_clean, cleaned_column):
    # Fill in the function to remove both @words and links from the given dataframe
    # lowercase, filters stopwords list, combining individual words into strings
    dataframe[cleaned_column] = dataframe[column_to_clean].str.lower().map(get_words).map(lambda words: remove_stopwords(words, stopwords)).map(remove_ats).map(remove_links).str.join(" ")
    return dataframe

import sklearn.feature_extraction.text
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes

def get_bag_of_words(dataframe, column):
    # Fill in the function to return a bag of words from the given dataframe
    count_vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    bow = count_vectorizer.fit_transform(dataframe[column])
    return count_vectorizer,bow

def categorize_sentiment(dataframe):
    # Fill in this function to use a classification model to predict sentiment.
    count_vectorizer,bow = get_bag_of_words(dataframe, "cleaned_tweet")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(bow, df["sentiment"], test_size=0.1, random_state=np.random.RandomState(0))
    model = sklearn.naive_bayes.MultinomialNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(sklearn.metrics.classification_report(y_test, preds))
    return y_test, preds


def predict_for_unknown_text(model, count_vectorizer):
    for sent in ["I'm very happy today", "I was pretty sad yesterday"]:
        print(sent, model.predict(count_vectorizer.transform([sent])))