
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
import textstat as ts
from numpy.random import seed
seed(100)
nltk.download('stopwords')


data_path = "/Users/zahra/Desktop/News/"
true_news = pd.read_csv(data_path + "True.csv")
fake_news = pd.read_csv(data_path + "Fake.csv")
true_news["label"] = "True"
fake_news["label"] = "Fake"
all_news = true_news.append(fake_news, ignore_index = True)
del all_news['subject']
del all_news['date']
del all_news['title']

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re

def remove_punctuations(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


def remove_nums(text):
    text = ''.join([i for i in text if not i.isdigit()])         
    return text


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F" 
                           u"\U0001F300-\U0001F5FF"  
                           u"\U0001F680-\U0001F6FF"  
                           u"\U0001F1E0-\U0001F1FF"  
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


stop_words = set(stopwords.words('english'))


def clean_stopwords(text):
    res = [w for w in text.split() if not w in stop_words]
    res_string = " ".join(str(x) for x in res)
    return res_string
    

all_news_processed = all_news.copy()

all_news_processed["text"] = all_news_processed["text"].apply(lambda x: remove_punctuations(x))
all_news_processed["text"] = all_news_processed["text"].apply(lambda x: remove_nums(x))
all_news_processed["text"] = all_news_processed["text"].apply(lambda x: remove_URL(x))
all_news_processed["text"] = all_news_processed["text"].apply(lambda x: remove_html(x))
all_news_processed["text"] = all_news_processed["text"].apply(lambda x: remove_emoji(x))
all_news_processed["text"] = all_news_processed["text"].apply(lambda x: clean_stopwords(x))

all_news_processed = all_news_processed.sample(frac = 1).reset_index(drop=True).reset_index(drop = True)

train_X = all_news_processed.loc[:37000, "text"].values
train_Y = all_news_processed.loc[:37000, "label"].values
validation_X = all_news_processed.loc[37000:, "text"].values
validation_Y = all_news_processed.loc[37000:, "label"].values


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_X)
validation_vectors = vectorizer.transform(validation_X)

from sklearn.naive_bayes import MultinomialNB
import time
start = time.time()
mnb_classifier = MultinomialNB().fit(train_vectors, train_Y)
end = time.time()

from  sklearn.metrics  import accuracy_score
mnb_predicted = mnb_classifier.predict(validation_vectors)
print("Accuracy of MNB: {}".format(accuracy_score(validation_Y, mnb_predicted)))

from sklearn.metrics import classification_report, confusion_matrix, roc_curve

print("A Report Of My Model:")

print("Using MNB Classifier")
print(classification_report(validation_Y, mnb_predicted))



