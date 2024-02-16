#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import pickle
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression, RidgeCV
import seaborn as sns
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, FunctionTransformer, PolynomialFeatures,     LabelEncoder, MultiLabelBinarizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import metrics
from sklearn.feature_selection import SelectKBest , f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
import time





def Select(df,Y):
    from sklearn.feature_selection import mutual_info_classif
    import matplotlib.pyplot as plt
    df = df.fillna(0)
    imp = mutual_info_classif(df, y_train)
    # Create a series of feature importances and plot them
    feat = pd.Series(imp, df.columns)
    feat.plot(kind='barh', color='teal')
    #plt.show()


def NumericPreProcessing(df):
    # removing zeros
    df[['budget', 'viewercount', 'revenue', 'runtime', 'vote_count', 'release_date']]         = df[['budget', 'viewercount', 'revenue', 'runtime', 'vote_count', 'release_date']]         .replace(0, df[['budget', 'viewercount', 'revenue', 'runtime', 'vote_count', 'release_date']].mean())

    # scaling
    cols_to_scale = ['viewercount', 'runtime', 'vote_count', 'budget', 'revenue', 'release_date']
    df[cols_to_scale] = (df[cols_to_scale] - df[cols_to_scale].min()) / (
                df[cols_to_scale].max() - df[cols_to_scale].min())
    cols = df[cols_to_scale]

    return cols


def colsToNormalize(df):
    data = pd.DataFrame()
    df['homepage'] = df['homepage'].notnull().astype(int)
    data = pd.concat([data, df['homepage']], axis=1)

     #label encoding
    df['status'] = df['status'].notnull().astype(int)
    data = pd.concat([data, df['status']], axis = 1)

    le = preprocessing.LabelEncoder()
    df['original_language'] = le.fit_transform(df['original_language'])
    df['original_language'] = le.fit_transform(df.original_language.values)

    data = pd.concat([data, df['original_language']], axis=1)
    return data


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    if pd.isna(text):
        text = ''
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize the text into words
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    # Perform lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join the list of tokens into a single string
    text = ' '.join(tokens)
    return text



mlb_key = MultiLabelBinarizer()
mlbgen = MultiLabelBinarizer()
mlbpc = MultiLabelBinarizer()
mlb2 = MultiLabelBinarizer()
mlbsp = MultiLabelBinarizer()


def dictionaryPreprocessing(df):
    data = pd.DataFrame()

    # Convert the 'genres' column to a list of genre names
    df['genres'] = df['genres'].apply(lambda x: ','.join([d['name'] for d in eval(x)]))

    df['keywords'] = df['keywords'].apply(lambda x: ','.join([d['name'] for d in eval(x)]))

    df['production_companies'] = df['production_companies'].apply(lambda x: ','.join([d['name'] for d in eval(x)]))

    # Convert the 'production_countries' column to a list of country names
    df['production_countries'] = df['production_countries'].apply(
        lambda x: ','.join([d['iso_3166_1'] for d in eval(x)]))

    # Convert the 'spoken_languages' column to a list of language names
    df['spoken_languages'] = df['spoken_languages'].apply(lambda x: ','.join([d['name'] for d in eval(x)]))

    mlb_key.fit(df['keywords'])
    transkey = mlb_key.transform(df['keywords'])
    transskey = pd.DataFrame(transkey)
    data = pd.concat([data, transskey], axis=1)

    mlbgen.fit(df['genres'])
    transgen = mlbgen.transform(df['genres'])
    transsgen = pd.DataFrame(transgen)
    data = pd.concat([data, transsgen], axis=1)

    mlbpc.fit(df['production_countries'])
    transpc = mlbpc.transform(df['production_countries'])
    transspc = pd.DataFrame(transpc)
    data = pd.concat([data, transspc], axis=1)

    mlbsp.fit(df['spoken_languages'])
    transp = mlbsp.transform(df['spoken_languages'])
    transsp = pd.DataFrame(transp)
    data = pd.concat([data, transsp], axis=1)
    #

    mlb2.fit(df['production_companies'])
    trans2 = mlb2.transform(df['production_companies'])
    transs2 = pd.DataFrame(trans2)
    data = pd.concat([data, transs2], axis=1)

    return data


def dictionaryPreprocessing_test(df):
    data = pd.DataFrame()

    # Convert the 'genres' column to a list of genre names
    df['genres'] = df['genres'].apply(lambda x: ','.join([d['name'] for d in eval(x)]))

    df['keywords'] = df['keywords'].apply(lambda x: ','.join([d['name'] for d in eval(x)]))

    df['production_companies'] = df['production_companies'].apply(lambda x: ','.join([d['name'] for d in eval(x)]))

    # Convert the 'production_countries' column to a list of country names
    df['production_countries'] = df['production_countries'].apply(
        lambda x: ','.join([d['iso_3166_1'] for d in eval(x)]))

    # Convert the 'spoken_languages' column to a list of language names
    df['spoken_languages'] = df['spoken_languages'].apply(lambda x: ','.join([d['name'] for d in eval(x)]))

    transkey = mlb_key.transform(df['keywords'])
    transskey = pd.DataFrame(transkey)
    data = pd.concat([data, transskey], axis=1)

    transgen = mlbgen.transform(df['genres'])
    transsgen = pd.DataFrame(transgen)
    data = pd.concat([data, transsgen], axis=1)

    transpc = mlbpc.transform(df['production_countries'])
    transspc = pd.DataFrame(transpc)
    data = pd.concat([data, transspc], axis=1)

    transp = mlbsp.transform(df['spoken_languages'])
    transsp = pd.DataFrame(transp)
    data = pd.concat([data, transsp], axis=1)

    trans2 = mlb2.transform(df['production_companies'])
    transs2 = pd.DataFrame(trans2)
    data = pd.concat([data, transs2], axis=1)

    return data


df = pd.read_csv('C:/Users/Nouran/OneDrive/Desktop/movies-classification-dataset.csv', parse_dates=['release_date'],
                 dayfirst=True, engine='python')
df.drop_duplicates(inplace=True)

df['release_date'] = pd.to_datetime(df['release_date'])
reference_date = datetime.now()
df['release_date'] = (reference_date - df['release_date']).dt.days
#vectorizer = TfidfVectorizer()
X_train= df.iloc[:, :-1]
y_train= df['Rate']

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=False, random_state=1000)








# In[2]:


df_test = pd.read_csv('C:/Users/Nouran/OneDrive/Desktop/day1/M2/movies-tas-test.csv', parse_dates=['release_date'],
                 dayfirst=True, engine='python')
df_test.drop_duplicates(inplace=True)

df_test['release_date'] = pd.to_datetime(df_test['release_date'])
reference_date = datetime.now()
df_test['release_date'] = (reference_date - df_test['release_date']).dt.days
X_test= df.iloc[:, :-1]
y_test= df['Rate']






# In[3]:


X_train['overview'] = X_train['overview'].apply(preprocess_text)
X_test['overview'] = X_test['overview'].apply(preprocess_text)

X_train['title']= X_train['title'].apply(preprocess_text)
X_test['title']=  X_test['title'].apply(preprocess_text)

X_train['tagline']= X_train['tagline'].apply(preprocess_text)
X_test['tagline']=  X_test['tagline'].apply(preprocess_text)

X_train['original_title']= X_train['original_title'].apply(preprocess_text)
X_test['original_title']=  X_test['original_title'].apply(preprocess_text)






Tag_vectorizer = TfidfVectorizer()

tagtrain = Tag_vectorizer.fit_transform(X_train['tagline'])
tag_train= pd.DataFrame(tagtrain.toarray(),columns=Tag_vectorizer.get_feature_names_out())

tag_test = Tag_vectorizer.transform(X_test['tagline'])
tag_test= pd.DataFrame(tag_test.toarray(),columns=Tag_vectorizer.get_feature_names_out())





OTitle_vectorizer = TfidfVectorizer()
Otitle_train = OTitle_vectorizer.fit_transform(X_train['original_title'])
Otitle_train= pd.DataFrame(Otitle_train.toarray(),columns=OTitle_vectorizer.get_feature_names_out())
Otitle_test = OTitle_vectorizer.transform(X_test['original_title'])
Otitle_test= pd.DataFrame(Otitle_test.toarray(),columns=OTitle_vectorizer.get_feature_names_out())





Title_vectorizer = TfidfVectorizer()
title_train = Title_vectorizer.fit_transform(X_train['title'])
title_train= pd.DataFrame(title_train.toarray(),columns=Title_vectorizer.get_feature_names_out())
title_test = Title_vectorizer.transform(X_test['title'])
title_test= pd.DataFrame(title_test.toarray(),columns=Title_vectorizer.get_feature_names_out())






vectorizer = TfidfVectorizer()
nlp_train = vectorizer.fit_transform(X_train['overview'])
nlp_train= pd.DataFrame(nlp_train.toarray(),columns=vectorizer.get_feature_names_out())
nlp_test = vectorizer.transform(X_test['overview'])
nlp_test= pd.DataFrame(nlp_test.toarray(),columns=vectorizer.get_feature_names_out())

selector1 = VarianceThreshold(threshold=0.0001)
X_train_selected_Num = selector1.fit_transform(NumericPreProcessing(X_train))
X_test_selected_Num = selector1.transform(NumericPreProcessing(X_test))
# print(X_train_selected.shape)
# print(X_test_selected.shape)
X_train_selected_Num = pd.DataFrame(X_train_selected_Num)
X_test_selected_Num = pd.DataFrame(X_test_selected_Num)
##

dataAbdallah = pd.DataFrame()
dataAbdallah = pd.concat([dataAbdallah, colsToNormalize(X_train)], axis=1)
print("norm")
print(dataAbdallah.shape)
dataAbdallah = dataAbdallah.reset_index(drop=True, inplace=True)

dataAbdallah = pd.concat([dataAbdallah, X_train_selected_Num], axis=1)

print("num")
print(dataAbdallah.shape)
# dataAbdallah = dataAbdallah.reset_index(drop=True)
### fea
print("fea")
selector = VarianceThreshold(threshold=0.08)
X_train_selected = selector.fit_transform(dictionaryPreprocessing(X_train))
X_test_selected = selector.transform(dictionaryPreprocessing_test(X_test))
# print(X_train_selected.shape)
# print(X_test_selected.shape)
X_train_selected = pd.DataFrame(X_train_selected)
X_test_selected = pd.DataFrame(X_test_selected)
####

###
dataAbdallah = pd.concat([dataAbdallah, X_train_selected], axis=1)
print("dict")
print(dataAbdallah.shape)

dataAbdallah = pd.concat([dataAbdallah, nlp_train], axis=1)
print("nlp")
print(dataAbdallah.shape)

dataAbdallah = dataAbdallah.fillna(0)
################################


dataNaguib = pd.DataFrame()
dataNaguib = pd.concat([dataNaguib, colsToNormalize(X_test)], axis=1)
print("norm")
print(dataNaguib.shape)
dataNaguib = dataNaguib.reset_index(drop=True, inplace=True)

dataNaguib = pd.concat([dataNaguib, X_test_selected_Num], axis=1)
print("num")
print(dataNaguib.shape)
dataNaguib = dataNaguib.reset_index(drop=True)

dataNaguib = pd.concat([dataNaguib, X_test_selected], axis=1)
print("dict")
print(dataNaguib.shape)

dataNaguib = pd.concat([dataNaguib, nlp_test], axis=1)
print("nlp")
print(dataNaguib.shape)

dataNaguib = dataNaguib.fillna(0)








dataAbdallah.columns = dataAbdallah.columns.astype(str)
dataNaguib.columns = dataNaguib.columns.astype(str)


def svm_model():
    filename = 'SVM_model.sav'
    if os.path.exists('SVM_model.sav'):
        print("Loading Trained Model")
        loaded_model = pickle.load(open(filename, 'rb'))
        prediction = loaded_model.predict(dataNaguib)
        print("SVM accuracy: ", accuracy_score(y_test, prediction))
    else:
        print("Creating and training a new model")
        print("Model training started")
        print("SVM")
        svc_model = LinearSVC(C=0.1, max_iter=1000)
        start_train = time.time()
        svc_model.fit(dataAbdallah, y_train)
        end_train = time.time()
        # Calculate training time
        training_time = end_train - start_train
        pickle.dump(svc_model, open(filename, 'wb'))
        start_test = time.time()
        prediction = svc_model.predict(dataNaguib)
        end_test = time.time()
        # Calculate testing time
        testing_time = end_test - start_test
        print("trianing time:", training_time)
        print("test time:", testing_time)
        print("SVM accuracy: ", accuracy_score(y_test, prediction))

        # Create 3 bar graphs
        fig, plot = plt.subplots(1, 3, figsize=(10, 4))

        # Bar graph for total training time
        plot[0].bar(0, training_time)
        plot[0].set_xticks([])
        plot[0].set_title("Total Training Time(sec)")

        # Bar graph for total testing time
        plot[1].bar(0, testing_time)
        plot[1].set_xticks([])
        plot[1].set_title("Total Testing Time(sec)")

        # Bar graph for SMV accuracy
        plot[2].bar(0, accuracy_score(y_test, prediction) * 100)
        plot[2].set_xticks([])
        plot[2].set_title("SVM accuracy(%)")


def log_model():
    filename = 'LOG_model.sav'
    if os.path.exists('LOG_model.sav'):
        print("Loading Trained Model")
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(dataNaguib)
        print('accuracy: ', accuracy_score(y_test, y_pred))
    else:
        print("Creating and training a new model")
        print("Model training started")
        print("LOGISTIC")
        # lr = LogisticRegression(penalty='l2', solver='sag', max_iter=1000, C=0.01)
        lr = LogisticRegression(penalty='l2', solver='saga', max_iter=1000, C=1.0)
        # lr = LogisticRegression(penalty='l2', solver='saga', max_iter=1000, C=0.5)
        # lr = LogisticRegression(penalty='l2', solver='newton-cg', max_iter=1000, C=0.01)
        # lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, C=0.1)
        start_train = time.time()
        lr.fit(dataAbdallah, y_train)
        end_train = time.time()
        # Calculate training time
        training_time = end_train - start_train
        pickle.dump(lr, open(filename, 'wb'))
        start_test = time.time()
        y_pred = lr.predict(dataNaguib)
        end_test = time.time()
        # Calculate testing time
        testing_time = end_test - start_test

        print("trianing time:", training_time)
        print("test time:", testing_time)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

        # Create 3 bar graphs
        fig, plot = plt.subplots(1, 3, figsize=(10, 4))

        # Bar graph for total training time
        plot[0].bar(0, training_time)
        plot[0].set_xticks([])
        plot[0].set_title("Total Training Time(sec)")

        # Bar graph for total test time
        plot[1].bar(0, testing_time)
        plot[1].set_xticks([])
        plot[1].set_title("Total Test Time(sec)")

        # Bar graph for logistic regression accuracy
        plot[2].bar(0, metrics.accuracy_score(y_test, y_pred) * 100)
        plot[2].set_xticks([])
        plot[2].set_title("Logistic reg accuracy(%)")


def tree_model():
    filename = 'TREE_model.sav'
    if os.path.exists('TREE_model.sav'):
        print("Loading Trained Model")
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(dataNaguib)
        print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred))
    else:
        print("Creating and training a new model")
        print("Model training started")
        print("Decison Tree")
        # clf = DecisionTreeClassifier()
        # clf = DecisionTreeClassifier(criterion = "entropy")
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)
        start_train = time.time()
        clf = clf.fit(dataAbdallah, y_train)
        end_train = time.time()
        # Calculate training time
        training_time = end_train - start_train
        pickle.dump(clf, open(filename, 'wb'))
        start_test = time.time()
        y_pred = clf.predict(dataNaguib)
        end_test = time.time()
        # Calculate testing time
        testing_time = end_test - start_test

        print("trianing time:", training_time)
        print("test time:", testing_time)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

        # Create 3 bar graphs
        fig, plot = plt.subplots(1, 3, figsize=(10, 4))

        # Bar graph for total training time
        plot[0].bar(0, training_time)
        plot[0].set_xticks([])
        plot[0].set_title("Total Training Time(sec)")

        # Bar graph for total test time
        plot[1].bar(0, testing_time)
        plot[1].set_xticks([])
        plot[1].set_title("Total Test Time(sec)")

        # Bar graph for decision tree accuracy
        plot[2].bar(0, metrics.accuracy_score(y_test, y_pred) * 100)
        plot[2].set_xticks([])
        plot[2].set_title("Decision Tree accuracy(%)")










svm_model()


log_model()



tree_model()

