# Written by Stuart Ryan, October 2021
import random
import os

# Python packages for Natrual Language Processing
import re
import nltk
from nltk.corpus.reader.chasen import test
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams
from itertools import groupby

# Python package for holding data
import pandas as pd

# Sklearn packages for model creation and testing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV

# Python package for parsing data from email files
import email

# Python package for saving machine learning models for later use
import pickle

'''
Email data gotten from: https://www.kaggle.com/veleon/ham-and-spam-dataset
Text data gotten from: https://www.kaggle.com/uciml/sms-spam-collection-dataset5
'''
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

#Will need to have an if statement to ignore grid search as naive bayes models have no hyperparameters
'''
Notes:
- SVC vs SGD taken from documentation for SVC model (Initially unsure if dataset is large enough for SGD to be better): 
    For large datasets consider using LinearSVC or SGDClassifier instead, possibly after a Nystroem transformer.
- For the SGDClassifier, hinge loss is a linear SVM (which is why we test alternate kernels later)
- the p metric in kNeighborsClassifier has the following mapping: 1 - manhattan distance, 2 - euclidean distance, 3 - minkowski distance
'''
models_of_interest = [
(SGDClassifier(random_state=1), {'clf__penalty': ['l1', 'l2'], 'clf__alpha': [.00001, .0001, .001, .01], 'clf__loss': ['log', 'hinge']}),

(DecisionTreeClassifier(random_state=1), {'clf__criterion': ['gini', 'entropy'], 'clf__max_depth': range(2, 10), 
'clf__min_samples_split': range(2,10), 'clf__min_samples_leaf': range(1,10)}),

(LogisticRegression(random_state=1), {'clf__penalty': ['l1', 'l2'], 'clf__solver': ['liblinear', 'saga'], 'clf__C': [.001, .01, .1, 1.0]}),

(SVC(random_state=1), {'clf__C': [.001, .01, .1, 1.0], 'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}),

(RandomForestClassifier(random_state=1), {'clf__criterion': ['gini', 'entropy'], 'clf__n_estimators': range(50, 175, 25), 
'clf__max_depth': range(2, 10, 2), 'clf__min_samples_split': range(2,10, 2), 'clf__min_samples_leaf': range(1,10)}),

(GradientBoostingClassifier(random_state=1), {'clf__criterion': ['friedman_mse', 'squared_error'], 
'clf__n_estimators': range(50, 175, 25), 'clf__loss': ['deviance', 'exponential'], 'clf__learning_rate': [0.01, 0.075, 0.1, 0.25, 0.2]}),

(KNeighborsClassifier(), {'clf__n_neighbors': range(1,10), 'clf__leaf_size': range(20, 40), 
'clf__p': [1, 2, 3], 'clf__weights': ['uniform', 'distance']}),

(MultinomialNB(), {})] 

# This function gets rid of html tags as well as excess whitespace between paragraphs
def clean_body(text):
    text = text.lower() # lowercase text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text

    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    
    text =  ' '.join(text.split())
    return text

# This function cleans addresses. Tested the validity of this with an SGD classifier and found the following improvements:
# Without cleaning: CV Accuracy: 0.884, Test Accuracy: 0.879
# With cleaning: CV Accuracy: 0.944 Test Accuracy: 0.963

'''
Breakdown email address into sections
1. Split things around the ., add num .'s found into tokens
2. Split around @, add num @'s (one token containing num @s) found into tokens
'''
def clean_address(text):
    text = text.lower() # lowercase text
    newString = text[0]
    for cIndex in range(1, len(text)):
        if text[cIndex] == ".":
            if text[cIndex-1] == ".":
                newString += "."
            else:
                newString += " ."
        elif text[cIndex] == "@":
            if text[cIndex-1] == "@":
                newString += "@"
            else:
                newString += " @"
        else:
            if text[cIndex-1] == "." or text[cIndex-1] == "@":
                newString += " " + text[cIndex]
            else:
                newString += text[cIndex]
    #print(result)
    return newString

'''
def clean_address(text):
    text = text.lower() # lowercase text
    text = BAD_SYMBOLS_RE.sub(' ', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
'''

def clean_text(text):
    emojis = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "])")

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                        text)

    text = text.lower() # lowercase text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', '')) # getting rid of emoticons
    text = re.sub(emojis, r' \1 ', text) # getting rid of emojis
    return text

def convert_labels(text):
    if text == "spam":
        return 1
    else:
        return 0

porter = PorterStemmer()
def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# This functiom takes in a classifier model, parameter set for hyperparamater tuning, and test / train data
# and outputs the following info in a list: Best Parameter Set, Cross Val Accuracy, Test Accuracy
def tune_hyper(clf, clf_params, X_train, X_test, y_train, y_test):
    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)
    '''
    Parameter choices:
    TfIdfVectorizer:
        - ngram_range: Chose to consider 1. unigrams and 2. unigrams with bigrams for text messages as texts are short and often times contain few keywords
        - tokenizer: Testing to see if stemming would be beneficial given the informal nature of words in texts (slang words)
    '''
    param_grid = [{**{'vect__ngram_range': [(1,1), (1, 2), (2,2)],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'vect__use_idf':[False, True],
                }, **clf_params}]

    lr_tfidf = Pipeline([('vect', tfidf),
                        ('clf', clf)])

    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                            scoring='accuracy',
                            cv=5,
                            verbose=2,
                            n_jobs=-1) # Utilizes all cores on machine to speed up grid search

    gs_lr_tfidf.fit(X_train, y_train)

    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

    bestModel = gs_lr_tfidf.best_estimator_
    print('Test Accuracy: %.3f' % bestModel.score(X_test, y_test))
    return [gs_lr_tfidf.best_params_, gs_lr_tfidf.best_score_, bestModel.score(X_test, y_test)]

# Function to read in email files taken from the kaggle site for the data source and written by Wessel Van Lit and updated by Stuart Ryan
def load_email(is_spam, filename):
    directory = "email_data/hamnspam/spam" if is_spam else "email_data/hamnspam/ham"
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser().parse(f)

# Function to make sure we shuffle our emails the same way each time we run the program
def seedSetter(): 
    return 0.3

# Function to test different models with our different data sources
def optimal_model_searching(classifier_params):
    ### Text Data Collecting and Preprocessing ###
    texts_df = pd.read_csv(r"text_data/spam.csv", encoding="ISO-8859-1")
    texts_df.dropna(1, inplace=True) #Removing excess variable columns
    texts_df['v2'] = texts_df['v2'].apply(clean_text)
    texts_df['v1'] = texts_df['v1'].apply(convert_labels)

    ### Email Data Collecting and Preprocessing ###
    os.listdir('email_data/hamnspam/')
    ham_filenames = [name for name in sorted(os.listdir('email_data/hamnspam/ham')) if len(name) > 20]
    spam_filenames = [name for name in sorted(os.listdir('email_data/hamnspam/spam')) if len(name) > 20]
    ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
    spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
    # The above lines of code were written by the creator of the dataset Wessel Van Lit

    # Making list of tuples containing all emails and shuffling in preparation for model training
    all_emails = [(email, 0) for email in ham_emails] + [(email, 1) for email in spam_emails]
    random.shuffle(all_emails, seedSetter)

    # Creating tuple lists for email subjects, email content, and email addresses to be fed into dataframes
    all_subjects = [(markedMail[0]['Subject'], markedMail[1]) for markedMail in all_emails]
    all_addresses = [(markedMail[0]['From'], markedMail[1]) for markedMail in all_emails if str(markedMail[0]['From'])[:5] != "=?iso"]
    all_bodies = [(markedMail[0].get_payload(), markedMail[1]) for markedMail in all_emails]
    
    #Some preprocessing work before text cleaning as we have straggler classes and objects alongside strings
    valid_subs = []
    for sub in all_subjects:
        if isinstance(sub[0], str):
            valid_subs.append(sub)

    valid_adds = []
    for add in all_addresses:
        if isinstance(add[0], str):
            valid_adds.append(add)

    valid_bods = []
    for bod in all_bodies:
        if isinstance(bod[0], str):
            valid_bods.append(bod)    

    # Creating corresponding dataframes
    subject_df = pd.DataFrame(data=valid_subs, columns=["subject", "is_spam"])
    address_df = pd.DataFrame(data=valid_adds, columns=["address", "is_spam"])
    body_df = pd.DataFrame(data=valid_bods, columns=["body", "is_spam"])

    subject_df['subject'] = subject_df['subject'].apply(clean_body)
    address_df['address'] = address_df['address'].apply(clean_address)
    body_df['body'] = body_df['body'].apply(clean_body)

    ### Text Data Model Creation ###
    # 70% of the data allocated to training
    X_text = texts_df.v2
    y_text = texts_df.v1
    X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.3, random_state = 1)

    ### Email Data Model Creation ###
    # Subject
    X_email_subj = subject_df.subject
    y_email_subj = subject_df.is_spam
    X_train_email_subj, X_test_email_subj, y_train_email_subj, y_test_email_subj = train_test_split(X_email_subj, y_email_subj, test_size=0.3, random_state = 1)

    # Address
    X_email_add = address_df.address
    y_email_add = address_df.is_spam
    X_train_email_add, X_test_email_add, y_train_email_add, y_test_email_add = train_test_split(X_email_add, y_email_add, test_size=0.3, random_state = 1)

    # Subject
    X_email_body = body_df.body
    y_email_body = body_df.is_spam
    X_train_email_body, X_test_email_body, y_train_email_body, y_test_email_body = train_test_split(X_email_body, y_email_body, test_size=0.3, random_state = 1)

    for moi in classifier_params:
        modelName = str(type(moi[0]).__name__)
        print("Testing model: " + modelName + " for text messages")
        # Text
        modelInfo = str(tune_hyper(moi[0], moi[1], X_train_text, X_test_text, y_train_text, y_test_text))
        f = open(modelName+".txt", "w")
        f.writelines("Text: ")
        f.writelines(modelInfo)
        f.writelines("\n")
        print("Testing model: " + modelName + " for email subjects")
        # Subject
        modelInfo = str(tune_hyper(moi[0], moi[1], X_train_email_subj, X_test_email_subj, y_train_email_subj, y_test_email_subj))
        f.writelines("Subject: ")
        f.writelines(modelInfo)
        f.writelines("\n")
        print("Testing model: " + modelName + " for email addresses")
        # Address
        modelInfo = str(tune_hyper(moi[0], moi[1], X_train_email_add, X_test_email_add, y_train_email_add, y_test_email_add))
        f.writelines("Address: ")
        f.writelines(modelInfo)
        f.writelines("\n")
        print("Testing model: " + modelName + " for email bodies")
        # Body
        modelInfo = str(tune_hyper(moi[0], moi[1], X_train_email_body, X_test_email_body, y_train_email_body, y_test_email_body))
        f.writelines("Body: ")
        f.writelines(modelInfo)
        f.writelines("")
        f.close()

# Function to save optimal models
def optimal_model_saving():
    ### Text Data Collecting and Preprocessing ###
    texts_df = pd.read_csv(r"text_data/spam.csv", encoding="ISO-8859-1")
    texts_df.dropna(1, inplace=True) #Removing excess variable columns
    texts_df['v2'] = texts_df['v2'].apply(clean_text)
    texts_df['v1'] = texts_df['v1'].apply(convert_labels)

    ### Email Data Collecting and Preprocessing ###
    os.listdir('email_data/hamnspam/')
    ham_filenames = [name for name in sorted(os.listdir('email_data/hamnspam/ham')) if len(name) > 20]
    spam_filenames = [name for name in sorted(os.listdir('email_data/hamnspam/spam')) if len(name) > 20]
    ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
    spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
    # The above lines of code were written by the creator of the dataset Wessel Van Lit

    # Making list of tuples containing all emails and shuffling in preparation for model training
    all_emails = [(email, 0) for email in ham_emails] + [(email, 1) for email in spam_emails]
    random.shuffle(all_emails, seedSetter)

    # Creating tuple lists for email subjects, email content, and email addresses to be fed into dataframes
    all_subjects = [(markedMail[0]['Subject'], markedMail[1]) for markedMail in all_emails]
    all_addresses = [(markedMail[0]['From'], markedMail[1]) for markedMail in all_emails if str(markedMail[0]['From'])[:5] != "=?iso"]
    all_bodies = [(markedMail[0].get_payload(), markedMail[1]) for markedMail in all_emails]
    
    #Some preprocessing work before text cleaning as we have straggler classes and objects alongside strings
    valid_subs = []
    for sub in all_subjects:
        if isinstance(sub[0], str):
            valid_subs.append(sub)

    valid_adds = []
    for add in all_addresses:
        if isinstance(add[0], str):
            valid_adds.append(add)

    valid_bods = []
    for bod in all_bodies:
        if isinstance(bod[0], str):
            valid_bods.append(bod)    

    # Creating corresponding dataframes
    subject_df = pd.DataFrame(data=valid_subs, columns=["subject", "is_spam"])
    address_df = pd.DataFrame(data=valid_adds, columns=["address", "is_spam"])
    body_df = pd.DataFrame(data=valid_bods, columns=["body", "is_spam"])

    subject_df['subject'] = subject_df['subject'].apply(clean_body)
    address_df['address'] = address_df['address'].apply(clean_address)
    body_df['body'] = body_df['body'].apply(clean_body)

    ### Text Data Model Creation ###
    # 70% of the data allocated to training
    X_text = texts_df.v2
    y_text = texts_df.v1
    X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.3, random_state = 1)

    ### Email Data Model Creation ###
    # Subject
    X_email_subj = subject_df.subject
    y_email_subj = subject_df.is_spam
    X_train_email_subj, X_test_email_subj, y_train_email_subj, y_test_email_subj = train_test_split(X_email_subj, y_email_subj, test_size=0.3, random_state = 1)

    # Address
    X_email_add = address_df.address
    y_email_add = address_df.is_spam
    X_train_email_add, X_test_email_add, y_train_email_add, y_test_email_add = train_test_split(X_email_add, y_email_add, test_size=0.3, random_state = 1)

    # Subject
    X_email_body = body_df.body
    y_email_body = body_df.is_spam
    X_train_email_body, X_test_email_body, y_train_email_body, y_test_email_body = train_test_split(X_email_body, y_email_body, test_size=0.3, random_state = 1)
    '''
    These models are created with the optimal hyperparameters found by calling the optimal_model_searching function.
    If optimal parameters are defaults when calling the models, they aren't explicitly defined below
    '''
    
    text_class_model = Pipeline([('vect', TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, tokenizer=tokenizer_porter)),
                        ('clf', CalibratedClassifierCV(SGDClassifier(random_state=1), cv=5))])

    subject_class_model = Pipeline([('vect', TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, tokenizer=tokenizer_porter)),
                        ('clf', CalibratedClassifierCV(SGDClassifier(random_state=1), cv=5))])

    address_class_model = Pipeline([('vect', TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, tokenizer=tokenizer, ngram_range=(1,2))),
                        ('clf', SVC(random_state=1, kernel='linear', probability=True))])

    body_class_model = Pipeline([('vect', TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, tokenizer=tokenizer, ngram_range=(1,2))),
                        ('clf', CalibratedClassifierCV(SGDClassifier(random_state=1, alpha=.00001), cv=5))])

    text_class_model.fit(X_train_text, y_train_text)
    pickle.dump(text_class_model, open("text_class_model.sav", 'wb'))

    subject_class_model.fit(X_train_email_subj, y_train_email_subj)
    pickle.dump(subject_class_model, open("subject_class_model.sav", 'wb'))

    address_class_model.fit(X_train_email_add, y_train_email_add)
    pickle.dump(address_class_model, open("address_class_model.sav", 'wb'))

    body_class_model.fit(X_train_email_body, y_train_email_body)
    pickle.dump(body_class_model, open("body_class_model.sav", 'wb'))

#optimal_model_saving()

# Attempting to wrap text class model in calibrated classifier to get a prediction
def prediction_probability_discovery():
    ### Text Data Collecting and Preprocessing ###
    texts_df = pd.read_csv(r"text_data/spam.csv", encoding="ISO-8859-1")
    texts_df.dropna(1, inplace=True) #Removing excess variable columns
    texts_df['v2'] = texts_df['v2'].apply(clean_text)
    texts_df['v1'] = texts_df['v1'].apply(convert_labels)

    ### Text Data Model Creation ###
    # 70% of the data allocated to training
    X_text = texts_df.v2
    y_text = texts_df.v1
    X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.3, random_state = 1)

    #This is a new line. Instead of using the SGD classifier in our pipeline, we wrap it in a calibrated classifier and then pass it to the pipeline 
    cal_model = CalibratedClassifierCV(SGDClassifier(random_state=1), cv=5)

    text_class_model = Pipeline([('vect', TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, tokenizer=tokenizer_porter)),
                        ('clf', cal_model)])

    text_class_model.fit(X_train_text, y_train_text)
    y_test_pred = text_class_model.predict_proba(["Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...", "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"])
    print("test with no prefitting:")
    print(y_test_pred)
    print()
