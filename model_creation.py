# Written by Stuart Ryan, October 2021
import random
import os

# Python packages for Natrual Language Processing
import re
from nltk.corpus.reader.chasen import test
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

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

# Python package for parsing data from email files
import email

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
(DecisionTreeClassifier(random_state=1), {'clf__criterion': ['gini', 'entropy'], 'clf__max_depth': range(2, 10), 'clf__min_samples_split': range(2,10), 'clf__min_samples_leaf': range(1,10)}),
(LogisticRegression(random_state=1), {'clf__penalty': ['l1', 'l2'], 'clf__solver': ['liblinear', 'saga'], 'clf__C': [.001, .01, .1, 1.0]}),
(SVC(random_state=1), {'clf__C': [.001, .01, .1, 1.0], 'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}),
(RandomForestClassifier(random_state=1), {'clf__criterion': ['gini', 'entropy'], 'clf__n_estimators': range(50, 175, 25), 'clf__max_depth': range(2, 10), 'clf__min_samples_split': range(2,10), 'clf__min_samples_leaf': range(1,10)}),
(GradientBoostingClassifier(random_state=1), {'clf__criterion': ['friedman_mse', 'squared_error'], 'clf__n_estimators': range(50, 175, 25), 'clf__loss': ['deviance', 'exponential'], 'clf__learning_rate': [0.01, 0.075, 0.1, 0.25, 0.2]}),
(KNeighborsClassifier(), {'clf__n_neighbors': range(1,10), 'clf__leaf_size': range(20, 40), 'clf__p': [1, 2, 3], 'clf__weights': ['uniform', 'distance']}),
(MultinomialNB(), {})] 

# This function gets rid of html tags as well as excess whitespace between paragraphs
def clean_body(text):
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
def clean_address(text):
    add = text.split("<")[-1][:-1].replace("@", " ").replace(".", " ")
    return add

#print(clean_address(test))
#quit()
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

# Made this function to run multiple tests on the email addresses to see what different preprocessing techniques can improve performance 
def test_model_creation():
    os.listdir('email_data/hamnspam/')
    ham_filenames = [name for name in sorted(os.listdir('email_data/hamnspam/ham')) if len(name) > 20]
    spam_filenames = [name for name in sorted(os.listdir('email_data/hamnspam/spam')) if len(name) > 20]
    ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
    spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
    # The above lines of code were written by the creator of the dataset Wessel Van Lit

    # Making list of tuples containing all emails and shuffling in preparation for model training
    all_emails = [(email, 0) for email in ham_emails] + [(email, 1) for email in spam_emails]
    random.shuffle(all_emails, seedSetter)

    print(type(all_emails[0][0]['From']).__name__)
    # Creating tuple lists for email subjects, email content, and email addresses to be fed into dataframes
    # The "=?iso" piece is so that we get rid of weird address lines
    all_addresses = [(markedMail[0].get_payload(), markedMail[1]) for markedMail in all_emails if str(markedMail[0]['From'])[:5] != "=?iso"]

    nonStr = []
    for add in all_addresses:
        if not isinstance(add[0], str):
        #     all_addresses.remove(add)
            nonStr.append(add)
    print(len(nonStr))
    print(nonStr[0][0][1].get_payload())

    for add in all_addresses:
        nonStr.append(type(add[0]))
    print(set(nonStr))
    
    quit()
   
    # <email.header.Header object at 0x000002BBF4041548>
    # Creating corresponding dataframes
    address_df = pd.DataFrame(data=all_addresses, columns=["address", "is_spam"])
    address_df['address'] = address_df['address'].apply(clean_address)
    ### Text Data Model Creation ###
    # 70% of the data allocated to training
    X = address_df.address
    y = address_df.is_spam
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)

    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)

    # Look to expand the paramater grid to include as many relevant variables as possible
    '''
    Parameter choices:
    TfIdfVectorizer:
        - ngram_range: Chose to consider 1. unigrams and 2. unigrams with bigrams for text messages as texts are short and often times contain few keywords
        - tokenizer: Testing to see if stemming would be beneficial given the informal nature of words in texts (slang words)
    SGDClassifier:
        - clf_penalty: Chose to use both l1 and l2 because not sure whether the built in feature selection of l2 will be more important than l1's ability to ignore outliers in our dataset
        - clf_loss: Comparing log to hinge to compare if SVM is better for classification than logisitc regression
    '''
    param_grid = [{'vect__ngram_range': [(1, 2)],
                'vect__tokenizer': [tokenizer_porter],
                'vect__use_idf':[True],
                'clf__penalty': [ 'l2'],
                'clf__alpha': [.00001],
                'clf__loss': ['log']
                }
                ]

    lr_tfidf = Pipeline([('vect', tfidf),
                        ('clf', SGDClassifier(loss='log', random_state=1, max_iter=5, tol=None))])

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


# This functiom takes in a classifier model, parameter set for hyperparamater tuning, and test / train data
# and outputs the following info in a list: Best Parameter Set, Cross Val Accuracy, Test Accuracy
def text_model_creation(clf, clf_params, X_train, X_test, y_train, y_test):
    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)

    # Look to expand the paramater grid to include as many relevant variables as possible
    '''
    Parameter choices:
    TfIdfVectorizer:
        - ngram_range: Chose to consider 1. unigrams and 2. unigrams with bigrams for text messages as texts are short and often times contain few keywords
        - tokenizer: Testing to see if stemming would be beneficial given the informal nature of words in texts (slang words)
    '''
    param_grid = [{**{'vect__ngram_range': [(1,1), (1, 2), (2,2)],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'vect__use_idf':[False, True],
                'clf__penalty': ['l1', 'l2'],
                'clf__alpha': [.00001, .0001, .001, .01],
                'clf__loss': ['log', 'hinge']
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

    '''
    The following was the best parameter set found using the above grid search:
    Best parameter set: {'clf__alpha': 1e-05, 'clf__loss': 'log', 'clf__penalty': 'l2', 'vect__ngram_range': (1, 2), 'vect__tokenizer': <function tokenizer_porter at 0x0000023946B04438>, 'vect__use_idf': True} 
    CV Accuracy: 0.984
    Test Accuracy: 0.983
    '''

# Function to read in email files taken from the kaggle site for the data source and written by Wessel Van Lit and updated by Stuart Ryan
def load_email(is_spam, filename):
    directory = "email_data/hamnspam/spam" if is_spam else "email_data/hamnspam/ham"
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser().parse(f)

# Function to make sure we shuffle our emails the same way each time we run the program
def seedSetter(): 
	return 0.3

def email_model_creation():
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
    
    #Getting rid of header objects
    for add in all_addresses:
        if not isinstance(add[0], str):
            all_addresses.remove(add)
    # Creating corresponding dataframes
    subject_df = pd.DataFrame(data=all_subjects, columns=["subject", "is_spam"])
    address_df = pd.DataFrame(data=all_addresses, columns=["address", "is_spam"])
    body_df = pd.DataFrame(data=all_bodies, columns=["body", "is_spam"])

    address_df['address'] = address_df['address'].apply(clean_address)

    print(subject_df)
    print(address_df)
    print(body_df)

def model_creation(classifier_params):
    ### Text Data Collecting and Preprocessing ###
    texts_df = pd.read_csv(r"text_data\spam.csv", encoding="ISO-8859-1")
    texts_df.dropna(1, inplace=True) #Removing excess variable columns
    texts_df['v2'] = texts_df['v2'].apply(clean_text)
    texts_df['v1'] = texts_df['v1'].apply(convert_labels)
    print(texts_df)

    ### Text Data Model Creation ###
    # 70% of the data allocated to training
    X_text = texts_df.v2
    y_text = texts_df.v1
    X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.3, random_state = 1)

    for moi in models_of_interest:
        modelInfo = text_model_creation(moi[0], moi[1], X_train_text, X_test_text, y_train_text, y_test_text)
        text_f = open(str(type(moi[0]).__name__)+".txt", "w")
        text_f.writelines(modelInfo)
        text_f.close()

#email_model_creation()
test_model_creation()