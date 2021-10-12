# Written by Stuart Ryan, October 2021

import random
# Python packages for Natrual Language Processing
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Python package for holding data
import pandas as pd

# Sklearn packages for model creation and testing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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
Note about SVC vs SGD taken from documentation for SVC model (Initially unsure if dataset is large enough for SGD to be better): 
For large datasets consider using LinearSVC or SGDClassifier instead, possibly after a Nystroem transformer.
'''
models_of_interest = [
(SGDClassifier(random_state=1, max_iter=5, tol=None), {'clf__penalty': ['l1', 'l2'], 'clf__alpha': [.00001, .0001, .001, .01], 'clf__loss': ['log', 'hinge']}),
(tree.DecisionTreeClassifier(random_state=1), {"clf__criterion": ["gini", 'entropy'], 'clf__max_depth': range(2, 10), 'clf__min_samples_split': range(2,10), 'clf__min_samples_leaf': range(1,10)}),
(MultinomialNB(), {}),  
] 

# Need to inclde logistic regression, SVM, k nearest, Random forest, gradient boosting

def model_creation(classifier_params):
    pass

###Model creation and testing (Consider using different classifiers as that will give us performances to compare that we can discuss in the paper)
### One approach would be to create an overarching function that is fed in a list of dictionaries where each dictionary is a parameter grid corresponding to a ML classification model so we can PROPERLY COMPARE THE PERFORMANCES OF DIFFERENT CLASSIFICATION MODELS THAT HAVE THEIR HYPERPARAMETERS OPTIMIZED

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
        
def text_model_creation():
    ### Text Data Collecting and Preprocessing ###
    texts_df = pd.read_csv(r"text_data\spam.csv", encoding="ISO-8859-1")
    texts_df.dropna(1, inplace=True) #Removing excess variable columns
    texts_df['v2'] = texts_df['v2'].apply(clean_text)
    texts_df['v1'] = texts_df['v1'].apply(convert_labels)
    print(texts_df)

    ### Text Data Model Creation ###
    # 70% of the data allocated to training
    X = texts_df.v2
    y = texts_df.v1
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
    param_grid = [{'vect__ngram_range': [(1,1), (1, 2), (2,2)],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'vect__use_idf':[False, True],
                'clf__penalty': ['l1', 'l2'],
                'clf__alpha': [.00001, .0001, .001, .01],
                'clf__loss': ['log', 'hinge']
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

    '''
    The following was the best parameter set found using the above grid search:
    Best parameter set: {'clf__alpha': 1e-05, 'clf__loss': 'log', 'clf__penalty': 'l2', 'vect__ngram_range': (1, 2), 'vect__tokenizer': <function tokenizer_porter at 0x0000023946B04438>, 'vect__use_idf': True} 
    CV Accuracy: 0.984
    Test Accuracy: 0.983
    '''

# Function to read in email files taken from the kaggle site for the data source and written by Wessel Van Lit
def load_email(is_spam, filename):
    directory = "../input/hamnspam/spam" if is_spam else "../input/hamnspam/ham"
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

# Function to make sure we shuffle our emails the same way each time we run the program
def seedSetter(): 
	return 0.3

def email_model_creation():
    ### Email Data Collecting and Preprocessing ###
    os.listdir('../input/hamnspam/')
    ham_filenames = [name for name in sorted(os.listdir('../input/hamnspam/ham')) if len(name) > 20]
    spam_filenames = [name for name in sorted(os.listdir('../input/hamnspam/spam')) if len(name) > 20]
    ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
    spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
    # The above lines of code were written by the creator of the dataset Wessel Van Lit

    # Making list of tuples containing all emails and shuffling in preparation for model training
    all_emails = [(email, 0) for email in ham_emails] + [(email, 1) for email in spam_emails]
    random.shuffle(all_emails, seedSetter)

    # Creating tuple lists for email subjects, email content, and email addresses to be fed into dataframes
    all_subjects = [(markedMail[0]['Subject'], markedMail[1]) for markedMail in all_emails]
    all_addresses = [(markedMail[0]['From'], markedMail[1]) for markedMail in all_emails]
    all_bodies = [(markedMail[0].get_content(), markedMail[1]) for markedMail in all_emails]

    # Creating corresponding dataframes
    subject_df = pd.Dataframe(data=all_subjects, columns=["subject", "is_spam"])
    address_df = pd.Dataframe(data=all_addresses, columns=["address", "is_spam"])
    body_df = pd.Dataframe(data=all_bodies, columns=["body", "is_spam"])
    

email_model_creation()



### Should turn all the preprocessing into one function that I can run the dataframes through

# Variable holds tuples with var 0 being the classifier in use, and var 1 being the associated parameter grid
